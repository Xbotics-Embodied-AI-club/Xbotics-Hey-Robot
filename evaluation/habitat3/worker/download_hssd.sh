#!/usr/bin/env bash
# Resumable, rate-limit-friendly HSSD downloader. Run with HF_TOKEN set.
set -u

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
hf_command="${HF_COMMAND:-/home/liber/.pyenv/shims/hf}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required" >&2
  exit 2
fi

data_root="${1:-artifacts/habitat3/data/scene_datasets/hssd-hab}"
mirror="${HF_ENDPOINT:-https://hf-mirror.com}"
delay_seconds=60
episodes_root="${HABITAT_EPISODES_ROOT:-artifacts/habitat3/data/datasets/hssd/rearrange}"

mkdir -p "$data_root"

# Download the small scene-instance manifests individually first.  Passing one
# filename to `hf download` avoids the full-repository tree listing that the
# public mirror rate-limits for HSSD.
while IFS= read -r scene_path; do
  until HF_ENDPOINT="$mirror" "$hf_command" download hssd/hssd-hab "$scene_path" \
    --repo-type dataset --local-dir "$data_root" --max-workers 1; do
    echo "$(date -Is) retrying scene manifest ${scene_path} in ${delay_seconds}s" >&2
    sleep "$delay_seconds"
  done
done < <(python - "$episodes_root/val/social_rearrange.json.gz" <<'PY'
import gzip
import json
import re
import sys

with gzip.open(sys.argv[1], "rt") as stream:
    episodes = json.load(stream)["episodes"]
for scene_id in sorted({episode["scene_id"] for episode in episodes}):
    print(scene_id.removeprefix("data/scene_datasets/hssd-hab/"))
PY
)

download_file() {
  local file_path="$1"
  until HF_ENDPOINT="$mirror" "$hf_command" download hssd/hssd-hab "$file_path" \
    --repo-type dataset --local-dir "$data_root" --max-workers 1; do
    echo "$(date -Is) retrying ${file_path} in ${delay_seconds}s" >&2
    sleep "$delay_seconds"
  done
}

# Get the exact assets referenced by the validation scenes.  This avoids
# enumerating tens of thousands of unrelated HSSD files before a smoke test.
while IFS= read -r asset_path; do
  download_file "$asset_path"
done < <(python - "$data_root/scenes-uncluttered" <<'PY'
import json
import re
import sys
from pathlib import Path

files = set()
for scene_path in Path(sys.argv[1]).glob("*.scene_instance.json"):
    scene = json.loads(scene_path.read_text())
    stage = scene["stage_instance"]["template_name"]
    files.update((f"{stage}.glb", f"{stage}.stage_config.json"))
    for obj in scene.get("object_instances", []):
        name = re.sub(r"_part_\d+$", "", obj["template_name"])
        if re.fullmatch(r"[0-9a-f]{40}", name):
            files.add(f"objects/{name[0]}/{name}.object_config.json")
print(*sorted(files), sep="\n")
PY
)

while IFS= read -r asset_path; do
  download_file "$asset_path"
done < <(python - "$data_root" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = set()
object_names = set()
for scene_path in (root / "scenes-uncluttered").glob("*.scene_instance.json"):
    scene = json.loads(scene_path.read_text())
    object_names.update(
        name
        for obj in scene.get("object_instances", [])
        if re.fullmatch(r"[0-9a-f]{40}", name := re.sub(r"_part_\d+$", "", obj["template_name"]))
    )
for name in object_names:
    config_path = root / "objects" / name[0] / f"{name}.object_config.json"
    config = json.loads(config_path.read_text())
    for key in ("render_asset", "collision_asset"):
        if config.get(key):
            files.add(str(config_path.parent.relative_to(sys.argv[1]) / config[key]))
print(*sorted(files), sep="\n")
PY
)

echo "$(date -Is) validation-scene minimal assets downloaded"
