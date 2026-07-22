#!/usr/bin/env bash
# Rebuild the isolated RoboCasa365 backend from pyproject.toml and uv.lock.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
env_dir="$repo_root/.robocasa365-venv"
canonical_assets="$repo_root/artifacts/robocasa365/merged-assets"
legacy_assets="$(dirname "$repo_root")/.cache/Xbotics-Hey-Robot/robocasa365/src/robocasa/robocasa/models/assets"
source_cache="$(dirname "$repo_root")/.cache/Xbotics-Hey-Robot/robocasa365/src"
lerobot_ref="cb73cf3ffa1cec60640a06b924c2174548ae2b1b"
robocasa_ref="56e355ccc64389dfc1b8a61a33b9127b975ba681"
robosuite_cache="$source_cache/../downloads/robosuite-git-cache"
robosuite_ref="aaa8b9b214ce8e77e82926d677b4d61d55e577ab"

if [[ "${1:-}" == "--recreate" ]]; then
  resolved_env="$(realpath -m "$env_dir")"
  if [[ "$resolved_env" != "$repo_root/.robocasa365-venv" ]]; then
    printf 'refusing to remove unexpected environment path: %s\n' "$resolved_env" >&2
    exit 2
  fi
  rm -rf -- "$resolved_env"
elif [[ -e "$env_dir" ]]; then
  printf '%s\n' 'environment already exists; pass --recreate to rebuild it' >&2
  exit 2
fi

if [[ ! -e "$canonical_assets" ]]; then
  if [[ ! -f "$legacy_assets/.robocasa-assets-ready" ]]; then
    printf 'RoboCasa365 assets are unavailable: %s\n' "$canonical_assets" >&2
    printf '%s\n' 'restore/download the merged assets before building the backend' >&2
    exit 2
  fi
  mkdir -p "$(dirname "$canonical_assets")"
  ln -s "$legacy_assets" "$canonical_assets"
fi

for required in \
  textures \
  generative_textures \
  fixtures \
  objects/lightwheel \
  .robocasa-assets-ready; do
  if [[ ! -e "$canonical_assets/$required" ]]; then
    printf 'RoboCasa365 asset is missing: %s\n' "$canonical_assets/$required" >&2
    exit 2
  fi
done

# A previously verified source cache may satisfy the immutable Git URLs when
# GitHub is slow or unavailable. Git still checks out the exact commits stored
# in uv.lock; a cache at any other revision is ignored.
git_rewrites=""
if [[ "$(git -C "$source_cache/lerobot" rev-parse HEAD 2>/dev/null || true)" == "$lerobot_ref" ]]; then
  git_rewrites="'url.file://$source_cache/lerobot/.insteadOf=https://github.com/huggingface/lerobot.git'"
fi
if [[ "$(git -C "$source_cache/robocasa" rev-parse HEAD 2>/dev/null || true)" == "$robocasa_ref" ]]; then
  if [[ -n "$git_rewrites" ]]; then
    git_rewrites+=" "
  fi
  git_rewrites+="'url.file://$source_cache/robocasa/.insteadOf=https://github.com/robocasa/robocasa.git'"
fi
if git -C "$robosuite_cache" cat-file -e "$robosuite_ref^{commit}" 2>/dev/null; then
  if [[ -n "$git_rewrites" ]]; then
    git_rewrites+=" "
  fi
  git_rewrites+="'url.file://$robosuite_cache/.insteadOf=https://github.com/ARISE-Initiative/robosuite.git'"
fi
if [[ -n "$git_rewrites" ]]; then
  export GIT_CONFIG_PARAMETERS="${GIT_CONFIG_PARAMETERS:-}$git_rewrites"
fi

UV_PROJECT_ENVIRONMENT="$env_dir" \
  uv sync --frozen --only-group robocasa365 --no-install-project

package_assets="$("$env_dir/bin/python" - <<'PY'
from importlib.metadata import distribution

print(distribution("robocasa").locate_file("robocasa/models/assets"))
PY
)"
packaged_backup="$package_assets.packaged"
if [[ -L "$package_assets" ]]; then
  unlink "$package_assets"
elif [[ -d "$package_assets" ]]; then
  if [[ -e "$packaged_backup" ]]; then
    printf 'unexpected packaged asset backup already exists: %s\n' "$packaged_backup" >&2
    exit 2
  fi
  mv "$package_assets" "$packaged_backup"
fi
ln -s "$canonical_assets" "$package_assets"

"$env_dir/bin/python" - <<'PY'
from importlib.metadata import version
from pathlib import Path

import grpc
import lerobot
import mujoco
import robocasa
import robosuite
import torch

assets = Path(robocasa.__file__).resolve().parent / "models" / "assets"
required = (
    assets / "textures",
    assets / "generative_textures",
    assets / "fixtures",
    assets / "objects" / "lightwheel",
    assets / ".robocasa-assets-ready",
)
if not all(path.exists() for path in required):
    raise SystemExit(f"RoboCasa365 asset validation failed: {assets}")
print(
    "RoboCasa365 environment ready:",
    f"torch={torch.__version__}",
    f"lerobot={version('lerobot')}",
    f"robocasa={version('robocasa')}",
    f"robosuite={version('robosuite')}",
    f"mujoco={mujoco.__version__}",
    f"grpc={grpc.__version__}",
    f"assets={assets.resolve()}",
)
PY
