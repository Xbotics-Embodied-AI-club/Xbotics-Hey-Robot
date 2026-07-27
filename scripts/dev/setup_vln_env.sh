#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
vln_env="${1:-${project_root}/.venv-vln}"

cd "$project_root"

# Large CUDA wheels are prone to transient proxy/CDN stalls. These remain
# caller-overridable while making the one-command setup reliable by default.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-5}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if [[ ! -f "third_party/InternNav/internnav/__init__.py" ]]; then
  git submodule update --init --recursive third_party/InternNav
fi

UV_PROJECT_ENVIRONMENT="$vln_env" \
  uv sync --frozen --no-default-groups --extra model-service --group vln

"$vln_env/bin/python" - <<'PY'
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

project_root = Path.cwd()
internnav_root = project_root / "third_party" / "InternNav"
sys.path.insert(0, str(internnav_root))

required = {
    "hey_robot": None,
    "internnav": None,
    "torch": "2.6.0+cu126",
    "torchvision": "0.21.0+cu126",
    "cv2": "4.10.0",
    "transformers": "4.51.0",
    "huggingface_hub": "0.33.4",
    "grpc": None,
}
failures: list[str] = []
for module_name, expected_version in required.items():
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        failures.append(f"missing module: {module_name}")
        continue
    module = __import__(module_name)
    actual_version = getattr(module, "__version__", None)
    if expected_version is not None and actual_version != expected_version:
        failures.append(
            f"{module_name} version mismatch: expected {expected_version}, got {actual_version}"
        )
    print(f"{module_name}: {actual_version or spec.origin}")

from internnav.model import get_config, get_policy

get_config("InternVLAN1_Policy")
get_policy("InternVLAN1_Policy")

if failures:
    raise SystemExit("VLN environment validation failed:\n- " + "\n- ".join(failures))
print("VLN environment is ready")
PY
