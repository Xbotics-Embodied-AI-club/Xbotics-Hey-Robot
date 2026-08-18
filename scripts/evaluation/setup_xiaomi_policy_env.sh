#!/usr/bin/env bash
# Prepare the isolated Xiaomi-Robotics-1 model-service environment.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
venv_path="$repo_root/.venv-xiaomi"
xiaomi_root="$repo_root/artifacts/xiaomi-robotics-1"
xiaomi_repo="$xiaomi_root/Xiaomi-Robotics-1"
xiaomi_commit="6bc75afb791a1938750fe5fc0aee2b0f28cf87e2"

if [[ "${1:-}" == "--recreate" ]]; then
  if [[ "$venv_path" != "$repo_root/.venv-xiaomi" ]]; then
    printf '%s\n' 'refusing to remove an unexpected environment path' >&2
    exit 2
  fi
  rm -rf -- "$venv_path"
elif [[ $# -gt 0 ]]; then
  printf '%s\n' 'usage: setup_xiaomi_policy_env.sh [--recreate]' >&2
  exit 2
fi

mkdir -p "$xiaomi_root"
if [[ ! -d "$xiaomi_repo/.git" ]]; then
  git clone https://github.com/XiaomiRobotics/Xiaomi-Robotics-1.git "$xiaomi_repo"
fi
git -C "$xiaomi_repo" fetch --depth 1 origin "$xiaomi_commit"
git -C "$xiaomi_repo" checkout --detach "$xiaomi_commit"

if [[ ! -x "$venv_path/bin/python" ]]; then
  uv venv --python 3.12 --seed "$venv_path"
fi

"$venv_path/bin/pip" install \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
"$venv_path/bin/pip" install \
  -e "$repo_root[model-service]" \
  transformers==4.57.1 ninja
"$venv_path/bin/pip" install \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

"$venv_path/bin/python" - <<'PY'
import torch
import transformers

assert torch.__version__.startswith("2.8.0")
assert transformers.__version__ == "4.57.1"
print(f"Xiaomi policy environment ready: torch={torch.__version__}, transformers={transformers.__version__}")
PY

HF_HOME="$repo_root/artifacts/huggingface" \
  "$venv_path/bin/hf" download \
  XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365 \
  --local-dir "$xiaomi_root/checkpoints/Xiaomi-Robotics-1-RoboCasa365"
