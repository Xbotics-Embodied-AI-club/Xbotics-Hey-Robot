#!/usr/bin/env bash
# Start the isolated worker and Hey Robot with one ephemeral credential pair,
# then run the only supported full-system benchmark entrypoint.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

printf '%s\n' 'robocasa365: provider environment loaded'

: "${DASHSCOPE_MODEL:?configure DashScope in .env}"
: "${DASHSCOPE_API_KEY:?configure DashScope in .env}"
: "${DASHSCOPE_BASE_URL:?configure DashScope in .env}"
: "${DEEPSEEK_MODEL:?configure DeepSeek in .env}"
: "${DEEPSEEK_API_KEY:?configure DeepSeek in .env}"
: "${DEEPSEEK_BASE_URL:?configure DeepSeek in .env}"

export ROBOCASA_EVALUATOR_TOKEN="$(openssl rand -hex 32)"
export ROBOCASA_DATA_TOKEN="$(openssl rand -hex 32)"
export ROBOCASA_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ROBOCASA_POLICY="${ROBOCASA_POLICY:-/root/.cache/huggingface/hub/models--lerobot--pi052_robocasa/snapshots/693102ebcd9b28aeec3728638dd433a04ffbdee6}"

# The host kernel driver is 535.309.01. Prefer its matching, locally extracted
# userspace EGL stack when present; this avoids the unstable Mesa fallback used
# when the system libnvidia-gl package is an older point release.
nvidia_user_lib="${ROBOCASA_NVIDIA_USER_LIB_DIR:-$(dirname "$repo_root")/.cache/Xbotics-Hey-Robot/nvidia-535.309.01/extracted}"
if [[ -f "$nvidia_user_lib/libEGL_nvidia.so.535.309.01" ]]; then
  export LD_LIBRARY_PATH="$nvidia_user_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Keep MuJoCo/EGL rendering off the CUDA device used by PI0.5. On this host,
# running both in one process on GPU 0 makes frames become corrupted directly
# after the first policy inference. Prefer GPU 1 when it exists, while keeping
# a single-GPU host usable via GPU 0.
if [[ -z "${MUJOCO_EGL_DEVICE_ID:-}" ]]; then
  nvidia_gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
  if (( nvidia_gpu_count >= 2 )); then
    export MUJOCO_EGL_DEVICE_ID=1
  else
    export MUJOCO_EGL_DEVICE_ID=0
  fi
fi
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
printf 'robocasa365: PI0.5 CUDA device=%s, MuJoCo EGL device=%s\n' \
  "${ROBOCASA_POLICY_DEVICE:-cuda}" "$MUJOCO_EGL_DEVICE_ID"

mkdir -p runtime/robocasa365/launcher-logs
worker_log="runtime/robocasa365/launcher-logs/worker.log"
agent_log="runtime/robocasa365/launcher-logs/agent.log"
cleanup() {
  kill "${worker_pid:-}" "${agent_pid:-}" 2>/dev/null || true
  wait "${worker_pid:-}" "${agent_pid:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

.robocasa365-venv/bin/python -m evaluation.robocasa365.worker.server \
  --host 127.0.0.1 --port 9092 \
  --evaluator-token "$ROBOCASA_EVALUATOR_TOKEN" \
  --data-token "$ROBOCASA_DATA_TOKEN" >"$worker_log" 2>&1 &
worker_pid=$!
printf '%s\n' 'robocasa365: worker started'

.venv/bin/hey-robot run \
  --config configs/evaluation/robocasa365.agent.yaml >"$agent_log" 2>&1 &
agent_pid=$!
printf '%s\n' 'robocasa365: Hey Robot started'

for _ in $(seq 1 60); do
  if curl --fail --silent http://127.0.0.1:18080/api/tasks >/dev/null; then
    break
  fi
  sleep 1
done

if ! curl --fail --silent http://127.0.0.1:18080/api/tasks >/dev/null; then
  sed -n '1,200p' "$worker_log" >&2
  sed -n '1,200p' "$agent_log" >&2
  exit 1
fi
printf '%s\n' 'robocasa365: web channel ready'

.venv/bin/python evaluation/robocasa365/full_system_benchmark.py \
  --agent-url http://127.0.0.1:18080/turn \
  --runtime-target grpc://127.0.0.1:9092 \
  "$@"
