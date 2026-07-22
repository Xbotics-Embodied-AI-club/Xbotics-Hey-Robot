#!/usr/bin/env bash
# Start the complete Hey Robot deployment, including its managed RoboCasa
# backend, then run the benchmark client.
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

# A host may need a userspace EGL bundle matching its kernel driver. This is a
# device-runtime concern, so discover it instead of encoding a driver version.
if [[ -z "${ROBOCASA_NVIDIA_USER_LIB_DIR:-}" ]]; then
  driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
  candidate_root="$(dirname "$repo_root")/.cache/Xbotics-Hey-Robot"
  if [[ -n "$driver_version" ]]; then
    candidate_dir="$candidate_root/nvidia-$driver_version/extracted"
    if [[ -d "$candidate_dir" ]]; then
      export ROBOCASA_NVIDIA_USER_LIB_DIR="$candidate_dir"
    fi
  fi
fi
if [[ -n "${ROBOCASA_NVIDIA_USER_LIB_DIR:-}" ]]; then
  export LD_LIBRARY_PATH="$ROBOCASA_NVIDIA_USER_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p runtime/robocasa365/launcher-logs
agent_log="runtime/robocasa365/launcher-logs/agent.log"
cleanup() {
  kill -TERM "${agent_pid:-}" 2>/dev/null || true
  wait "${agent_pid:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

.venv/bin/hey-robot run \
  --config configs/evaluation/robocasa365.agent.yaml >"$agent_log" 2>&1 &
agent_pid=$!
printf '%s\n' 'robocasa365: Hey Robot deployment started'

for _ in $(seq 1 60); do
  if curl --fail --silent http://127.0.0.1:18080/api/tasks >/dev/null; then
    break
  fi
  sleep 1
done

if ! curl --fail --silent http://127.0.0.1:18080/api/tasks >/dev/null; then
  sed -n '1,200p' "$agent_log" >&2
  exit 1
fi
printf '%s\n' 'robocasa365: web channel ready'

.venv/bin/python -m evaluation.robocasa365.full_system_benchmark \
  --agent-url http://127.0.0.1:18080/turn \
  --runtime-target grpc://127.0.0.1:9092 \
  "$@"
