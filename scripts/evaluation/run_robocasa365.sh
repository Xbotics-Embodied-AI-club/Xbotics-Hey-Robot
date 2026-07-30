#!/usr/bin/env bash
# Start one managed RoboCasa365 deployment and run a single or batch benchmark.
set -euo pipefail

usage() {
  printf '%s\n' \
    'usage: run_robocasa365.sh <single|batch> [--config PATH] [benchmark options]'
}

mode="${1:-}"
case "$mode" in
  single) benchmark_module="evaluation.robocasa365.full_system_benchmark" ;;
  batch) benchmark_module="evaluation.robocasa365.batch_full_system_benchmark" ;;
  -h|--help|"") usage; exit 0 ;;
  *) printf 'unknown evaluation mode: %s\n' "$mode" >&2; usage >&2; exit 2 ;;
esac
shift

config_path="configs/evaluation/robocasa365.yaml"
benchmark_args=()
while (($#)); do
  case "$1" in
    --config)
      if (($# < 2)); then
        printf '%s\n' '--config requires a path' >&2
        exit 2
      fi
      config_path="$2"
      shift 2
      ;;
    --config=*)
      config_path="${1#*=}"
      shift
      ;;
    *)
      benchmark_args+=("$1")
      shift
      ;;
  esac
done

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

if [[ ! -f "$config_path" ]]; then
  printf 'deployment config does not exist: %s\n' "$config_path" >&2
  exit 2
fi

mapfile -t launcher_settings < <(.venv/bin/python - "$config_path" <<'PY'
import math
import sys

from hey_robot.config import DeploymentConfig

config = DeploymentConfig.from_yaml(sys.argv[1])
managed = [
    spec
    for spec in config.robots.values()
    if spec.type == "robocasa" and bool(spec.settings.get("managed_backend", False))
]
if len(managed) != 1:
    raise SystemExit("config requires exactly one managed RoboCasa robot")
print(config.resources.runtime_dir)
print(math.ceil(float(managed[0].settings.get("backend_startup_timeout_sec") or 600) + 60))
PY
)
runtime_dir="${launcher_settings[0]}"
startup_timeout_sec="${launcher_settings[1]}"
credentials_path="$runtime_dir/robocasa.credentials.json"
launcher_log_dir="$runtime_dir/launcher-logs"
agent_log="$launcher_log_dir/agent.log"

: "${DASHSCOPE_MODEL:?configure DashScope in .env}"
: "${DASHSCOPE_API_KEY:?configure DashScope in .env}"
: "${DASHSCOPE_BASE_URL:?configure DashScope in .env}"
: "${DEEPSEEK_MODEL:?configure DeepSeek in .env}"
: "${DEEPSEEK_API_KEY:?configure DeepSeek in .env}"
: "${DEEPSEEK_BASE_URL:?configure DeepSeek in .env}"

# A host may need a userspace EGL bundle matching its kernel driver.
if [[ -z "${ROBOCASA_NVIDIA_USER_LIB_DIR:-}" ]]; then
  driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
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

mkdir -p "$launcher_log_dir"
cleanup() {
  kill -TERM "${agent_pid:-}" 2>/dev/null || true
  wait "${agent_pid:-}" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

.venv/bin/hey-robot run --config "$config_path" >"$agent_log" 2>&1 &
agent_pid=$!
printf 'robocasa365: deployment started with %s\n' "$config_path"

deadline=$((SECONDS + startup_timeout_sec))
until curl --fail --silent http://127.0.0.1:18080/api/tasks >/dev/null; do
  if ! kill -0 "$agent_pid" 2>/dev/null || ((SECONDS >= deadline)); then
    sed -n '1,200p' "$agent_log" >&2
    exit 1
  fi
  sleep 1
done
printf '%s\n' 'robocasa365: web channel ready'

.venv/bin/python -m "$benchmark_module" \
  --config "$config_path" \
  --agent-url http://127.0.0.1:18080/turn \
  --runtime-target grpc://127.0.0.1:9092 \
  --credentials-file "$credentials_path" \
  "${benchmark_args[@]}"
