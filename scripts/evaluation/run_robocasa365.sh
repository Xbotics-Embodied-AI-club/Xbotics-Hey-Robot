#!/usr/bin/env bash
# Start one managed RoboCasa365 deployment and run the unified batch benchmark.
set -euo pipefail

usage() {
  printf '%s\n' \
    'usage: run_robocasa365.sh [--config PATH] [benchmark options]'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

config_path="configs/evaluation/robocasa365.rldx.yaml"
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
launcher_python="${HEY_ROBOT_PYTHON:-.venv/bin/python}"
launcher_cli="${HEY_ROBOT_CLI:-.venv/bin/hey-robot}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Some local HTTP proxies intermittently stall TLS handshakes for long-running
# multimodal Agent trials. Allow an evaluation to use the configured model API
# directly without changing the user's general shell or .env proxy settings.
if [[ "${HEY_ROBOT_DIRECT_MODEL_API:-0}" == "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi

if [[ ! -f "$config_path" ]]; then
  printf 'deployment config does not exist: %s\n' "$config_path" >&2
  exit 2
fi

mapfile -t launcher_settings < <("$launcher_python" - "$config_path" <<'PY'
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
print(str(managed[0].settings.get("target") or "grpc://127.0.0.1:9092"))
required = set()
for agent in config.agents.values():
    models = agent.settings.get("models") or {}
    for model in models.values():
        if not isinstance(model, dict):
            continue
        for field in ("model_env", "api_key_env", "base_url_env"):
            if model.get(field):
                required.add(str(model[field]))
for name in sorted(required):
    print(name)
PY
)
runtime_dir="${launcher_settings[0]}"
startup_timeout_sec="${launcher_settings[1]}"
runtime_target="${launcher_settings[2]}"
credentials_path="$runtime_dir/robocasa.credentials.json"
launcher_log_dir="$runtime_dir/launcher-logs"
agent_log="$launcher_log_dir/agent.log"

for env_name in "${launcher_settings[@]:3}"; do
  if [[ -z "${!env_name:-}" ]]; then
    printf 'configure %s in .env for %s\n' "$env_name" "$config_path" >&2
    exit 2
  fi
done

# A host may need a userspace EGL bundle matching its kernel driver.
if [[ -z "${ROBOCASA_NVIDIA_USER_LIB_DIR:-}" ]]; then
  driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
  candidate_root="$repo_root/.cache/robocasa365"
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

"$launcher_cli" run --config "$config_path" >"$agent_log" 2>&1 &
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

# The web gateway can become ready before the managed backend has written the
# role-scoped evaluator credentials.  Do not let benchmark startup race that
# sidecar, especially when multiple deployments are loading models in parallel.
until [[ -f "$credentials_path" ]]; do
  if ! kill -0 "$agent_pid" 2>/dev/null || ((SECONDS >= deadline)); then
    sed -n '1,200p' "$agent_log" >&2
    printf 'robocasa365: managed backend credentials were not created: %s\n' "$credentials_path" >&2
    exit 1
  fi
  sleep 1
done
printf '%s\n' 'robocasa365: managed backend credentials ready'

runtime_port="${runtime_target##*:}"
until ss -ltn | grep -q ":${runtime_port}[[:space:]]"; do
  if ! kill -0 "$agent_pid" 2>/dev/null || ((SECONDS >= deadline)); then
    sed -n '1,200p' "$agent_log" >&2
    printf 'robocasa365: managed backend did not listen on %s\n' "$runtime_target" >&2
    exit 1
  fi
  sleep 1
done
printf '%s\n' 'robocasa365: managed backend runtime ready'

"$launcher_python" -m evaluation.robocasa365.benchmark \
  --config "$config_path" \
  --agent-url http://127.0.0.1:18080/turn \
  --runtime-target grpc://127.0.0.1:9092 \
  --credentials-file "$credentials_path" \
  "${benchmark_args[@]}"
