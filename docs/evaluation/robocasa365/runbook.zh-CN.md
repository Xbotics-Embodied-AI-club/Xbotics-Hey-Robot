# RoboCasa365 VLA 评估运行手册

本文只说明如何启动和运行 RoboCasa365 embodied-agent option 评估。架构设计见
`docs/evaluation/robasaca365/robocasa365-embodied-agent-evaluation.zh-CN.md`。

当前已验证成功的命令是：

- policy：`lerobot/pi052_robocasa`
- task：`CloseFridge`
- seed：`1000`
- max steps：`300`
- 结果：官方 success predicate 成功，232 steps 完成

## 1. 进入仓库

```bash
cd /workspace/caofuping/Xbotics-Hey-Robot
```

## 2. 检查 GPU

```bash
nvidia-smi
```

当前机器已验证环境是两张 RTX 3090。默认命令使用 `cuda`，会优先占用 GPU 0。

## 3. 激活 RoboCasa365 venv

```bash
cd /workspace/caofuping/Xbotics-Hey-Robot
source .robocasa365-venv/bin/activate
```

检查关键依赖：

```bash
python - <<'PY'
import sys
print(sys.version)
for name in ["torch", "lerobot", "robocasa", "robosuite", "mujoco"]:
    mod = __import__(name)
    print(name, getattr(mod, "__version__", "import-ok"))
PY
```

预期版本大致如下：

```text
Python 3.12.x
torch 2.7.1+cu126
lerobot 0.6.1
robocasa 1.0.0
robosuite 1.5.2
mujoco 3.3.1
```

## 4. 检查 pi0.5 checkpoint 缓存

当前成功运行依赖本地 Hugging Face cache，不要在网络不可达时强制改 `HF_HOME`。

```bash
test -d /root/.cache/huggingface/hub/models--lerobot--pi052_robocasa
du -sh /root/.cache/huggingface/hub/models--lerobot--pi052_robocasa
```

预期大小约为 11G。

## 5. 设置运行环境变量

```bash
cd /workspace/caofuping/Xbotics-Hey-Robot
source .robocasa365-venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ROBOCASA_POLICY="lerobot/pi052_robocasa"
export ROBOCASA_POLICY_DEVICE="cuda"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export LD_LIBRARY_PATH="/workspace/caofuping/.cache/Xbotics-Hey-Robot/robocasa365/driver/535.309.01:${LD_LIBRARY_PATH:-}"
export __EGL_VENDOR_LIBRARY_FILENAMES="/workspace/caofuping/.cache/Xbotics-Hey-Robot/robocasa365/driver/535.309.01/egl_vendor.json"
```

不要在这组命令里设置 `HF_HOME=/workspace/caofuping/.cache/huggingface`。本机已验证的
pi0.5 cache 位于 `/root/.cache/huggingface`。

## 6. 运行 pi0.5 CloseFridge 成功用例

```bash
cd /workspace/caofuping/Xbotics-Hey-Robot
source .robocasa365-venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ROBOCASA_POLICY="lerobot/pi052_robocasa"
export ROBOCASA_POLICY_DEVICE="cuda"
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export LD_LIBRARY_PATH="/workspace/caofuping/.cache/Xbotics-Hey-Robot/robocasa365/driver/535.309.01:${LD_LIBRARY_PATH:-}"
export __EGL_VENDOR_LIBRARY_FILENAMES="/workspace/caofuping/.cache/Xbotics-Hey-Robot/robocasa365/driver/535.309.01/egl_vendor.json"

mkdir -p runtime/robocasa365
RUN_DIR="runtime/robocasa365/agent-pi052-closefridge-seed1000-ms300-$(date -u +%Y%m%dT%H%M%SZ)"

python evaluation/robocasa365/agent_benchmark.py \
  --task CloseFridge \
  --episodes 1 \
  --seed 1000 \
  --policy-path lerobot/pi052_robocasa \
  --max-steps 300 \
  --device cuda \
  --output-dir "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR.log"
```

退出码含义：

- `0`：所有 trial 成功；
- `2`：流程跑完但至少一个 trial 未成功；
- 其他非 0：运行时错误。

## 7. 查看结果

```bash
cd /workspace/caofuping/Xbotics-Hey-Robot

python - <<'PY'
import json
from pathlib import Path

runs = sorted(Path("runtime/robocasa365").glob("agent-pi052-closefridge-seed1000-ms300-*"))
if not runs:
    raise SystemExit("No pi052 CloseFridge run found")
run = runs[-1]
manifest = json.loads((run / "agent_benchmark_manifest.json").read_text())
print("run_dir:", run)
print("success_count:", manifest.get("success_count"))
print("success_rate:", manifest.get("success_rate"))
print("trials:", manifest.get("trials"))
print("policy_path:", manifest.get("policy_path"))
print("max_steps:", manifest.get("max_steps"))

for result_path in sorted(run.glob("trial_*/result.json")):
    result = json.loads(result_path.read_text())
    print("result:", result_path)
    print("  task:", result.get("task"))
    print("  seed:", result.get("seed"))
    print("  success:", result.get("success"))
    print("  status:", result.get("status"))
    print("  failure_mode:", result.get("failure_mode"))
    option = result["option_results"][-1]
    metrics = option.get("metrics", {})
    print("  option_summary:", option.get("summary"))
    print("  steps:", metrics.get("steps"))
    print("  episode_success:", metrics.get("episode_success"))
    print("  last_reward:", metrics.get("last_reward"))
    print("  last_info:", metrics.get("last_info"))
PY
```

成功时应看到类似：

```text
success_count: 1
success_rate: 1.0
trials: 1
success: True
status: completed
failure_mode: None
option_summary: RoboCasa option CloseFridge: success after 232 steps
episode_success: True
last_reward: 1.0
```

## 8. 跑 smolvla 对照

`smolvla` 更小，适合检查流程是否能快速启动，但它在此前 smoke 中没有完成
`CloseFridge`。

```bash
cd /workspace/caofuping/Xbotics-Hey-Robot
source .robocasa365-venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ROBOCASA_POLICY="lerobot/smolvla_robocasa"
export ROBOCASA_POLICY_DEVICE="cuda"
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export LD_LIBRARY_PATH="/workspace/caofuping/.cache/Xbotics-Hey-Robot/robocasa365/driver/535.309.01:${LD_LIBRARY_PATH:-}"
export __EGL_VENDOR_LIBRARY_FILENAMES="/workspace/caofuping/.cache/Xbotics-Hey-Robot/robocasa365/driver/535.309.01/egl_vendor.json"

mkdir -p runtime/robocasa365
RUN_DIR="runtime/robocasa365/agent-smolvla-closefridge-seed1000-ms50-$(date -u +%Y%m%dT%H%M%SZ)"

python evaluation/robocasa365/agent_benchmark.py \
  --task CloseFridge \
  --episodes 1 \
  --seed 1000 \
  --policy-path lerobot/smolvla_robocasa \
  --max-steps 50 \
  --device cuda \
  --output-dir "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR.log"
```

## 9. 跑多个任务

当前入口允许这些任务：

```text
CloseFridge
OpenDrawer
OpenCabinet
TurnOnMicrowave
TurnOffStove
```

示例：

```bash
cd /workspace/caofuping/Xbotics-Hey-Robot
source .robocasa365-venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ROBOCASA_POLICY="lerobot/pi052_robocasa"
export ROBOCASA_POLICY_DEVICE="cuda"
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export LD_LIBRARY_PATH="/workspace/caofuping/.cache/Xbotics-Hey-Robot/robocasa365/driver/535.309.01:${LD_LIBRARY_PATH:-}"
export __EGL_VENDOR_LIBRARY_FILENAMES="/workspace/caofuping/.cache/Xbotics-Hey-Robot/robocasa365/driver/535.309.01/egl_vendor.json"

mkdir -p runtime/robocasa365
RUN_DIR="runtime/robocasa365/agent-pi052-tasks3-seed1000-ms300-$(date -u +%Y%m%dT%H%M%SZ)"

python evaluation/robocasa365/agent_benchmark.py \
  --tasks CloseFridge,OpenDrawer,OpenCabinet \
  --episodes 1 \
  --seed 1000 \
  --policy-path lerobot/pi052_robocasa \
  --max-steps 300 \
  --device cuda \
  --output-dir "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR.log"
```

注意：目前只确认 `CloseFridge` 成功。其他任务可以用同一流程评测，但不要预设成功。

## 10. 跑多个 seed

```bash
cd /workspace/caofuping/Xbotics-Hey-Robot
source .robocasa365-venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ROBOCASA_POLICY="lerobot/pi052_robocasa"
export ROBOCASA_POLICY_DEVICE="cuda"
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export LD_LIBRARY_PATH="/workspace/caofuping/.cache/Xbotics-Hey-Robot/robocasa365/driver/535.309.01:${LD_LIBRARY_PATH:-}"
export __EGL_VENDOR_LIBRARY_FILENAMES="/workspace/caofuping/.cache/Xbotics-Hey-Robot/robocasa365/driver/535.309.01/egl_vendor.json"

mkdir -p runtime/robocasa365
RUN_DIR="runtime/robocasa365/agent-pi052-closefridge-seeds1000-1002-ms300-$(date -u +%Y%m%dT%H%M%SZ)"

python evaluation/robocasa365/agent_benchmark.py \
  --task CloseFridge \
  --seeds 1000,1001,1002 \
  --policy-path lerobot/pi052_robocasa \
  --max-steps 300 \
  --device cuda \
  --output-dir "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR.log"
```

## 11. 常见问题

### Hugging Face 网络错误

如果看到：

```text
Network is unreachable
... requesting HEAD https://huggingface.co/lerobot/pi052_robocasa/resolve/main/config.json
```

说明当前没有使用已有本地 cache。先确认：

```bash
test -d /root/.cache/huggingface/hub/models--lerobot--pi052_robocasa
```

然后不要覆盖 `HF_HOME`，并设置：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### 输出目录已存在

`--output-dir` 必须是不存在的新目录。推荐始终使用：

```bash
RUN_DIR="runtime/robocasa365/name-$(date -u +%Y%m%dT%H%M%SZ)"
```

### 只跑通流程但没成功

如果 `failure_mode` 是 `option_timeout`，说明：

- 模型已加载；
- 环境已 reset；
- VLA 已输出动作；
- RoboCasa 已执行 step；
- 但官方 success predicate 没有触发。

这属于任务未成功，不是集成链路失败。

### pi0.5 加载慢

`lerobot/pi052_robocasa` 是 4B 级模型。首次加载可能需要数分钟，期间可能只看到：

```text
PI052: liger-kernel is not installed; skipping fused Triton kernels
```

这是性能 warning，不是错误。

