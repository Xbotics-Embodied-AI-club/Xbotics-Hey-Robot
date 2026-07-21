# RoboCasa365 完整系统评测

本文是 Hey Robot 集成 RoboCasa365 的唯一说明文档，只介绍当前有效架构和评测启动方式。

## 1. 集成目标

RoboCasa365 用于在仿真厨房中评估 Hey Robot 完整 embodied-agent 系统，而不是只测试一个
独立 VLA。一次 B1 评测会实际经过：

```text
用户根任务
  -> DeepSeek Agent / 快系统规划
  -> DashScope 场景理解
  -> Skill OS: inspect_scene / robocasa_option
  -> ModelService RPC
  -> lerobot/pi052_robocasa
  -> EpisodeManager
  -> RoboCasa365 environment
  -> RoboCasa 官方成功谓词
```

worker 内只有一个 `EpisodeManager`，它是 simulator、observation、frame ID 和 action step
的唯一状态源。Agent 看不到 RoboCasa 官方成功标签；成功与失败由 benchmark 独立读取并写入
评测结果。

当前只保留一条执行路线：

- 单任务入口：`evaluation/robocasa365/full_system_benchmark.py`；
- 批量入口：`evaluation/robocasa365/batch_full_system_benchmark.py`；
- 唯一任务清单：`configs/evaluation/robocasa365.tasks.yaml`；
- 唯一 Agent 配置：`configs/evaluation/robocasa365.agent.yaml`；
- 唯一推荐启动器：`scripts/evaluation/run_robocasa365_full_system.sh`。

## 2. 已验证状态

真实 checkpoint `lerobot/pi052_robocasa` 已通过完整 B1 链路验证：

- task：`CloseFridge`；
- split：`target`；
- environment 与 PI052 RNG seed：`1000`；
- 结果：`official_success=true`；
- 完成位置：第 227 个 environment step；
- PI052 option 数：5；
- Agent step 数：10。

验证产物位于：

```text
runtime/robocasa365/long-horizon-close-fridge-b1-r14/
```

其中 `result.json` 是汇总结果，`video.mp4` 是成功视频。

## 3. 关键运行约束

PI052 必须遵守其独立 LeRobot evaluator 的推理契约：

1. 模型的 `task` 使用 RoboCasa 环境自然语言 `task_description`，不能替换成 Agent 生成的
   option 标签；
2. option horizon 与 checkpoint 的 `n_action_steps=50` 对齐；
3. `policy.reset()` 每个 trial 只调用一次，不能在 option 边界清空层级 subtask 状态；
4. trial seed 同时设置 environment 以及隔离 PI052 进程的 Python、NumPy、Torch/CUDA RNG；
5. Agent 配置使用 `hard_max_skills=64`，以覆盖最多 1000 个环境步和中间重新观察。

PI052 在独立 spawn 子进程中使用 CUDA，MuJoCo/EGL 留在 worker 主进程。这个隔离用于避免
Torch/CUDA inference 与 EGL 共进程时出现渲染缓冲异常，不能合并回同一进程。

当前宿主 kernel driver 为 NVIDIA 535.309.01。启动器会自动优先加载已经解压的匹配版本
用户态 EGL 库；双 GPU 主机默认让 PI052 使用 GPU 0、MuJoCo EGL 使用 GPU 1。

## 4. 运行前准备

项目根目录的 `.env` 需要配置以下变量：

```dotenv
DASHSCOPE_MODEL=...
DASHSCOPE_API_KEY=...
DASHSCOPE_BASE_URL=...

DEEPSEEK_MODEL=...
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=...
```

不要把真实 API key 写入本文、命令行参数或评测 artifact。唯一启动器会自动读取 `.env`。

本地 PI052 checkpoint 默认使用：

```text
/root/.cache/huggingface/hub/models--lerobot--pi052_robocasa/snapshots/693102ebcd9b28aeec3728638dd433a04ffbdee6
```

若模型位于其他位置，可设置：

```bash
export ROBOCASA_POLICY=/absolute/path/to/pi052_robocasa
```

## 5. 启动一个完整评测

在仓库根目录执行：

```bash
bash scripts/evaluation/run_robocasa365_full_system.sh \
  --task CloseFridge \
  --seed 1000 \
  --objective "Close the fridge." \
  --producer b1 \
  --output-dir runtime/robocasa365/close-fridge-b1-seed1000 \
  --timeout-sec 1200
```

启动器会自动完成：

1. 读取 `.env`；
2. 生成本轮短生命周期 Runtime credentials；
3. 配置 NVIDIA/EGL；
4. 启动 RoboCasa worker；
5. 启动 Hey Robot、DeepSeek planner 和 DashScope scene captioner；
6. 创建 trial 并预热真实 PI052；
7. 运行完整 Agent 闭环；
8. 保存结果并关闭本轮服务。

PI052 checkpoint 约 10.9 GB，当前宿主首次冷加载通常需要 4～5 分钟。frame 在加载期间保持
为 0，属于正常现象。

每次运行必须使用新的 `--output-dir`。入口拒绝覆盖已有目录，从而避免历史实验被静默覆盖。

## 6. B0、B1、B2

`--producer` 可选：

- `b0`：直接使用根任务文本；
- `b1`：真实 Hey Robot AutonomousAgentService，正式系统评测默认使用该项；
- `b2`：冻结的上限规划提示，用于区分规划问题和 VLA 执行问题。

三者共用相同的 Gateway、Skill OS、RPC、VLA 和 EpisodeManager，不存在绕过完整系统的
direct runner。

## 7. 批量评测

当 worker 与 Hey Robot 服务已经启动时，可以运行：

```bash
.venv/bin/python evaluation/robocasa365/batch_full_system_benchmark.py \
  --output-root runtime/robocasa365/atomic-b1-seeds \
  --suite atomic_gate \
  --producer b1 \
  --seeds 1000,1001 \
  --agent-url http://127.0.0.1:18080/turn \
  --runtime-target grpc://127.0.0.1:9092 \
  --timeout-sec 1800
```

任务分组来自 `configs/evaluation/robocasa365.tasks.yaml`：

- `atomic_gate`：基础原子能力门禁；
- `composite_seen`：组合已见任务；
- `composite_unseen`：组合未见任务。

建议先运行 `atomic_gate`，确认基础 VLA 能力和系统链路，再投入 composite 长程批量实验。

## 8. 评测产物

每个 trial 的输出目录包含：

```text
manifest.json
trial_spec.json
root_task.json
agent_trace.jsonl
observations.jsonl
agent_events.jsonl
skill_events.jsonl
model_service_events.jsonl
actions.jsonl
options.jsonl
evaluator_truth.json
result.json
summary.json
video.mp4
```

最重要的字段位于 `result.json`：

- `official_success`：RoboCasa 官方成功谓词；
- `episode_done`：environment 是否终止；
- `frame_id` / `action_count`：执行的环境步数；
- `option_count`：PI052 option 数量；
- `planner_steps`：Agent 步骤数；
- `false_completion`：Agent 宣称完成但官方谓词未成功；
- `failure_stage` 与 `termination_reason`：失败阶段和终止原因。

不要只根据 Agent 文本判断成功，正式统计必须使用 `official_success`。

## 9. 运行前检查与常见问题

运行相关测试：

```bash
.venv/bin/pytest -q --no-cov \
  tests/integration/test_robocasa365_contract.py \
  tests/robot_runtime/test_robocasa_remote_driver.py

.venv/bin/ruff check \
  evaluation/robocasa365 \
  src/hey_robot/robot_runtime/robocasa_remote \
  src/hey_robot/skill_os/builtins/robocasa.py
```

常见现象：

- frame 长时间为 0：通常是 PI052 冷加载；先检查 worker 日志和 GPU 显存；
- 出现彩色噪声帧：检查是否误把 PI052 CUDA 与 MuJoCo EGL 放回同一进程，以及是否加载了
  与 535.309.01 匹配的用户态 EGL；
- 固定 seed 结果不一致：确认 trial seed 同时传入 environment 和 PI052 子进程；
- 运行在约 500 步被阻断：检查是否退回通用 `hard_max_skills=24`；
- option 每 30 步结束：配置过时，PI052 必须使用 50 步 chunk；
- Agent option 名改变模型行为：配置过时，PI052 根任务必须来自 environment
  `task_description`；
- `official_success=false`：保留完整 artifact，先区分 planner、observation、RPC、VLA、
  environment 或 completion verifier，再决定是否重跑。

失败结果不能被覆盖，也不要在 Agent 不知情的情况下手动追加 action 或 option。重新实验应
使用新输出目录和新 trial ID。
