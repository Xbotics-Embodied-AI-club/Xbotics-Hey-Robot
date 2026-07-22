# RoboCasa365 完整系统评测

本文是 Hey Robot 集成 RoboCasa365 的唯一说明文档，只介绍当前有效架构和评测启动方式。

## 1. 集成目标

RoboCasa365 用于在仿真厨房中评估 Hey Robot 完整 embodied-agent 系统，而不是只测试一个
独立 VLA。一次 B1 评测会实际经过：

```text
用户根任务
  -> DeepSeek Agent / 快系统规划
  -> DashScope 场景理解
  -> Skill OS: inspect_scene / manipulate
  -> 标准 ModelService RPC（每次只推理一个 action）
  -> lerobot/pi052_robocasa（12D action）
  -> RobotAction / Robot Runtime
  -> RoboCasaRemoteDriver / Runtime.Step
  -> EpisodeManager
  -> RoboCasa365 environment
  -> RoboCasa 官方成功谓词
```

backend 内只有一个 `EpisodeManager`，它是 simulator、observation、frame ID 和 action step
的唯一状态源。Agent 看不到 RoboCasa 官方成功标签；成功与失败由 benchmark 独立读取并写入
评测结果。

当前只保留一条执行路线：

- 单任务入口：`evaluation/robocasa365/full_system_benchmark.py`；
- 批量入口：`evaluation/robocasa365/batch_full_system_benchmark.py`；
- 唯一任务清单：`configs/evaluation/robocasa365.tasks.yaml`；
- 唯一 Agent 配置：`configs/evaluation/robocasa365.agent.yaml`；
- 唯一推荐启动器：`scripts/evaluation/run_robocasa365_full_system.sh`。

## 2. 已验证状态

真实 checkpoint `lerobot/pi052_robocasa` 已通过完整链路验证：

- task：`CloseFridge`；
- split：`target`；
- environment 与 PI052 RNG seed：`1000`；
- 结果：`official_success=true`；
- condition：`b2`；
- 完成位置：第 218 个 environment step；
- `manipulate` option 数：5；
- Agent step 数：9；
- 所有 218 个 action 都经过 Robot Runtime；
- 验证环境由 `uv.lock` 从空目录重新创建，不依赖旧虚拟环境中残留的包。

验证产物位于：

```text
runtime/robocasa365/fresh-group-close-fridge-b2-seed1000/
```

其中 `result.json` 记录了本次真实成功。PI0.5/CUDA 的推理并非逐 bit 确定，因此固定 seed
用于追踪输入条件，但不承诺每次都在完全相同的 frame 成功。

## 3. 关键运行约束

PI052 必须遵守其独立 LeRobot evaluator 的推理契约：

1. 模型的 `task` 使用 RoboCasa 环境自然语言 `task_description`，不能替换成 Agent 生成的
   option 标签；
2. option horizon 与 checkpoint 的 `n_action_steps=50` 对齐；
3. `policy.reset()` 每个 trial 只调用一次，不能在 option 边界清空层级 subtask 状态；
4. trial seed 同时设置 environment 以及隔离 PI052 进程的 Python、NumPy、Torch/CUDA RNG；
5. Agent 配置使用 `hard_max_skills=220`，覆盖 target50 最长 2900 个环境步及中间重新观察。

PI052 在独立 spawn 子进程中使用 CUDA，MuJoCo/EGL 留在 managed backend 主进程。这个隔离用于避免
Torch/CUDA inference 与 EGL 共进程时出现渲染缓冲异常，不能合并回同一进程。

启动器会查询当前 NVIDIA kernel driver，并自动优先加载已经解压的匹配版本用户态 EGL 库；
`mujoco_device: auto_separate` 会在多 GPU 主机上尽量让 MuJoCo 与 PI052 分离。配置中没有固定
某个主机驱动版本。

## 4. 运行前准备

### 4.1 从锁文件创建独立后端环境

RoboCasa365 的 Python 依赖只有一个事实源：
`pyproject.toml` 中的 `robocasa365` dependency group 和仓库根目录的 `uv.lock`。
本地与 Docker 都执行同一条安装语义：

```bash
uv sync --frozen --only-group robocasa365 --no-install-project
```

第一次安装，或者需要彻底重建时运行：

```bash
scripts/evaluation/setup_robocasa365_env.sh --recreate
```

该脚本只删除并重建项目根目录下的 `.robocasa365-venv`，不会删除模型权重或约 5 GB 的
RoboCasa assets。它会验证 assets 完整性，并把新环境中的 RoboCasa package 指向统一的
`artifacts/robocasa365/merged-assets`。CUDA driver、EGL/OpenGL 系统库、模型权重和
RoboCasa assets 是运行资源，不属于 Python dependency group。

### 4.2 Provider 配置

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

PI052 checkpoint、device、prompt mode、horizon 和 timeout 只在
`configs/evaluation/robocasa365.agent.yaml` 中配置：

```text
policy_path: lerobot/pi052_robocasa
policy_device: cuda
prompt_mode: environment_root
option_horizon: 50
```

`pi052_robocasa` 使用 `subtask_mem` recipe：推理时从官方根任务生成并保持自己的低层 subtask。
因此当前 checkpoint 必须使用 `environment_root`；把外部 Agent 子目标直接替换到 `task`
通道会形成二次分解，并破坏独立 evaluator 的输入契约。若后续接入明确支持外部短指令的
steerable VLA，应作为同一 `manipulate` 接口的另一个模型配置验证，而不是修改动作链。

若模型位于其他位置，应修改或覆盖这份 deployment YAML；不要另设一套 shell 默认值。backend
会将同一配置映射给隔离模型进程。

## 5. 启动一个完整评测

在仓库根目录执行：

```bash
bash scripts/evaluation/run_robocasa365_full_system.sh \
  --task CloseFridge \
  --seed 1000 \
  --condition b1 \
  --output-dir runtime/robocasa365/close-fridge-b1-seed1000 \
  --timeout-sec 7200
```

默认不传 `--objective`：评测器会在创建 live trial 后读取环境的官方语言指令，并把它作为
根任务。只有专门评估语言改写鲁棒性时才使用 `--objective` 显式覆盖；artifact 会同时记录
`official_objective` 和 `objective_source`，避免把错误任务描述误判成策略失败。

环境步数上限也不再固定为 1000：backend 会读取 RoboCasa dataset registry 中每个任务的
官方 `horizon`，同时配置 wrapper truncation 和底层 robosuite horizon。target50 的长任务可达
2900 步，因此正式批量评测默认 wall-clock timeout 为 7200 秒。

启动器会自动完成：

1. 读取 `.env`；
2. 由 `DeploymentRunner` 生成 mode `0600` 的短生命周期 Runtime/ModelService credentials；
3. 配置 NVIDIA/EGL；
4. 通过 `hey-robot run` 托管并健康检查 RoboCasa backend；
5. 启动 Hey Robot、DeepSeek planner 和 DashScope scene captioner；
6. 同时通过 Runtime 和标准 ModelService 健康门禁，然后创建 trial；
7. 运行完整 Agent 闭环；
8. 保存结果，并由 Hey Robot 回收 backend 和模型子进程。

PI052 checkpoint 约 10.9 GB，当前宿主首次冷加载通常需要 4～5 分钟。frame 在加载期间保持
为 0，属于正常现象。

每次运行必须使用新的 `--output-dir`。入口拒绝覆盖已有目录，从而避免历史实验被静默覆盖。

## 6. B0、B1、B2

`--condition` 可选：

- `b0`：一次完整根目标 `manipulate`，benchmark 在第一次调用结束后强制结束 trial，保证
  flat-policy 对照不依赖 Agent 是否遵守自然语言提示；
- `b1`：使用正常层级规划并在 option 边界重新观察；
- `b2`：冻结“观察—根目标操作—再观察”的 oracle pattern。

三者只是同一 Agent 入口的实验提示，共用相同 Gateway、Skill OS、RPC、VLA 和
EpisodeManager，不存在 condition 专属 runner 或动作路径。

## 7. 批量评测

当 `hey-robot run` 已经启动时，可以运行：

```bash
.venv/bin/python -m evaluation.robocasa365.batch_full_system_benchmark \
  --output-root runtime/robocasa365/atomic-b1-seeds \
  --suite atomic_gate \
  --condition b1 \
  --seeds 1000,1001 \
  --agent-url http://127.0.0.1:18080/turn \
  --runtime-target grpc://127.0.0.1:9092 \
  --timeout-sec 7200
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
actions.jsonl
options.jsonl
evaluator_truth.json
result.json
summary.json
runtime_metadata.json
video.mp4
```

其中已删除重复且没有独立事实来源的 `model_service_events.jsonl`；option 生命周期以 Skill OS
的 `options.jsonl` 为准，动作以 evaluator-only action ledger 的 `actions.jsonl` 为准。

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
  src/hey_robot/skill_os/builtins/manipulation.py
```

常见现象：

- frame 长时间为 0：通常是 PI052 冷加载；先检查 managed backend 日志和 GPU 显存；
- 出现彩色噪声帧：检查是否误把 PI052 CUDA 与 MuJoCo EGL 放回同一进程，以及是否加载了
  与 535.309.01 匹配的用户态 EGL；
- 固定 seed 结果不一致：确认 trial seed 同时传入 environment 和 PI052 子进程；
- 长任务在官方 horizon 前被阻断：检查是否退回通用 `hard_max_skills=24` 或固定 1000 步；
- option 每 30 步结束：配置过时，PI052 必须使用 50 步 chunk；
- Agent option 名改变模型行为：配置过时，PI052 根任务必须来自 environment
  `task_description`；
- `official_success=false`：保留完整 artifact，先区分 planner、observation、RPC、VLA、
  environment 或 completion verifier，再决定是否重跑。

失败结果不能被覆盖，也不要在 Agent 不知情的情况下手动追加 action 或 option。重新实验应
使用新输出目录和新 trial ID。
