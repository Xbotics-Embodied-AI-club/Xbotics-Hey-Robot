# RoboCasa365 Embodied Agent 测评方案

## 1. 文档目标

状态：唯一保留的 canonical 文档，合并 RoboCasa365 embodied-agent 测评方案、
Hey Robot VLA 集成设计要点和 2026-07-20 smoke 评测结果。

更新时间：2026-07-20。

适用范围：

- 说明 Hey Robot 集成 VLA 时 Agent、Skill OS、ModelService、Robot Runtime 的职责边界；
- 定义 B0 flat VLA、B1 Agent + VLA、B2 Oracle Decomposition + VLA 等评测分层；
- 定义任务集、指标、产物、失败归因和最终验收标准；
- 记录当前已完成的 RoboCasa365 smoke 评测结果和后续未完成项。

当前落地状态：

- 已实现 B1/B2 共享的 option 评测执行器：`evaluation/robocasa365/worker/option_runner.py`；
- 已实现 `robocasa_option` ModelService skill 和 Hey Robot 内置 skill；
- 已实现多 task、多 seed、option plan 和 trial 目录结构：
  `evaluation/robocasa365/agent_benchmark.py`；
- 已保留单一评测配置：`configs/evaluation/robocasa365.agent.yaml`；
- 本文中 Long-horizon、Reasoning、Recovery、真实 LLM planner、多 seed 正式统计、
  B0 20-episode flat baseline 仍是未完成的正式评测范围。

本文定义如何使用 RoboCasa365 测评 Hey Robot embodied agent 系统，而不仅是测评一个
VLA checkpoint。方案建立在现有 RoboCasa365 worker、远程 Runtime、Agent、Skill OS 和
VLA 集成设计之上，覆盖系统边界、实验协议、任务分层、基线、指标、产物和分阶段验收。

相关文档：

- [Agent 与 Skill 边界](agent-skill-boundaries.md)
- [ModelService Proto 与 Codegen](model-service-rpc-proto.md)
- [What Matters in Orchestrating Robot Policies](../references/What%20Matters%20in%20Orchestrating%20Robot%20Policies.md)

## 2. 核心结论

RoboCasa365 可以用于测评 Hey Robot embodied agent，但必须同时保留两个不同层级的评测：

```text
Level A: Flat VLA benchmark
  完整任务 -> VLA -> 完整 episode -> success

Level B: Embodied agent benchmark
  完整任务 -> Agent 子目标/Skill -> VLA 在线执行 -> observation feedback
           -> Agent replanning -> root task success
```

当前 `robocasa_rollout` 属于 Level A。它可以验证模型、环境、相机、状态、动作和依赖集成，
但不能单独证明 Agent 的任务分解、感知使用、恢复和 orchestration 能力。

只有在同一任务、场景、seed 和 VLA 下比较 Level A 与 Level B，才能判断 Agent 层带来的
收益或损失。

## 2.1 VLA 集成边界

Hey Robot 集成 VLA 的长期边界如下：

```text
User objective
  -> Agent: select one semantic skill or bounded option
  -> Skill OS: own bounded VLA control loop
  -> ModelService / RoboCasa worker: load checkpoint and run policy inference
  -> Robot Runtime / RoboCasa env: validate and apply action
  -> fresh observation and structured result
  -> Agent: replan at semantic boundary
```

责任划分：

- Agent 只做低频语义决策，不逐帧生成动作；
- Skill OS 拥有 session、option timeout、取消、失败归因和 evidence；
- ModelService / worker 负责 checkpoint、processor、feature rename、normalization 和 VLA
  推理；
- Robot Runtime / RoboCasa env 是动作执行和物理状态权威；
- `done_hint` 只能作为提示，根任务最终计分以 RoboCasa 官方 success predicate 为准。

RoboCasa365 当前实现采用 worker 内快循环：20Hz 图像、state、VLA inference 和 12D action
step 留在同一进程内，Hey Robot 侧只传 bounded semantic option 和结构化结果。

## 2.2 环境与运维事实

RoboCasa365 使用独立 Python 环境或独立容器，不安装进 Hey Robot 主环境。原因是
RoboCasa、robosuite、MuJoCo、LeRobot、NumPy、Numba、图形后端和厨房资产需要整体锁定，
且 checkpoint、资产和 CUDA 运行依赖体积较大。

当前本地 venv 诊断布局：

```text
.robocasa365-venv/                                  # 本地 Python 环境，gitignore 排除
/workspace/caofuping/.cache/Xbotics-Hey-Robot/
  robocasa365/src/                                  # LeRobot/RoboCasa/robosuite 源码
  robocasa365/driver/535.309.01/                    # 与宿主驱动匹配的 EGL 用户态库
  robocasa365/downloads/                            # 原始下载缓存，可重建
runtime/robocasa365/<run-id>/                       # eval_info、trace、视频和 manifest
```

Docker/Compose 路径仍可用，但不是当前唯一方式。独立镜像相关文件：

```text
docker/Dockerfile.robocasa365
evaluation/robocasa365/worker/
  benchmark.py
  server.py
  runtime_server.py
  runtime_smoke.py
  rollout.py
  option_runner.py
configs/evaluation/robocasa365.agent.yaml
evaluation/robocasa365/
  agent_benchmark.py
```

目录边界：

- `src/hey_robot/` 只保留 Hey Robot 核心系统代码；
- `evaluation/` 放评估入口、实验编排和结果结构；
- `evaluation/robocasa365/worker/` 放 RoboCasa365 worker、RPC server、heavy runner 和运行时 adapter；
- `runtime/` 只放运行产物。

资产规则：

- 首轮只启用 `lightwheel` registry；
- 不要在未下载完整资产时启用 `objaverse` 或 `aigen`；
- 运行目录必须是源码基础 assets 与下载包合并后的目录，不能用下载包直接覆盖整个
  `robocasa/models/assets`；
- `fixtures` 必须来自 Lightwheel fixture 包，缺少完整 fixture 会导致 fridge、stove、
  window 等 XML 缺失；
- `runtime/` 不存放 venv、第三方源码、资产压缩包或驱动库。

EGL/离线运行关键环境变量：

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
LD_LIBRARY_PATH=/workspace/caofuping/.cache/Xbotics-Hey-Robot/robocasa365/driver/535.309.01
__EGL_VENDOR_LIBRARY_FILENAMES=/workspace/caofuping/.cache/Xbotics-Hey-Robot/robocasa365/driver/535.309.01/egl_vendor.json
```

在多 GPU 环境中，`MUJOCO_EGL_DEVICE_ID` 必须与 `CUDA_VISIBLE_DEVICES` 暴露的物理 GPU id
一致。例如 `CUDA_VISIBLE_DEVICES=1` 时使用 `MUJOCO_EGL_DEVICE_ID=1`。

模型契约：

- `lerobot/smolvla_robocasa`：checkpoint config 使用 `camera1/2/3` 和 6D state，checkpoint
  preprocessor 负责从 RoboCasa 原始 `robot0_*` 相机键映射；
- `lerobot/pi052_robocasa`：直接使用三路 `robot0_*` 相机键、16D state 和 12D action；
- 两者都输出 RoboCasa 原生 12D action；
- 切换模型必须通过 `policy_path` / 配置完成，不修改业务代码；
- `ROBOCASA_RENAME_MAP` 只用于覆盖 checkpoint 自带映射。

早期接入验收事实：

- `robocasa_rollout` ModelService v1 调度、cancel、health 和结果解析已跑通；
- frame-level Runtime 已验证 `CreateEpisode -> Observe -> Step -> Reset -> CloseEpisode`，
  三路 JPEG、16D state、12D action 和 frame id 推进均可用；
- rollout 与 frame-level episode 之间有共享资源互斥；
- 20-episode flat benchmark 曾在早期环境中完整写出 manifest、episode、视频和
  `eval_info.json`，成功率为 0/20，作为 checkpoint 质量结果保留。

## 3. 测评对象

### 3.1 被测系统

Level B 的被测对象包括：

- Agent 的目标理解和语义决策；
- PlannerObservation 和场景理解；
- Skill 选择和参数生成；
- Skill OS 的 VLA 控制循环；
- VLA 与语言子目标的接口；
- observation freshness 和动作因果性；
- 子目标终止、超时和停滞检测；
- 失败恢复和重规划；
- 根任务 evidence 审计；
- 模型、运行时和仿真之间的协议兼容性。

### 3.2 不在首轮结论中的内容

首轮结果不能直接代表：

- XLeRobot 或 SO101 真机成功率；
- RoboCasa 到真实厨房的 sim-to-real 能力；
- 人机共处和动态人物场景中的安全性；
- 未下载资产 registry 上的任务覆盖；
- VLA 训练算法或数据集本身的优劣；
- 使用 simulator privileged state 后的真实视觉能力。

## 4. 公平比较原则

所有对照组必须固定：

- VLA checkpoint 和不可变 revision；
- RoboCasa、robosuite、LeRobot 和 MuJoCo 版本；
- task、layout、style、object instances 和 seed；
- camera、state 和 action schema；
- split 和 object registry；
- episode 最大步数；
- GPU dtype 和推理配置；
- 成功判定函数；
- 资产集合。

允许变化的实验因素必须单独记录，例如：

- 是否启用 Agent；
- planner 模型；
- scene representation；
- termination strategy；
- memory strategy；
- VLA prompt；
- action-prefix 长度。

禁止在比较 Agent 与 flat VLA 时同时更换 checkpoint、seed 或资产集合。

## 5. 总体架构

### 5.1 控制平面

```text
User/Benchmark Task
        |
        v
Hey Robot Agent
  root objective
  current planner observation
  recent verified Skill outcomes
        |
        | one bounded semantic command
        v
Skill OS / Agent Evaluation Runtime
  policy session
  freshness barrier
  termination and recovery
        |
        | StartOption / AbortOption / status
        v
RoboCasa365 Worker
  persistent RoboCasa episode
  VLA policy
  12D action loop
  MuJoCo step
        |
        | keyframes + progress + option result
        v
SkillResult / Evidence
        |
        v
Agent replans or completes root task
```

### 5.2 高频数据平面

20Hz 图像、16D state 和 12D action 应尽量留在 RoboCasa worker 内。跨进程边界只传：

- episode 创建和销毁；
- 当前有界语义命令；
- option 控制命令；
- 低频关键帧或按需 observation；
- option 进度、终止原因和证据；
- episode 最终 metrics。

现有逐帧 `CreateEpisode/Observe/Step` Runtime 继续用于：

- contract 测试；
- 单步诊断；
- action schema 检查；
- 不含 VLA 的控制实验；
- 评估传输延迟的实验。

它不应默认成为生产 Agent 测评的 20Hz 图像传输路径。

## 6. 持久 Episode 与有界 Option

### 6.1 Episode 生命周期

同一个根任务必须共享一个持久环境 episode：

```text
CreateEpisode(task, seed, profile)
  -> initial observation
  -> StartOption #1
  -> option result
  -> StartOption #2
  -> option result
  -> ...
  -> root success/failure
CloseEpisode
```

切换子目标时不能 reset RoboCasa 环境。只有以下情况允许 reset：

- 开始新的独立 trial；
- 当前 episode 因仿真错误不可恢复；
- 实验协议明确要求重新开始；
- operator 主动请求 reset。

### 6.2 Option 请求

建议新增逻辑 contract；初期可复用现有 gRPC `Struct`，稳定后再决定是否增加独立 proto：

```python
StartOptionRequest(
    episode_id: str,
    option_id: str,
    root_task: str,
    command: str,
    policy_path: str,
    policy_revision: str,
    max_duration_sec: float,
    max_steps: int,
    action_schema_id: str,
    start_frame_id: int,
    metadata: dict,
)
```

命令要求：

- 主动语态；
- 只表达一个当前可执行目标；
- 不包含关节、action index 或仿真内部 ID；
- 不声称未知的物理事实；
- 与 checkpoint 训练语言保持兼容；
- 在设定的 option horizon 内有完成可能。

### 6.3 Option 状态

```text
CREATED
RUNNING
SUCCEEDED
FAILED
TIMED_OUT
STALLED
INTERRUPTED
UNSAFE
UNCERTAIN
```

Option result 至少包括：

```python
OptionResult(
    option_id,
    status,
    success,
    start_frame_id,
    end_frame_id,
    steps,
    duration_sec,
    termination_source,
    failure_mode,
    evidence,
    key_observations,
    policy_metrics,
)
```

### 6.4 Prompt 切换

开始新 Option 时必须：

1. 停止接收上一 Option 的新动作；
2. 清空 action chunk queue；
3. reset 必要的 VLA temporal state；
4. 保留 RoboCasa 环境状态；
5. 绑定新的 `option_id + command + start_frame_id`；
6. 等待一帧新 observation 后才开始推理。

旧 option 的迟到响应必须被 session 和 option ID 拒绝。

## 7. Agent 侧运行方式

### 7.1 Agent 可见信息

默认 `vision-only` 协议下，Agent 只能看到：

- 当前关键帧或场景描述；
- 机器人可公开 state；
- 当前根任务；
- 已执行 option 的结构化结果；
- 可验证 evidence；
- 超时、停滞和安全事件；
- 当前可用 Skill contract。

Agent 不得看到：

- RoboCasa task success 内部谓词；
- 对象精确世界坐标；
- fixture 内部 joint state，除非真实机器人也能提供等价传感；
- ground-truth contact graph；
- 目标物体内部 instance ID；
- 未来随机种子或初始化参数；
- evaluator 的答案和任务分解模板。

### 7.2 PlannerObservation

建议向 Agent 提供：

```python
PlannerObservation(
    episode_id,
    frame_id,
    observed_at,
    images,
    scene_summary,
    entities,
    relations,
    robot_state_summary,
    option_history,
    risks,
    confidence,
    provenance,
)
```

所有实体、关系和 summary 必须绑定 `frame_id`。缓存事实超过 freshness 阈值或发生显著世界
变化后必须失效。

### 7.3 Agent 决策频率

Agent 只在以下事件发生时重新决策：

- episode 创建；
- option 成功；
- option 失败、停滞或超时；
- scene 发生重大变化；
- operator interrupt；
- 根任务可能完成，需要最终审计。

Agent 不参与 20Hz 动作循环。

## 8. VLA 快循环

RoboCasa worker 内的在线循环：

```text
active option command
  -> current three-camera observation + 16D state
  -> checkpoint processor
  -> VLA inference
  -> validate 12D finite bounded action
  -> env.step(action)
  -> done/safety/progress/termination check
  -> repeat
```

要求：

- checkpoint camera rename 自动恢复；
- state/action normalization 来自 checkpoint；
- 每个动作关联输入 `frame_id`；
- policy 输出不能绕过 action-space 校验；
- action queue 在 option 切换时清空；
- 记录 per-step inference 和 environment latency；
- worker 必须响应 AbortOption；
- episode 环境和 policy session 的所有权明确。

RoboCasa 原生动作 schema 建议固定标识为：

```text
robocasa.panda_omron.normalized.v1
```

schema 内容必须来自当前固定 LeRobot/RoboCasa 版本的事实源，不能只根据“12 维”推断动作
语义。

## 9. 成功与终止

### 9.1 子目标终止

子目标终止使用以下优先级：

```text
safety/exception
  > deterministic option predicate
  > learned or VLM success detector
  > policy done hint
  > max duration/max steps
```

`policy done` 只能作为 hint。它必须由 effect check、成功检测器或 evaluator evidence 确认，
否则返回 `UNCERTAIN` 或继续执行。

### 9.2 根任务成功

RoboCasa 环境的官方 `is_success` 是根任务最终评分的唯一权威。Agent 不能直接读取它，但
evaluator 可以在每个 step 后检查并记录。

建议区分：

- `agent_declared_complete`：Agent 是否认为任务完成；
- `environment_success`：RoboCasa 官方成功谓词；
- `completion_precision`：Agent 声称完成时真正成功的比例；
- `completion_recall`：环境已成功时 Agent 能否及时结束。

这样可以发现 Agent 过早结束或任务已经完成却继续操作的问题。

### 9.3 Privileged 信息

提供两种正式协议：

```text
vision-only
  Agent、scene model 和 VLA 不读取 simulator ground truth

privileged-oracle
  可使用 contact、object state 和官方 success predicate
```

`privileged-oracle` 用于诊断系统上限、验证 termination 和隔离视觉误差，不能与
`vision-only` 混合统计为同一个 Agent 成绩。

## 10. 任务集设计

### 10.1 Smoke 集

目的：验证环境和在线闭环，不评价复杂规划。

候选类别：

- CloseFridge；
- OpenDrawer；
- OpenCabinet；
- TurnOnMicrowave；
- TurnOffStove。

现有 allowlist 可以覆盖这一阶段。每个任务先运行少量固定 seed，确认 reset、观察、动作、
终止和视频产物正确。

### 10.2 Atomic 集

目的：测量低层 VLA 和集成上限。

任务特征：

- 单个 fixture 状态改变；
- 单物体 pick/place；
- 单次开关或按钮操作；
- 与 VLA 训练轨迹长度接近；
- 不需要非平凡任务分解。

Atomic 集主要用于比较 checkpoint，不应被用来证明高层 Agent 的价值。

### 10.3 Long-horizon 集

目的：测量 Agent 的任务分解、子目标衔接和恢复。

任务特征：

- 两个或更多对象操作；
- 打开 fixture、取物、放置、关闭 fixture；
- 清理、分类和多容器操作；
- 需要保持中间世界状态；
- 完成时间显著超过单条 VLA 训练轨迹。

### 10.4 Reasoning 集

目的：测量语义理解和观察使用。

任务特征：

- 间接指令；
- 属性或类别选择；
- 冷/热、食物/非食物等语义分类；
- 相对空间关系；
- 需要根据场景确定目标对象。

### 10.5 Recovery 集

目的：测量失败后的系统行为，而不是只测首次成功。

通过可复现方式注入：

- VLA 空动作；
- 短暂 observation stale；
- 一次 action 拒绝；
- 一次抓取失败；
- 物体被推离原位置；
- planner 生成不可执行 prompt；
- option timeout；
- 模型服务短暂 busy。

每种故障必须记录注入位置和随机种子，不能依赖人工临时干预。

## 11. 对照组

每个正式任务至少包含以下组：

### B0：Flat VLA

```text
full task prompt -> VLA -> full episode
```

使用当前 `robocasa_rollout`。它是 checkpoint 基线。

### B1：Agent + VLA

```text
full task -> Hey Robot Agent -> bounded commands -> same VLA
```

这是主要被测系统。

### B2：Oracle Decomposition + VLA

```text
human-authored correct bounded commands -> same VLA
```

它用于区分“Agent 分解失败”与“VLA 无法执行正确子目标”。Oracle 分解只决定命令序列，
不能提供 oracle action。

### B3：Oracle Executor，可选

```text
Agent commands -> scripted/privileged executor
```

它用于测量 Agent 规划上限，但结果必须明确标注为 oracle executor，不能称为 VLA 成绩。

### 可选消融

- 无 scene caption；
- raw image only；
- 固定 horizon 与成功检测器；
- 无 recovery；
- 不同 prompt template；
- SmolVLA 与 PI0.5；
- action prefix 1 与短前缀；
- 无历史与最近 1–3 个 option outcome。

## 12. 指标

### 12.1 根任务指标

- `root_success_rate`；
- `episode_return`；
- `steps_to_success`；
- `wall_time_to_success`；
- `timeout_rate`；
- `agent_declared_completion_precision`；
- `agent_declared_completion_recall`。

### 12.2 Agent 指标

- `options_per_episode`；
- `invalid_option_rate`；
- `duplicate_option_rate`；
- `replans_per_episode`；
- `planner_calls`；
- `planner_latency_ms`；
- `premature_completion_rate`；
- `unnecessary_action_after_success_rate`。

### 12.3 VLA 与控制指标

- `option_success_rate`；
- `policy_inference_latency_ms`；
- `action_acceptance_rate`；
- `empty_action_rate`；
- `stale_frame_rate`；
- `observation_timeout_rate`；
- `vla_stalled_rate`；
- `action_queue_flush_count`；
- `policy_done_hint_precision`。

### 12.4 恢复与安全指标

- `recovery_attempt_rate`；
- `recovery_success_rate`；
- `same_failure_repeat_count`；
- `safety_stop_count`；
- `unsafe_action_rejection_count`；
- `abort_latency_ms`；
- `post_abort_action_count`，目标必须为 0。

### 12.5 资源指标

- GPU 显存峰值；
- GPU 利用率；
- VLM/VLA token 或推理成本；
- 图像传输字节数；
- worker 重启次数；
- episode reset 失败率。

不能只报告最终成功率，否则无法定位 Agent、VLA、视觉、控制或环境中的失败来源。

## 13. 运行产物

每个 trial 使用唯一目录：

```text
runtime/robocasa365-agent/<experiment_id>/<trial_id>/
  manifest.json
  versions.json
  root_task.json
  planner_trace.jsonl
  option_trace.jsonl
  policy_trace.jsonl
  environment_trace.jsonl
  evidence.jsonl
  stdout.log
  stderr.log
  videos/
  result.json
```

`manifest.json` 至少保存：

```text
experiment configuration hash
git revision and dirty flag
container image digest
LeRobot/RoboCasa/robosuite/MuJoCo versions
policy repo and immutable revision
planner model identifier
task/split/registry/seed
evaluation protocol: vision-only or privileged-oracle
baseline group
GPU assignment
```

思维链和敏感 provider 内容不应默认持久化。保留结构化 planner decision、tool call、prompt
摘要、证据和模型 usage 即可。

## 14. 实验清单

建议使用机器可读 manifest 驱动实验：

```yaml
experiment_id: robocasa365-agent-v1
protocol: vision-only
split: target
obj_registries: [lightwheel]
tasks:
  - CloseFridge
  - OpenDrawer
seeds: [1000, 1001, 1002, 1003, 1004]
groups:
  - id: flat-smolvla
    mode: flat_rollout
    policy_path: lerobot/smolvla_robocasa
  - id: agent-smolvla
    mode: embodied_agent
    policy_path: lerobot/smolvla_robocasa
    planner: configured-agent-planner
  - id: oracle-smolvla
    mode: oracle_decomposition
    policy_path: lerobot/smolvla_robocasa
```

正式统计前应冻结 manifest。临时修改 prompt、timeout 或 termination 后必须产生新的
experiment ID，不能覆盖原结果。

## 15. 统计方法

### 15.1 Trial 数量

Smoke 阶段可使用每任务 3–5 个 trial。正式比较不能依赖单次视频，应根据目标置信区间
确定 trial 数量。资源有限时至少：

- 每任务每组使用相同的一组 seed；
- 报告成功数和总 trial 数；
- 报告二项比例置信区间；
- 使用 paired seed 比较 Agent 与 flat VLA；
- 不把不同任务简单合并后只报告一个总平均。

### 15.2 分层报告

分别报告：

- Atomic；
- Long-horizon；
- Reasoning；
- Recovery；
- 按 checkpoint；
- 按 vision-only/privileged-oracle；
- 按失败模式。

### 15.3 失败归因

每个失败必须归入一个首要阶段：

```text
environment_reset
observation
scene_understanding
agent_planning
skill_contract
policy_inference
action_validation
robot_execution
option_termination
root_completion
timeout_or_infrastructure
```

允许记录次要原因，但不能把所有失败统一归为 `task_unsuccessful`。

## 16. 配置与部署建议

保留单一评测 profile，避免多份子集配置互相漂移：

```text
configs/evaluation/robocasa365.agent.yaml  embodied-agent evaluation
```

`robocasa365.agent.yaml` 当前包含 Hey Robot agent evaluation 所需的 `robocasa_option`
和 `robocasa_rollout` skill 配置。具体 task、seed、policy、输出目录和环境变量应由
评测 CLI 参数、运行脚本或 `runtime/robocasa365/<run-id>/manifest.json` 记录，不再
保存在通用配置目录里。

长期完整 profile 仍应包含：

- `robots.robocasa365` remote embodiment；
- Agent 和 planner provider；
- Skill policy；
- scene captioner；
- VLA policy service；
- agent-visible semantic skills；
- evaluation timeout 和资源配置；
- explicit policy/action/state schema；
- vision-only 或 privileged-oracle profile。

双 RTX 3090 的建议分配：

```text
GPU 0: RoboCasa + VLA policy
GPU 1: local VLM/scene model，或保留给第二个并行 trial
```

同一个 RoboCasa worker 当前只允许一个活动 episode/rollout。并行评测需要独立 worker、
独立端口、独立输出目录和独立 GPU 资源预算，不能绕过共享资源锁。

## 17. 分阶段实施

### Gate 0：基线冻结

- 固定容器 image、依赖和资产；
- 固定 SmolVLA 与 PI0.5 revision；
- 保存现有 flat rollout 结果；
- 确认相同 task/seed 可重复 reset；
- 记录当前 allowlist 和资产覆盖。

通过条件：flat baseline 可重复运行，所有失败均可区分环境错误与任务失败。

### Gate 1：持久 Episode

- Agent 创建、观察和关闭 episode；
- 多个 no-op/诊断 option 共享同一环境状态；
- option 切换不 reset 环境；
- unknown/stale episode 被拒绝；
- shared lock 和取消行为正确。

通过条件：一个根任务可以跨至少三个 option 保持连续环境状态。

### Gate 2：在线 VLA Option

- worker 内加载 checkpoint；
- command 可在 episode 中切换；
- 12D action schema 校验；
- action queue 在切换和取消时清空；
- 记录逐步 frame/action 因果关系；
- AbortOption 后无残留动作。

当前状态：SmolVLA 和 PI0.5 已能通过 `evaluation/robocasa365/agent_benchmark.py`
完成 checkpoint load、
processor、RoboCasa reset、VLA inference、12D action、`env.step` 和 trace 写入。
短步数 smoke 结果均为 `option_timeout`，因此只证明集成链路，不证明 task success。

通过条件：Atomic smoke task 能通过 Agent 提交的单个 option 完成，且 trace 完整。

### Gate 3：Agent 闭环

- 使用 `configs/evaluation/robocasa365.agent.yaml`；
- Agent 能获取关键 observation；
- SkillResult 返回 option evidence；
- Agent 能执行多 option 根任务；
- 根任务完成由官方 success predicate 审计；
- Agent 过早完成能够被拒绝。

通过条件：至少一个 Long-horizon 任务形成完整的观察、决策、执行、反馈和重规划链路。

### Gate 4：公平对照

- B0/B1/B2 使用相同 checkpoint、task 和 seed；
- vision-only 与 privileged-oracle 分离；
- manifest 冻结；
- 失败归因和 metrics 完整；
- 汇总脚本不读取人工挑选的视频结果。

通过条件：可以回答 Agent 提升或降低了多少成功率，以及变化来自哪个阶段。

### Gate 5：规模化评测

- 扩展 allowlist 和资产；
- 建立 Atomic/Long-horizon/Reasoning/Recovery 分层；
- 支持断点续跑和 trial 去重；
- 固定统计脚本和报告模板；
- 对失败样本做可重放诊断。

通过条件：批量运行不会因单个 trial 崩溃而丢失实验状态，结果可由 manifest 重现。

## 18. 测试要求

### Contract 测试

- episode/option ID 不匹配；
- stale frame；
- 错误的 state/action 维度；
- NaN、Inf 和越界动作；
- policy revision 不匹配；
- option 切换后的迟到 action；
- active episode 资源冲突；
- cancellation 和 close 幂等。

### 控制测试

- command 切换清空 action queue；
- 每个 env step 关联唯一输入 frame；
- VLA 推理失败时环境不 step；
- AbortOption 后 `post_abort_action_count == 0`；
- environment success 后不会继续无界动作；
- option timeout 不会错误关闭整个可恢复 episode；
- root timeout 会关闭 episode 和 policy session。

### Agent 测试

- observation 缺失时不能猜测场景；
- Agent 每次只提交一个有界 command；
- failed option 后能够观察或选择不同策略；
- 重复相同失败达到阈值后停止；
- 根任务完成必须引用动作后 evidence；
- privileged 字段不会进入 vision-only planner context。

### 回归测试

- 当前 flat `robocasa_rollout` 不受 Agent 模式影响；
- SmolVLA 与 PI0.5 均能通过同一 checkpoint probe；
- frame-level Runtime contract 保持可用；
- output path、视频和日志仍按 trial 隔离。

## 19. 风险与控制

### VLA 不接受子目标重述

部分 checkpoint 可能只适合训练语料中的完整任务表达。需要先做 prompt steerability 测试，
比较原始 task prompt、规范化短 prompt 和 Agent 生成 prompt。若正确短 prompt 仍显著降低成功
率，应把该 checkpoint 标记为不适合 orchestration，而不是把失败归因于 Agent。

### 高频跨进程延迟

默认把 VLA 和环境快循环放在 worker 内。逐帧远程模式只用于测量和诊断，除非延迟数据证明
它可以稳定满足控制频率。

### Privileged 信息污染

Agent 输入和 evaluator truth 必须使用不同数据结构和调用接口。日志中应明确每条 evidence
的 provenance，测试应阻止 vision-only profile 访问 simulator truth。

### 资产和任务覆盖不一致

扩展任务前先验证 registry、mesh、texture 和 fixture。因资产缺失无法 reset 的 trial 不计为
Agent 失败，但必须计入 infrastructure failure。

### Agent 成本和延迟

Agent 仅在 option 边界运行。设置 planner call、总 wall time 和最大 option 数预算，防止通过
无限重试换取成功率。

### 结果不可复现

所有实验使用 manifest、immutable revision 和唯一输出目录。dirty worktree、未固定模型
revision 或修改后的 prompt 必须写入结果元数据。

## 20. 最小可行版本

首个可交付版本不需要覆盖全部 365 类任务。最小范围是：

1. 保留当前五任务 flat baseline；
2. 新增一个持久 episode Agent profile；
3. 支持 SmolVLA 和 PI0.5 配置切换；
4. 支持一个 command 的在线 VLA option；
5. 支持至少一个两阶段根任务；
6. 提供 vision-only 与 privileged termination 两种结果；
7. 运行 B0、B1、B2 三组相同 seed 对照；
8. 生成完整 trace、视频、失败归因和汇总报告。

完成以上内容后，系统才可以宣称：

> RoboCasa365 已被用于测评 Hey Robot embodied agent 的感知、语义决策、VLA 执行和反馈
> 闭环。

在此之前，应准确表述为：

> RoboCasa365 已用于 Hey Robot 外部 VLA rollout 和集成基线测试。

## 21. 最终验收标准

方案完成的最终判据是：

- 同一 checkpoint、task 和 seed 可以运行 flat、Agent 和 oracle decomposition 三组实验；
- Agent trial 中环境 episode 跨多个 Skill/Option 保持连续；
- Agent 不读取 evaluator privileged truth；
- VLA 动作与 observation frame 存在可审计因果关系；
- option 切换、超时和取消不会执行残留动作；
- 根任务以 RoboCasa 官方 success predicate 计分；
- 每个失败可以归因到明确系统阶段；
- 所有结果包含固定版本、配置、seed、trace 和视频；
- 批量结果可由 manifest 重现；
- flat 与 Agent 的差异能够被统计和解释。

## 22. 当前 Smoke 评测结果

日期：2026-07-20。

环境：

- LeRobot：0.6.1；
- RoboCasa：1.0.0；
- robosuite：1.5.2；
- MuJoCo：3.3.1；
- GPU：2 x RTX 3090；
- 渲染：EGL；
- 离线缓存：`HF_HUB_OFFLINE=1`，`TRANSFORMERS_OFFLINE=1`。

已完成实现：

- `evaluation/robocasa365/worker/option_runner.py`：持久 episode 和 bounded option；
- `evaluation/robocasa365/worker/server.py`：`robocasa_option` ModelService skill；
- `src/hey_robot/skill_os/builtins/robocasa.py`：`RoboCasaOptionSkill`；
- `evaluation/robocasa365/agent_benchmark.py`：多 task、多 seed、option plan、trial 目录和 summary
  manifest；
- PI0.5 离线推理 processor fallback：推理时关闭训练用 `enable_fast_action_loss`；
- SmolVLA checkpoint camera alias 和 6D state contract；
- venv/editable RoboCasa asset health。

已运行实验：

```text
agent-smolvla
  policy: lerobot/smolvla_robocasa
  tasks: CloseFridge, OpenDrawer, OpenCabinet
  seeds: 1000
  max_steps: 20
  output: runtime/robocasa365/agent-smolvla-smoke-v1-t3-s1-ms20
  result: 3 trials, success_count=0, success_rate=0.0
  failure_mode: option_timeout

agent-pi052
  policy: lerobot/pi052_robocasa
  task: CloseFridge
  seed: 1000
  max_steps: 5
  output: runtime/robocasa365/agent-pi052-smoke-v1-closefridge-s1-ms5
  result: 1 trial, success_count=0, success_rate=0.0
  failure_mode: option_timeout
```

解释：

- 两个 policy 都已完成 checkpoint load、processor/preprocess、RoboCasa reset、VLA
  `select_action`、native 12D action、`env.step` 和 trace 写入；
- 当前失败类型均为 `option_timeout`，表示给定短步数预算内 RoboCasa 官方 success predicate
  未触发；
- 这些结果证明集成链路可运行，不证明 checkpoint 任务成功率。

尚未完成的正式评测：

- B0 flat VLA baseline：`benchmark.py` 固定 20 episodes；
- 更长 B1/B2 horizon，例如每 task 300 到 1000 steps；
- 多 seed 统计，例如 seeds 1000 到 1004；
- Long-horizon / Reasoning / Recovery 任务集；
- 真实 LLM planner 调用路径，而不是当前 oracle option plan。

满足这些条件后，RoboCasa365 不再只是模型 smoke test，而是 Hey Robot embodied agent 的
系统级测评环境。
