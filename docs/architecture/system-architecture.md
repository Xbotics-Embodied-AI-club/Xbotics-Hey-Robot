# Hey Robot 系统架构

本文描述当前代码的实际运行结构。历史设计稿可能仍使用“三层架构”或
“CapabilityService”等旧术语；当前实现以四层边界、NATS 主链和 gRPC ModelService
为准。

## 1. 系统定位

Hey Robot 是一个不依赖通用 LLM Agent 框架、面向真实机器人原生构建的
Embodied Agent Harness，不是单独的 LLM loop，也不是端到端神经网络控制器。

核心 Agent Runtime、tool protocol、任务状态机、记忆路由和 execution-feedback loop
均由项目自主实现。OpenAI-compatible SDK 只作为模型 provider client，不承担 Agent
编排。这里的“原生”指系统从一开始围绕异步物理执行、观测过期、硬件状态和失败恢复设计。

系统围绕以下工程问题提供统一实现：

- 多渠道用户入口、身份和会话连续性；
- LLM tool loop、任务状态、记忆和主动感知；
- Skill contract、资源调度、超时和中断；
- VLA/VLN 等 Foundation Model 的独立部署；
- MuJoCo 和真机的统一 RobotDriver 边界；
- observation、execution feedback、recovery、审计和任务 UI。

当前主线 embodiment 是 XLeRobot，真机由 SO101 机械臂、LeKiwi 移动底盘、相机和
舵机总线电池监控组成。

## 2. 异步快慢双系统与四层架构

快慢双系统描述决策层级：

| 系统 | 负责什么 | 当前代码 |
|---|---|---|
| 慢系统 | 语言理解、目标分解、任务状态、记忆、长程规划、恢复 | `RobotAgentCore`、`AgentRuntime`、`TaskRunManager`、`MemoryBroker` |
| 快系统 | VLA/VLN、短时域技能闭环、安全门控、局部控制、机器人执行 | Foundation Model、Skill OS、Human Follow、Robot Runtime |

两套系统不是同步函数调用栈，而是通过 `SkillIntent`、`SkillEvent`、`SkillResult`、
`RobotStatus` 和 `RobotObservation` 异步协作。慢系统下发目标，快系统执行并持续产生
状态和证据，慢系统再根据反馈继续规划、恢复或结束任务。

这里的“快”表示更接近具身控制、决策周期和任务视野更短，不表示所有 VLA/VLN 推理都比
LLM 更低延迟，也不表示当前 Python asyncio 执行路径是硬实时控制器。

四层架构描述这套快慢系统在代码和部署中的所有权边界：

```text
User Channels
  Web / Voice / Feishu / CLI
          |
          v
GatewayService
  identity / episode / history / presentation
          |
          | NATS: user.turn
          v
1. Agent / Cognition
  RobotAgentService
  RobotAgentLoop: restore -> build -> run -> save
  RobotAgentCore / AgentRuntime
  task state / memory / perception / feedback / recovery
          |
          | NATS: skill.intent
          v
2. Skill OS
  SkillControllerService
  SkillContractRuntime / SkillScheduler / SkillRuntime
          |                           |
          | NATS: robot.action        | gRPC
          v                           v
4. Robot Runtime              3. Foundation Model
  RobotService                   ModelService
  RobotRuntime                   VLA / VLN executors
  MuJoCo / native                GetHealth / ExecuteSkill / CancelSkill
          |                           |
          +-------------+-------------+
                        |
                        v
  robot.status / robot.observation / skill.event / skill.result
```

四层与快慢系统的对应关系如下：

1. Agent 层属于慢系统，决定“当前应该请求什么能力”，但不接触硬件动作。
2. Skill OS 属于快系统的执行编排层，决定能力是否允许、如何组合和占用资源。
3. Foundation Model 层属于快系统的学习型决策层，提供 VLA/VLN 等短时域结果。
4. Robot Runtime 属于快系统的执行层，决定如何在当前 embodiment 上执行 primitive。

Gateway、消息总线、持久化和通知是贯穿四层的系统基础设施，不作为第五个机器人决策层。

## 3. 默认部署形态

`hey-robot run` 使用 `DeploymentRunner` 在一个 asyncio 进程中构造：

- `RobotService`
- 可选 `HumanFollowService`
- `SkillControllerService`
- `TaskSupervisorService`
- 一个或多个 `RobotAgentService`
- `GatewayService`

这些服务各自创建 NATS client，并通过协议消息协作。ModelService 和 NATS broker 是独立
进程。CLI 同时提供 `agent`、`gateway`、`robot`、`task-supervisor` 和
`model-service` 等入口，因此主服务也可拆分部署。

这个形态应表述为：

> 协议上服务化、可拆分；默认主系统本地一体化。

默认 NATS 配置使用 core publish/subscribe。只有显式设置 `use_jetstream` 时才启用
JetStream，因此不能默认假设消息会持久化或重放。任务的 durable state 主要来自本地
JSON/JSONL store。

## 4. 一次 turn 的实际链路

### 4.1 Gateway

Channel 将外部输入归一化为 `UserTurn`。Gateway：

1. 解析 `user_id`；
2. 选择 `agent_id` 和 `robot_id`；
3. 按身份、渠道和会话维度分配 episode；
4. 保存用户 turn；
5. 发布 `user.turn`。

`Envelope` 贯穿后续消息，携带 `trace_id`、`episode_id`、`robot_id`、`agent_id`、
channel 和用户身份。

### 4.2 Agent turn

`RobotAgentService` 为 episode 和 robot 加锁，避免同一机器人同时处理冲突 turn。
`RobotAgentLoop` 依次执行：

- `restore`：恢复 checkpoint 和 task state；
- `build`：组装 history、memory、recovery 和最新 robot snapshot；
- `run`：调用 `RobotAgentCore`；
- `save`：保存 task state 和 robot episode state。

`RobotAgentCore` 的决策顺序是：

1. turn mode 和任务安全；
2. stop、reset、home、gripper 等确定性短命令路由；
3. memory 和主动感知上下文；
4. `AgentRuntime` 的 LLM tool loop。

LLM 可见的生产工具包括状态查询、任务上下文、感知、记忆、等待、动作提议和
`request_skill`。Agent 层不能直接构造 `RobotAction`，也不能依赖 driver primitive。

### 4.3 Skill 请求和等待策略

所有物理能力请求统一经过 `SkillGateway`。它负责：

- 检查 skill 是否在 deployment surface 中；
- 应用 channel/task safety；
- 检查 recovery block；
- 对 motion skill 检查相机健康和 freshness；
- 阻止没有新感知证据的连续运动；
- 构造并发布 `SkillIntent`。

等待策略：

- `wait_result`：tool call 等待最终 `SkillResult`，LLM 可在同一 turn 中继续规划；
- `wait_acceptance`：发布后立即返回已受理，适合 stop 等短命令；
- `return_handle`：只返回 `skill_id` 句柄。

当前有限长程任务主要依赖 `wait_result`，在一个 turn 内形成多次
“观察—动作—反馈—继续”的循环。

## 5. Skill OS

`BaseSkill.spec` 是 Skill contract 的事实源，描述：

- 输入和必填参数；
- required resources；
- dependencies 和 driver primitives；
- required model service；
- supported robots；
- safety level、timeout 和 interruptibility；
- success criteria、failure modes 和 recovery hints；
- goal effects 和 evidence outputs。

`SkillControllerService` 接收 `SkillIntent` 后：

1. 解析 contract；
2. 检查参数、robot state 和 readiness；
3. 检查资源冲突；
4. 创建 `SkillRun`；
5. 通过 `SkillRuntime` 执行 plugin；
6. 等待 RobotStatus 或 ModelService result；
7. 发布 `SkillEvent` 和 `SkillResult`。

资源可以根据参数实例化。例如 `arm=left` 会把通用 `arm` 资源转换为
`left_arm`。相机资源是共享资源，base、arm 和 gripper 默认互斥。

### Production 与 bringup

- `production`：`skills.enabled` 只能列出 `agent_visible=True` 的 semantic skill。
- `bringup`：允许把 primitive/implementation skill 直接暴露给 Agent，用于联调。

当前系统仍处于开发和联调阶段，仓库提供的 real/sim 主配置使用 `bringup`。
这些配置会把 `move_base`、`turn_base`、`base_velocity_step`、`set_arm_pose`、
`move_arm_joints`、`set_gripper`、`detect_marker` 等底层调试 skill 显式暴露出来，
便于验证硬件、仿真和 Skill OS 到 Robot Runtime 的完整链路。

最终生产 profile 仍应使用 `production`，只暴露 `agent_visible=True` 的 semantic skill。

## 6. Foundation Model 层

ModelService wire contract 的 source of truth 是：

```text
proto/hey_robot/model_service/v1/model_service.proto
```

当前 RPC：

- `GetHealth`
- `ExecuteSkill`
- `CancelSkill`

`ModelServiceRegistry` 按 `model_services.<id>.provides` 和 `robot_id` 路由请求。
Skill OS 在调用前检查服务是否 online、loaded 和 busy。

### VLN

VLN executor 是 planner-only：

```text
camera frame + instruction
  -> ModelService
  -> pixel_goal / heading / stop
  -> Skill OS adapter
  -> move_base / turn_base / stop_motion
  -> refresh observation
```

它不是 SLAM 或全局路径规划器，当前输出会被转换为粗粒度局部 primitive。

### VLA

VLA Skill 的目标结构是：

```text
current observation + task prompt
  -> one inference step
  -> joint/gripper primitives
  -> Robot Runtime
  -> refresh observation
  -> repeat
```

当前实现仍处于实验阶段：

- `xlerobot.sim.vla_vln.yaml` 的 VLA `model_path` 为空，因此当前只能走内部接口测试路径；
- real inference 仍构造 LeRobot RobotClient 和 camera/arm config，尚未完全成为只消费
  injected observation 的纯推理服务；
- `pick_object`、`place_object` 的 contract 依赖 `vla_manipulation`，但执行时按各自
  semantic name 查询 ModelService；部署时必须确保 `provides` 与实际调用名一致。

因此默认 real/sim 配置不开放 VLA；实验配置也不应被描述为已经验证的真实 VLA 闭环。

## 7. Robot Runtime

Skill OS 将 primitive 编码为 `RobotAction`。当前 skill action 的主要内容位于
`metadata.skill`，而不是连续向量 `values`：

```text
RobotAction
  metadata.action_type = "skill"
  metadata.skill.name
  metadata.skill.arguments
```

`RobotRuntime` 统一处理：

- driver lifecycle；
- capabilities 和 health；
- action safety；
- perception skill；
- observation materialization；
- reset 和 status。

`RobotManager` 根据 `family + environment + driver` 选择：

- `XLeRobotSimDriver`
- `XLeRobotDriver`
- `SO101Driver`
- `LeKiwiDriver`

代码中另有仅供自动化测试使用的 driver test double；它不作为对外支持的机器人环境。

XLeRobot 真机 primitive 通过 `ClassicSkillExecutor` 路由到
`NativeXLeRobotClient`，再访问 LeKiwi 底盘、SO101 机械臂、SCServo 总线和 OpenCV
相机。MuJoCo driver 实现相同协议，因此 Agent 和 Skill 不需要区分真机与仿真。

## 8. Camera、Observation 和 Scene

Driver 输出 `DriverObservation`。`ObservationPipeline`：

1. 检查空图、shape 和黑帧；
2. 将图像和大 artifact 保存到 local media store；
3. 在 `RobotObservation` 中只携带 `ImageRef`、`ArtifactRef` 和小型 metadata；
4. 添加 `valid_image_count` 和 quality issues。

RobotService 的 merged observation loop 用一次 `driver.observe()` 同时生成：

- `robot.observation`：结构化、引用式 observation；
- `robot.camera.frame.<robot_id>`：低延迟 raw frame packet。

Agent、场景 captioner 和 memory 使用结构化 observation；human follow 和需要当前画面的
Foundation consumer 使用 raw frame stream。

## 9. Task、Memory、Feedback 和 Recovery

主要 durable state：

- `JsonlEpisodeStore`：用户和 Agent 对话；
- `TaskRunStore`：root task、attempt、skill binding、feedback 和 recovery；
- `RobotEpisodeStateStore`：最近 robot state；
- `SceneMemoryStore`：场景证据；
- Long-term memory：偏好、地点、经验和事件；
- `RuntimeEventStore` / `SkillStore`：审计和 lifecycle。

这些存储目前是本地文件，适合单机部署和可解释审计，不等价于多节点事务数据库。

SkillResult 到达 Agent 后会生成 execution feedback。反馈区分：

- subgoal 是否成功；
- root task 是否成功；
- confidence；
- failure reason；
- next hint；
- recommended action。

Task Supervisor 监控 skill timeout、status/observation stale、camera quality 和 recovery
state。重复相同失败会逐步升级为 `ask_operator` 和 `safe_abort`。

当前 autonomy 仍是 turn-driven：Task Supervisor 负责监控和通知，不会在无新 turn 时
持续唤醒 LLM。`autonomous` 表示 tool-using task execution mode，而不是永久运行的
后台目标循环。

## 10. 安全边界

当前采用多层确定性 gate：

```text
task/channel safety
  -> SkillGateway camera/recovery/consecutive-motion checks
  -> Skill contract/readiness/resource checks
  -> RobotRuntime health/battery/estop checks
  -> driver contract validation
```

这些 gate 能降低 LLM 误调用风险，但不构成工业安全系统。当前没有完整碰撞检测、SLAM、
全局避障或硬实时安全控制；Web、NATS 和 gRPC 的默认开发配置也没有认证。真机部署必须
隔离网络，并保留物理急停或断电手段。

## 11. Web 与可观察性

主要页面：

- `/chat`：交互入口；
- `/tasks`：任务列表；
- `/tasks/{episode_id}`：任务详情；
- `/admin`：运行时总览。

`/cockpit` 页面路由保留为兼容入口并重定向到 `/tasks`；
`/cockpit/{episode_id}` 仍是 TaskSession 聚合数据 API。

任务视图由 `TaskSessionQueryService` 聚合 TaskRun、robot state、scene memory、
skill lifecycle 和 recovery。

## 12. 能力边界

默认 real/sim bringup 配置暴露以下非 VLA semantic skill 和底层调试 skill：

```text
inspect_scene, look_around, human_follow,
stop_motion, reset_posture,
detect_marker,
move_base, turn_base, base_velocity_step,
set_arm_pose, move_arm_joints, set_gripper
```

因此主线系统支持观察、短步底盘运动、视觉人体跟随、安全停止/复位、机械臂命名姿态、
关节控制和夹爪控制。

实验配置 `xlerobot.sim.vla_vln.yaml` 的 bringup surface 额外声明：

```text
navigate_to, approach_object,
pick_object, place_object
```

这些能力需要独立 ModelService，且受前述 VLA/VLN 限制。当前系统不应被描述为已经具备
稳定 SLAM、全局避障或真实通用抓取能力。

## 13. 架构守卫

测试明确限制：

- cognition 不得在 SkillGateway 之外构造 SkillIntent；
- cognition 不得依赖 RobotAction 或 driver primitive；
- robot_runtime 不得导入 cognition、skill_os 或 foundation backends；
- foundation 不得导入 cognition 或 skill_os；
- legacy package 和旧 generated contract 不得重新出现；
- proto source 与 generated artifacts 必须保持一致。

这些约束使四层结构成为可执行的代码规则，而不只是文档约定。
