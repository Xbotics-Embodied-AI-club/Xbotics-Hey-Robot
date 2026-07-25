# Hey Robot 系统架构

本文描述当前代码的实际运行结构。历史设计稿中的 `RobotAgentCore`、`AgentRuntime`、
`SkillGateway`、`SkillControllerService`、`TaskSupervisorService` 和 event-driven Skill
控制链均已删除，不属于当前生产拓扑。

## 1. 系统定位

Hey Robot 是面向真实机器人构建的 Embodied Agent Harness。它不依赖通用 LLM Agent
框架，也不是端到端神经网络控制器。系统自行实现：

- 多渠道接入、身份绑定、episode 和会话连续性；
- LLM tool loop、持续任务状态和恢复；
- Skill 参数校验、资源互斥、超时、取消和运行记录；
- VLA/VLN ModelService 路由；
- MuJoCo、真机与远程 RoboCasa 的 RobotDriver 边界；
- observation、execution event、media、审计和 Tasks UI。

当前主线 embodiment 是 XLeRobot，真机由 SO101 机械臂、LeKiwi 移动底盘、相机和
舵机总线电池监控组成。

## 2. 快慢系统与四个代码所有权层

快慢系统描述决策时间尺度：

| 系统 | 责任 | 当前实现 |
|---|---|---|
| 慢系统 | 语言理解、任务推进、模型工具决策、持久状态和恢复 | `Agent`、`AgentRunner`、`AgentTaskStore`、`TaskCoordinator` |
| 快系统 | 有界 Skill闭环、模型短时域决策、安全检查和机器人执行 | `SkillWorker`、VLA/VLN option、`RobotRuntime`、driver |

四个代码所有权层为：

1. **Cognition**：决定调用哪个可见 Tool/Skill，不构造 RobotAction；
2. **Skill**：定义有界能力，管理参数、资源、timeout、取消和生命周期；
3. **Foundation Model**：通过 gRPC 提供 VLA/VLN 推理，不拥有机器人环境；
4. **Robot Runtime**：拥有 observation、安全、control plane 和 driver执行。

Gateway、bus、persistence、media 和 logging 是横切基础设施。

## 3. 当前实际拓扑

```text
User Channels
  Web / Voice / Feishu / CLI
          |
          v
GatewayService
  identity / episode / history / presentation
          |
          | NATS or in-memory bus: conversation.turn
          v
AutonomousAgentService
          |
          v
Agent -> AgentRunner -> AgentToolExecutor
          |
          v
TaskCoordinator
          |
          | in-process SkillClient
          v
SkillWorker -> SkillRunner -> Skill handler
          |                       |
          |                       +---- gRPC ----> ModelService
          v
LocalRobotClient -> RobotRuntime -> RobotDriver
                           |
                           v
           robot.status / robot.observation / skill.event projection
```

关键边界：

- Gateway 到 Agent 使用 bus 消息；
- Agent 到 Skill 使用进程内 `SkillClient`；
- Skill 到 Robot 使用进程内 `LocalRobotClient`；
- Skill 到 VLA/VLN 使用 gRPC `ModelService`；
- Skill执行事实写入 `FileRunStore`，再异步投影到 bus；
- RobotService 独立发布 observation、status 和 raw camera frame。

运行时已经停用旧 `skill.intent` 与 `robot.action` 总线控制入口。物理能力只通过进程内
`SkillClient -> LocalRobotClient -> RobotRuntime` 主链提交；Human Follow 的受限速度流
仍使用独立 topic。项目已经发布 `1.0.0`，对应 DTO 和 `Topics` 名称暂时仅作为 1.x
源码兼容面保留；它们没有生产订阅者，不表示仍支持旧执行拓扑。

## 4. 默认部署形态

`hey-robot run` 的 `DeploymentRunner` 在一个 asyncio 进程中创建：

- `RobotService`；
- 一个本地 `SkillWorker`（以 managed service `skills` 启停）；
- 一个 enabled `AutonomousAgentService`；
- 可选 `HumanFollowService`；
- `GatewayService`；
- 可选受管 RoboCasa sidecar。

配置校验只接受 `skills.execution_mode: local`，并拒绝一个 deployment 中启用多个
autonomous Agent。因此当前部署边界应描述为：

> Agent、Skill Worker 与 Robot Runtime 默认且目前必须在主 Harness 进程中组合；
> ModelService 和 RoboCasa backend 可以独立进程或容器部署。

NATS不是始终必需：`deployment.bus.type: in_memory` 可用于同进程开发与评测；NATS用于
外部 Channel/Agent通信、运行投影和 raw frame consumer。Core NATS publish/subscribe
默认不持久化；只有显式启用 JetStream 才有 broker侧持久化语义。

## 5. 一次用户 turn

### 5.1 Gateway

Channel 把输入归一化为 `UserTurn`。Gateway：

1. 解析统一身份；
2. 选择 Agent 和 Robot；
3. 分配 episode/session；
4. 保存用户历史；
5. 发布 `ConversationTurn`。

`Envelope` 携带 trace、episode、channel、user、agent、robot 和 deployment身份。

### 5.2 Agent

`AutonomousAgentService` 为 session 延迟创建一个 `Agent`。Agent：

1. 由 `AgentContextBuilder` 构造对话或恢复上下文；
2. 调用纯决策 `AgentRunner`；
3. 接受最终文本，或一个 harness/physical typed proposal；
4. 由 `AgentToolExecutor` 执行 proposal；
5. 非物理 Tool结果可在同次唤醒继续进入模型；
6. 物理 Skill提交后返回 `waiting`，等待终态事件恢复。

一次唤醒最多进行 8 次模型决策。用户在物理动作期间 steer 时，意图写入 conversation，
Agent在安全点继续；急停走确定性 control path，不依赖 LLM。

### 5.3 持续任务与恢复

`TaskCoordinator` 在提交 Skill前先写入 pending step，然后构造 `SkillCommand` 并调用
`SkillClient.submit()`。Skill终态通过 `SkillClient.events()` 回到 Agent，更新 task step
并触发下一次模型决策。

启动恢复会查询 active run：

- 已有终态事件则归并到 Agent task；
- 持久化为非终态但没有 worker所有者时，标记 `execution_lost`；
- 未知状态的物理动作不会被自动重放。

## 6. Skill 层

`hey_robot.skills.models.Skill` 是当前 Skill contract 的事实源：

```text
name / description / parameters / handler
resources / timeout_sec
supported_robots
required_actions / required_models
```

`SkillWorker` 负责 queue、managed task、取消、订阅者、run store 和事件投影。
`SkillRunner` 负责：

1. 查找 Skill并验证 JSON Schema参数；
2. 发布 accepted/running；
3. 按 `(robot_id, resource)` 获取资源锁；
4. 创建 `SkillContext` 并执行有界 handler；
5. 归一化成功、失败、取消和 timeout；
6. 持久化并发布 terminal event。

`SkillContext` 当前提供：

- `robot`：`RobotClient`端口；
- `models`：`ModelRouter`端口；
- `observe()`；
- `progress()`；
- `raise_if_cancelled()`。

### Deployment surface

`skills.modules` 决定加载哪些 registry模块，`skills.tools` 是直接投影为 Agent tool 的
显式 allowlist。`skills.implementations` 可为同一语义 Skill选择实现。

`skills.mode` 当前只接受 `production` 或 `bringup`，但没有自动过滤逻辑；`Skill`也没有
`agent_visible`字段。因此安全审查必须检查实际 `skills.tools`，不能只看 mode名称。

## 7. Foundation Model

wire contract 的事实源是：

```text
proto/hey_robot/model_service/v1/model_service.proto
```

RPC为 `GetHealth`、`ExecuteSkill` 和 `CancelSkill`。`ModelServiceRegistry` 按
`model_services.<id>.provides` 与 `robot_id`选择服务。

### VLN

VLN Skill执行有界 observe-plan-act循环：

```text
Robot observation
  -> gRPC planner request
  -> stop / heading / pixel_goal
  -> move_base / turn_base / stop_motion
  -> fresh observation
```

它不是 SLAM或全局路径规划器，输出被转换为粗粒度局部 primitive。

### VLA

VLA Skill执行有界 observe-infer-act循环：

```text
observation + prompt
  -> one model inference step
  -> embodiment_native_action
  -> RobotRuntime safety/control plane
  -> fresh observation
```

默认 real/sim profile 不开放 VLA；实验配置需要独立模型环境、权重、GPU和真实路由验证，
不能表述为已交付的通用抓取能力。

## 8. Robot Runtime

Skill handler 通过 `RobotClient.execute()`请求动作。`LocalRobotClient`将动作转为
`RobotAction`并调用 `RobotRuntime.apply_action()`。RobotRuntime统一处理：

- driver lifecycle、capabilities和health；
- RobotSafetySupervisor；
- perception skill与observation pipeline；
- RobotControlPlane；
- reset、status和emergency stop。

`RobotManager`支持：

- Mock driver；
- XLeRobot MuJoCo driver；
- XLeRobot native driver；
- RoboCasa remote driver。

仓库还保留 SO101/LeKiwi独立 driver代码，但当前 manager的公开配置分派以以上四类为准。

## 9. Observation、Media 与事件

Driver输出 `DriverObservation`。`ObservationPipeline`检查图像质量，将大对象写入
`LocalMediaStore`，在 `RobotObservation`中只保留 `ImageRef`、`ArtifactRef`和小型
metadata。

RobotService merged loop 用一次 driver observation生成：

- `robot.observation`；
- `robot.status`；
- `robot.camera.frame.<robot_id>` raw frame packet。

Skill的内部 `skills.models.SkillEvent` 是执行事实；`SkillEventProjector`把它转换为
`protocol.SkillEvent`并发布到 bus。投影失败不会阻塞 durable Skill执行。

## 10. 持久化与一致性

当前主要存储：

- `JsonlEpisodeStore`：用户/Agent历史；
- `ConversationStore`：Agent会话上下文；
- `AgentTaskStore`：持续任务与step；
- `FileRunStore`：Skill command、事件与artifact；
- `RuntimeEventStore`：运行事件；
- `LocalMediaStore`：图像和大对象。

它们是本地 SQLite或文件存储，不构成跨存储事务。系统采用 persist-before-submit、幂等
run ID和启动 reconciliation实现最终一致性，并优先避免崩溃后重复执行物理动作。

## 11. 安全边界

当前确定性安全链为：

```text
explicit skills.tools allowlist
  -> Agent tool argument validation
  -> TaskCoordinator single-active-run rule
  -> Skill JSON Schema / resource lock / timeout
  -> RobotRuntime health / battery / estop / action checks
  -> driver validation and hardware limits
```

这些机制不构成工业安全系统。系统没有完整碰撞检测、SLAM、全局避障或硬实时安全控制；
真机仍必须隔离网络并保留物理急停或断电手段。

## 12. 当前能力边界

真实能力以所选 YAML 的 `skills.tools` 为准。主线 XLeRobot配置目前保持一个最小移动能力
面，主要包含 `inspect_scene`、`move_base` 和 `turn_base`；其他 Skill虽可注册在内置
registry中，只有显式加入 `skills.tools` 才对 Agent可见。

RoboCasa365 profile暴露 `inspect_scene`和`manipulate`。VLA/VLN实验 profile需要独立
ModelService，不能据此推断默认部署拥有稳定的通用导航或抓取能力。

## 13. 架构守卫

当前测试明确限制：

- Cognition不构造 `SkillIntent`或`RobotAction`；
- `AgentRunner`不执行IO；
- Agent只通过 `SkillClient`消费Skill事件；
- Robot Runtime不依赖 Cognition、旧 `skill_os`或Foundation backend；
- Foundation不依赖 Cognition或Robot Runtime；
- event-driven Skill execution mode保持移除；
- proto source与generated artifacts保持一致。

架构守卫以当前 native local拓扑为准。任何恢复跨进程Skill控制面的工作，都必须先新增
明确的 RemoteSkillClient/RemoteRobotClient契约和端到端测试。
