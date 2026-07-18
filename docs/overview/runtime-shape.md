# 部署与运行形态

Hey Robot 是一个不依赖通用 LLM Agent 框架、面向真实机器人原生构建的
Embodied Agent Harness。当前主线目标是 `XLeRobot`。

从决策层级看，它是异步快慢双系统：

```text
慢系统（长程）
  LLM Agent / task state / memory / planning / recovery
          |
          | typed SkillIntent + asynchronous feedback
          v
快系统（短时域）
  VLA/VLN / Skill OS / safety gates / Robot Runtime
          |
          v
  physical or simulated robot
```

“慢/快”描述认知和控制的时间尺度，不等同于代码模块的运行速度。VLA/VLN 单次推理仍
可能耗时；它们属于快系统，是因为其输出面向短时域具身决策。四层架构描述代码所有权，
快慢系统描述决策分工，两者是正交视角。

## 主执行链路

```text
User Channel
  -> GatewayService
      -> identity resolution
      -> episode allocation / history persistence
  -> NATS topic: user.turn
  -> AutonomousAgentService
      -> RobotAgentLoop
          -> restore -> build -> run -> save
      -> RobotAgentCore
      -> AgentRuntime
      -> request_skill / request_perception
      -> SkillGateway
  -> NATS topic: skill.intent
  -> SkillControllerService
      -> SkillContractRuntime
      -> SkillScheduler / SkillRuntime
      -> optional ModelServiceRegistry -> gRPC ModelService
  -> NATS topic: robot.action
  -> RobotService / RobotRuntime
      -> XLeRobotDriver / SO101Driver / LeKiwiDriver
  -> NATS topics: robot.status / robot.observation / skill.event / skill.result
  -> Agent execution feedback / Gateway events and replies
```

默认 `hey-robot run` 把主服务放在同一 asyncio 进程，但每个服务使用独立 NATS
client。模型服务通过 gRPC 单独部署。因此系统“协议上可拆分、默认本地一体化”，
并非所有组件都天然运行在不同机器上。

## 边界摘要

- Agent 层只做任务理解、上下文组织和 Skill 级决策。
- Skill 层负责能力契约、资源门禁、就绪检查和执行生命周期。
- Foundation Model 层负责 VLA/VLN 推理，不直接拥有任务状态或 Robot Driver。
- Robot 层负责真实硬件或仿真执行，并通过 `RobotRuntime / PerceptionService` 产出 observation 和 status。
- VLA 等外部模型能力通过 ModelService 暴露，并保持在 Skill 边界之后。

## 两条通信路径

- 主控制链：NATS event-driven 消息，包括 `UserTurn`、`SkillIntent`、`RobotAction`、
  `RobotStatus`、`RobotObservation`、`SkillEvent` 和 `SkillResult`。
- 模型边界：gRPC `ModelService`，只包含 `GetHealth`、`ExecuteSkill` 和
  `CancelSkill`。

所有消息通过 `Envelope` 传播 `trace_id`、`episode_id`、`robot_id` 和
`agent_id`。图像等大对象不进入 JSON 消息，而是落到 media store，消息只携带
`media://local/...` 引用；需要低延迟图像的 consumer 使用按机器人分区的 raw camera
frame topic。

## 当前自治边界

当前系统主要由用户 turn 驱动。`AgentRuntime` 可以在一个 turn 内通过
`wait_result` 完成多次“感知—动作—反馈”循环，Task Supervisor 负责监控、恢复标记和
通知，但不会在没有新 turn 的情况下无限唤醒 LLM。配置中的 `autonomous` 应理解为
tool-using task execution mode，而不是永久运行的后台目标追逐器。
