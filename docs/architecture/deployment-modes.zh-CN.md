# 部署模式边界

当前系统的bus模式决定Channel/Agent消息和运行投影是否跨进程，但不改变Skill执行模式。
Skill执行目前固定为`local`。

## 单机：`in_memory`

`deployment.bus.type: in_memory`让同一Python进程中的Gateway、Agent、RobotService和
Skill事件投影共享一个Hub。

- 不需要NATS；
- 不提供跨进程通信、持久化、重连或安全隔离；
- 适合Mock、单机仿真和RoboCasa评测组合；
- 所有使用该Hub的consumer必须位于同一进程。

## 网络消息：`nats`

`deployment.bus.type: nats`用于网络Channel、独立consumer、raw frame和运行事件投影。
生产网络应配置身份凭证、TLS、topic ACL和必要的JetStream策略。

NATS当前不承载Agent到Skill的生产提交。即使使用NATS配置，以下链路仍在主进程内：

```text
Agent -> SkillClient -> SkillWorker -> LocalRobotClient -> RobotRuntime
```

因此不能仅把`hey-robot agent`和`hey-robot robot`放到不同主机，就得到等价的完整系统。
恢复这种拆分能力需要RemoteSkillClient/RemoteRobotClient及其契约测试。

## 模型与评测sidecar

以下边界是真正支持独立进程的：

- VLA/VLN ModelService：gRPC；
- RoboCasa runtime service：gRPC；
- RoboCasa policy ModelService：gRPC；
- NATS broker和外部raw frame consumer。

RoboCasa managed backend由`DeploymentRunner`启动和监控子进程，但环境所有权仍在remote
runtime sidecar，不在Foundation policy进程。

## 状态所有权

- Gateway拥有episode历史、identity binding和展示投影；
- AgentTaskStore拥有持续任务与step；
- FileRunStore拥有Skill command和SkillEvent事实；
- RobotRuntime拥有当前机器人状态；
- ModelService只拥有模型加载和短时推理状态。

这些本地存储没有跨进程共享事务。一个deployment的runtime目录应只有一个主Harness
写入者。NATS或JetStream不能替代该单写者约束。

## 故障语义

- conversation和projection消息是否可重放取决于bus配置；
- Skill执行事实先写本地RunStore，投影失败不会令动作失败；
- 主进程重启后会reconcile已持久化run；
- 没有active worker所有者的非终态run标记`execution_lost`；
- 未知状态的物理动作不自动重放。
