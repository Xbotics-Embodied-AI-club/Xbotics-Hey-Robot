# 部署模式边界

系统支持两种互斥的消息总线模式；选择它们是在选择部署边界，而不是性能开关。

## 单机开发：`in_memory`

`configs/mock.dev.yaml` 使用 `deployment.bus.type: in_memory`。`hey-robot run`
在一个 Python 进程内启动 Gateway、Supervisor、Agent、Skill Controller 与 Mock Robot，所有
`InMemoryBusClient` 通过同一进程内的 Hub 同步投递消息。

- 不需要 NATS；适用于本地功能开发、调试和 Mock 冒烟。
- 不提供跨进程通信、消息持久化、重连或隔离。
- 不得将任何服务拆到另一个终端、容器或主机运行。

## 多进程与真机：`nats`

仿真拆分、模型服务联调、多进程部署和真机配置必须使用
`deployment.bus.type: nats`。每个逻辑服务独立创建 NATS 客户端，消息协议才跨进程成立。

- 使用独立 `runtime_dir`，并确保同一任务数据库只有一个 Supervisor 实例写入。
- NATS 提供传输边界，不替代任务数据库的单写者约束；生产环境还应配置认证、TLS 和持久化策略。
- 不要把 `in_memory` 配置用于真机，或把 NATS 当作单机开发的必需依赖。

## 状态与准入职责

Gateway 只读任务投影、处理渠道和持有自身的投递回执；它把创建、取消、确认、急停和
reconcile 请求发布给 Supervisor。Supervisor 是 `autonomy.sqlite3` 的唯一任务状态写者。

Supervisor 的 `DispatchPreflight` 是调度预检，用于尽早拒绝明显无效的 intent。Skill
Controller 在实际执行前重新调用 `SkillContractRuntime.validate` 并检查资源冲突；这是最终
准入点，避免消息传输期间机器人状态或资源占用变化造成 TOCTOU 问题。
