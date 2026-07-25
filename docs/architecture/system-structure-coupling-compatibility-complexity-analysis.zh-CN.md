# 系统结构、耦合、兼容性与复杂度分析

> 审计基线：2026-07-25，提交 `c4a9f36`。本文以当前代码和配置为事实源，
> 历史设计文档仅用于解释演进背景。

> 2026-07-25 后续更新：Robot 子系统已经按本文发现的问题完成无兼容层重构。
> 当前边界和删除清单见 [Robot Runtime 边界重构](robot-runtime-refactor.zh-CN.md)。

## 1. 结论摘要

Hey Robot 当前已经收敛为“单 Agent、进程内 Skill Worker、进程内 Robot Client、
外部 gRPC ModelService”的具身 Agent Harness。目录层次、端口抽象、安全检查和测试基础
较好，但代码架构已经领先于部分旧文档：旧文档描述的
`SkillGateway -> skill.intent -> SkillControllerService -> robot.action` 不再是主执行链。

综合判断：

| 维度 | 评价 | 主要依据 |
|---|---:|---|
| 结构清晰度 | 7/10 | composition root 明确，Agent、Skill、Model、Robot 的所有权总体清楚 |
| 实际解耦 | 5/10 | 端口抽象良好，但 Skill 和 Robot 主链只支持进程内执行 |
| 兼容性 | 6/10 | Windows/Linux 和多种 driver 有适配，完整模型能力仍依赖 Linux/CUDA 和隔离环境 |
| 复杂度 | 6/10 | 总规模可控，配置解析、Gateway、Web、仿真 driver 和 VLA/VLN 已形成热点 |
| 测试质量 | 8/10 | 715 个测试通过，覆盖率 85.63%，生产源码 mypy 与 Ruff 通过 |
| 架构一致性 | 5/10 | 代码内部较一致，协议遗留、重复模型和旧文档曾造成双重架构叙事 |

最高优先级不是局部重构，而是保持“当前执行拓扑、部署边界、Skill 契约”只有一个事实
版本。本文与同步更新后的架构文档共同记录这一事实版本。

## 2. 规模与验证基线

本次静态盘点结果：

- `src/hey_robot` 约 33,986 行 Python；
- `robot_runtime` 从 14,440 行降至 2,421 行；后端、硬件、传输、媒体和 RoboCasa
  环境服务已经使用独立包；
- `tests` 约 18,497 行 Python；
- 219 个生产 Python 文件；
- 715 个测试全部通过；
- 测试覆盖率 85.63%；
- `ruff check src tests scripts evaluation` 通过；
- `mypy src` 通过（218 个纳入检查的源码文件）。

仓库根目录仍有一份旧 `poe_tasks.toml`，其 mypy 任务检查 `src tests`，会得到 126 个
测试替身相关类型错误；当前 `pyproject.toml` 中的正式任务检查 `src`。工程入口应继续以
`pyproject.toml` 为准，并删除或显式标记旧任务配置，避免不同环境使用不同质量口径。

## 3. 实际运行结构

当前 `hey-robot run` 的主链如下：

```text
Web / Voice / Feishu / CLI
             |
             v
GatewayService
  identity / episode / history / presentation
             |
             | NATS: conversation.turn
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
             |                    |
             |                    +-- gRPC -> ModelService (optional)
             v
LocalRobotClient -> RobotRuntime -> RobotDriver
```

`DeploymentRunner` 是主 composition root。它创建 `RobotService`，再通过
`build_local_runtime_components()` 从 `RobotService.runtimes` 构造
`LocalRobotClient`、`SkillWorker` 和 `ModelServiceRegistry`，最后把同一个
`SkillClient` 注入 `AutonomousAgentService`。

NATS 当前承担：

- Channel/Gateway 与 Agent 的会话消息；
- robot observation/status 和 raw camera frame；
- Skill 事件的只读投影；
- Human Follow 的流式控制入口；
- 保留的 `robot.action` 兼容入口。

NATS 不再承担 Agent 到 Skill 的生产提交。配置校验明确拒绝非 `local` 的
`skills.execution_mode`。因此当前系统应表述为：

> 模型服务可独立部署；主 Harness 的 Agent、Skill Worker 和 Robot Runtime 在
> `hey-robot run` 中进程内组合。Channel/Agent 消息和运行投影使用 NATS 或 in-memory bus。

## 4. 分层与所有权

### 4.1 Gateway 与 Channel

Gateway 负责身份绑定、episode 分配、历史、渠道投递、运行事件展示和安全控制命令。
它不决定具体机器人动作。Web、Voice、Feishu 和 CLI 通过 Channel 边界归一化输入。

### 4.2 Cognition

`AutonomousAgentService` 为 session 创建有状态 `Agent`。`AgentRunner` 是纯模型决策器：
它接收模型消息和允许的工具定义，返回文本或一个 typed proposal，不执行外部 IO。
`AgentToolExecutor` 执行 harness tool，或把物理 Skill proposal 交给 `TaskCoordinator`。

Agent 每次唤醒最多运行 8 次模型决策。遇到物理 Skill 后返回 `waiting`，Skill 终态事件
再触发恢复。这是 turn-driven 的有限自治，不是永久后台目标循环。

### 4.3 Skill

`Skill` dataclass 是当前可执行能力的事实源，字段为：

- `name`、`description`、`parameters`、`handler`；
- `resources`、`timeout_sec`；
- `supported_robots`；
- `required_actions`、`required_models`。

`SkillWorker` 管理 command queue、运行 task、取消、订阅者和持久化；`SkillRunner` 负责
参数校验、资源锁、timeout 和 lifecycle event；handler 通过 `SkillContext.robot`、
`SkillContext.models`、`observe()`、`progress()` 和 `raise_if_cancelled()` 使用下层能力。

需要特别说明：当前没有 `agent_visible`、嵌套 `ctx.run()`、Skill dependency graph、
success criteria 或 recovery hints 等契约字段。`skills.mode` 目前只校验取值，真正的
Agent 能力面由 `skills.tools` 显式 allowlist 决定。

### 4.4 Foundation Model

VLA/VLN 通过 gRPC `ModelService` 独立部署。wire contract 的事实源是
`proto/hey_robot/model_service/v1/model_service.proto`，RPC 包括 `GetHealth`、
`ExecuteSkill` 和 `CancelSkill`。`ModelServiceRegistry` 按 capability name 和
`robot_id` 路由。

### 4.5 Robot Runtime

`RobotRuntime` 拥有 driver lifecycle、observation materialization、action safety、
control plane、status 和 reset。`RobotManager` 根据配置选择 Mock、MuJoCo、XLeRobot
native 或 RoboCasa remote driver。Agent 不导入 driver primitive；Skill 通过
`RobotClient` 端口访问动作。

## 5. 耦合分析

### 5.1 正向评价

- `AgentRunner` 与 IO、Robot、持久化分离；
- Foundation backend 不反向依赖 Cognition；
- Robot Runtime 不依赖 Cognition；
- `SkillClient`、`RobotClient`、`ModelRouter` 和 `MessageBus` 提供了可测试端口；
- composition 集中在 `app`，没有依赖全局 service locator；
- Robot 安全检查位于 Runtime，LocalRobotClient 不会绕开最终安全边界。

### 5.2 包级强耦合区

原审计中，`config`、`bus`、`events`、`foundation`、`skills`、`persistence`、
`robot_runtime` 曾形成概念上的强连通区。此次重构已经处理其中 Robot 相关的三项：

- Driver primitive 契约和 inventory 分别移入 `robot_api` 与 `config`；
- `RobotDriverContext` 不再依赖 `RobotSpec`；
- Skill 与 Runtime 从 `robot_api` 共享 `RobotActionSpec` 和 `RobotClient`；
- `persistence.run_store` 直接持有 Skill domain model；
- Skill handler 同时依赖 ModelRouter 与 RobotClient。

Robot Runtime 相关依赖不再构成上述反向耦合；Persistence 与 Skill domain model 的关系
仍维持现状。

### 5.3 双执行入口

主链通过 `LocalRobotClient` 直接调用 `RobotRuntime.apply_action()`；与此同时，
`RobotService` 仍订阅 `robot.action`。Bus 路径会发布更完整的 received/executed 事件，
Local 路径不会经过同一段事件代码。因此两条路径的可观察性语义并不完全相同。

建议明确其一：

1. 将 `robot.action` 标为外部兼容/调试入口，并让两条路径共享同一个 application service；
2. 或引入正式 `RemoteRobotClient`，把 local/remote 做成等价适配器并添加契约测试。

### 5.4 重复事件模型

系统同时存在 `protocol.messages.SkillEvent/SkillResult` 和
`skills.models.SkillEvent/SkillResult`。前者是总线 DTO，后者是执行事实；
`SkillEventProjector` 手工转换 phase、ID 和 result 字段。

内部模型与 wire model 分离是合理的，但同名会提高误用概率。建议重命名为
`SkillExecutionEvent` 与 `SkillEventProjection`，并为转换添加完整字段契约测试。

### 5.5 持久化一致性

Conversation、Agent task、Skill run、episode、runtime event 和 media 分布在 SQLite、
JSON/JSONL 和文件目录中，无法共享事务。当前通过 persist-before-submit、run ID 幂等、
启动 reconciliation 和 `execution_lost` 终态实现最终一致性。

这更接近 saga，而不是 exactly-once。其安全取舍是“崩溃后不自动重放未知状态的物理
动作”。这个选择合理，但必须作为正式故障语义写入运维说明。

## 6. 兼容性分析

### 6.1 操作系统与硬件

- Python 严格限定为 3.12；
- uv lock 环境声明 Linux 与 Windows，不声明 macOS；
- 真机支持 Windows COM 和 Linux `/dev/tty*`；
- Camera 对 Windows DShow 和 Linux V4L2 做适配；
- MuJoCo 根据平台选择 WGL、EGL 或 OSMesa；
- 完整 VLA/VLN、RoboCasa365 主要面向 Linux x86_64、CUDA 和独立环境。

因此“主 Harness 与原生机器人路径支持 Windows/Linux”基本成立，但“完整具身模型栈
跨平台”不成立。

### 6.2 Python 依赖

项目已经把 `sim`、`vla`、`vln` 和 `robocasa365` 分组，并声明 VLA/VLN/RoboCasa 的
互斥关系。RoboCasa依赖固定到 commit，gRPC/protobuf 也有明确范围，这是优点。

核心依赖仍包含 Torch、Ultralytics、音频、Feishu、OpenCV、串口和舵机 SDK。仅运行
Web/CLI Agent 也需要解析较重的硬件和媒体依赖。后续适合继续拆分 channel、robot-native
和 perception extras，降低 Windows、headless 和 CPU-only 环境的安装风险。

### 6.3 模型 API

Agent LLM 使用 OpenAI-compatible Chat Completions，供应商切换成本较低，但仍假设对方
兼容 streaming tool calls、`max_tokens`、可选 `reasoning_effort` 和 multimodal
`image_url`。Provider兼容应通过契约测试验证，而不能只依据“OpenAI-compatible”名称。

### 6.4 协议演进

ModelService 使用版本化 proto `v1`，演进边界清楚。NATS JSON dataclass 消息缺少显式
schema version 和 producer version。如果这些消息只作为单部署内投影，风险可接受；若要
恢复跨版本多进程控制面，需要补充 schema version、兼容规则和 contract tests。

## 7. 复杂度分析

以下为 AST 分支结构的近似复杂度，不等同于正式 McCabe 值，但足以定位维护热点：

| 热点 | 行数/近似复杂度 | 风险 |
|---|---:|---|
| `DeploymentConfig.from_dict` | 268 行 / 约 98 | 手写解析所有配置，新增字段影响集中 |
| `WebChannel.start` | 262 行 / 约 28 | HTTP、WebSocket、页面和生命周期混合 |
| `XLeRobotSimDriver._apply_action` | 224 行 / 约 40 | 动作分派、仿真状态和失败处理混合 |
| `VLAOption.run` | 214 行 / 约 15 | 观察、推理、执行与终止集中 |
| `VLNOption.run` | 153 行 / 约 19 | planner适配与闭环控制集中 |
| `Agent._drive` | 98 行 / 约 22 | 模型循环、任务状态和反馈路由集中 |
| `GatewayService` | 947 文件行 | 身份、历史、安全命令、投影和展示职责较多 |
| `MockRobotDriver` | 1,262 文件行 | test double 已演化为小型世界模拟器 |

Ruff 当前显式忽略 `C901`，因此复杂度不会阻塞质量门禁。建议先给热点添加 characterization
tests，再按状态机、adapter 和 route group 拆分，避免纯机械切文件。

## 8. 测试与架构守卫

当前测试覆盖 Agent恢复与 steer、Skill幂等与取消、Robot safety、硬件 SDK、MuJoCo、
VLA/VLN、gRPC 和 RoboCasa。架构测试还能阻止 Cognition 直接构造 RobotAction、Robot
Runtime 反向依赖上层等明显回退。

现有架构测试主要依赖源码字符串和正则，不能识别间接依赖、动态 import、运行时拓扑和
文档漂移。建议增加：

- 基于 AST 的顶层包依赖图和允许边检查；
- LocalRobotClient 与未来 RemoteRobotClient 的契约测试；
- 文档引用的类、模块、CLI action 和 topic 存在性检查；
- production/bringup语义测试，或删除没有运行语义的 mode；
- 协议投影的全字段 round-trip/compatibility测试。

## 9. 风险优先级与建议

### P0：维持唯一架构事实源

- 当前主链只描述 native local Skill execution；
- 不再把 `skill.intent`、SkillController或TaskSupervisor写成当前服务；
- 若恢复远程 Skill执行，必须新增正式 `RemoteSkillClient`，而不是只恢复 topic；
- 将历史方案放入 reference/roadmap，不与运行手册混写。

### P0：明确 Skill surface 的真实安全语义

- 当前 `skills.tools` 是唯一 Agent能力 allowlist；
- `skills.mode` 不会自动过滤 primitive；
- 若需要 production自动过滤，应先扩展 Skill contract 和部署校验，再更新文档承诺。

### P1：收敛双路径和重复模型

- 统一 Local与Bus Robot执行的事件语义；
- 区分执行事件和投影 DTO；
- 将旧 topic 标成兼容面或删除无生产消费者的控制协议。

### P1：拆分复杂度热点和核心依赖

- 优先拆配置解析、Web route composition、仿真动作分派和 VLA/VLN loop；
- 把 Voice、Feishu、native robot和大型视觉模型依赖从 core 中进一步分组。

### P2：统一工程质量入口

- 以 `pyproject.toml` 为唯一 Poe/Ruff/mypy/pytest配置；
- 清理旧 `poe_tasks.toml`；
- 明确 tests 是否纳入 mypy。如果纳入，应先修复 test double 的 Protocol类型，而不是
  依赖两套命令口径。

## 10. 相关事实文档

- [系统架构](system-architecture.md)
- [部署与运行形态](../overview/runtime-shape.md)
- [Agent 与 Skill 边界](agent-skill-boundaries.md)
- [部署模式边界](deployment-modes.zh-CN.md)
- [Skill 扩展指南](../development/skill-extension.md)
