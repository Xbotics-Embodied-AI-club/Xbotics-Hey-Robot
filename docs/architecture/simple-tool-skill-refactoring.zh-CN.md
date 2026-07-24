# Hey Robot Tool 与 Skill 简化重构方案

## 1. 文档状态

本文是当前重构后的架构说明。状态快照：2026-07-24，基线 commit `de207f1` 加当前工作树改动。
本轮已经按“保持简单，不保留兼容”的方向完成核心切换：

- `skill_os` 源码目录已删除；
- `SkillControllerService`、legacy Skill Runtime、legacy builtin module 和 legacy Skill transport
  已退出生产路径；
- `skills.execution_mode` 只支持 `local`；
- 部署配置统一使用 `hey_robot.skills.builtins` 与 `skills.tools`；
- `ActionProposal`、`ShortOperationCommand`、`SkillCatalog/SkillSpec`、`RobotExecutionGateway`
  和 `legacy_catalog` 已从生产代码删除；
- `human_follow` 不再伪装成 Skill Tool，保留为独立 service/client；
- RoboCasa 当前只暴露 native `inspect_scene`，`manipulate` 等 native action contract 闭合后再加入。

按本次“简化系统、删除兼容、默认 native local 可运行”的代码级目标估算，整体完成度为
**100%**。RoboCasa/VLA native action contract、固定 benchmark 和真机长跑属于后续验证/产品化
工作，不阻塞本次简化重构完成。

具体文件、接口、数据库 migration、分 PR 顺序和测试改造见
[Tool 与 Skill 代码级重构计划](simple-tool-skill-refactoring-code-plan.zh-CN.md)。

重构目标是将现有 Skill OS 收缩为一个简单、通用、可观察的 Embodied Agent Harness，
同时保留长程任务所需的异步快慢双系统、分布式部署、分层解耦、真实机器人安全边界、
VLA 进程隔离和任务完成验证。简单化针对的是协议数量和重复抽象，不是将系统限制为单进程
或取消必要的服务边界。重构必须能够按阶段合并，任何阶段都不应要求同时替换全部运行链路。

设计参考：

- [Harness Engineering for Self-Improvement](../references/Harness%20Engineering%20for%20Self-Improvement.md)
- [Harness VLA](../references/harness_VLA.md)
- [Agent 与 Skill 边界](agent-skill-boundaries.md)
- [部署模式边界](deployment-modes.zh-CN.md)
- [ModelService RPC](model-service-rpc-proto.md)

## 2. 问题陈述

重构前，Agent 通过单一 `request_skill(skill, objective, slots)` Tool 提交操作。Skill 名称
和真实参数被放在通用 `slots` 对象中，模型无法在 function calling schema 中直接看到每个
Skill 的参数约束。Skill catalog 只能作为额外 JSON 文本加入上下文。

重构前的 legacy 执行路径包含这组相邻抽象，当前已从生产代码删除：

```text
Agent
  -> RequestSkillTool
  -> ActionProposal
  -> RobotExecutionGateway
  -> ShortOperationCommand / SkillIntent
  -> NATS
  -> SkillControllerService
  -> SkillScheduler
  -> SkillContractRuntime
  -> SkillRuntime
  -> SkillContext
  -> API / Port adapter
  -> Robot Runtime / ModelService
```

这些抽象分别解决了部分合理问题，但整体存在以下重复；本轮已经删除其中的兼容执行层：

- `SkillSpec`、`SkillContract` 和 Tool description 重复描述同一能力；
- `SkillCatalog`、`SkillContractCatalog` 和 `SkillRegistry` 重复维护能力集合；
- `SkillContractRuntime`、`SkillRuntime` 和 `SkillControllerService` 共同参与准入和执行；
- `apis.py` 与 `ports.py` 为相同调用增加了两层接口；
- NATS 传输语义渗透到 Controller 和 Gateway，导致 ID 关联、Future、超时和 late status
  分散在业务代码中；
- VLA、经典算法和组合 Skill 容易继续演化出彼此独立的框架。

这条路径已经从生产代码中删除。重构继续减少核心机制，但不删除 Robot Runtime 的物理安全边界。

### 2.1 当前问题收敛状态

独立 Skill Tool 已替代模型侧的 `request_skill` 聚合接口，因此 Tool schema 问题已经基本
解决。执行和部署层已经收敛为 native local：

```text
Agent -> TaskCoordinator -> LocalSkillClient -> SkillRunner -> native Skill
```

`request_skill` 聚合接口和旧 Controller 执行路径均已退出当前主线。后续重点是补齐 native
RoboCasa/VLA contract 和真机/benchmark 验证。

## 3. 设计原则

### 3.1 Skill 是能力，Tool 是接口投影

系统内部只维护一份 Skill 定义。Agent Tool schema 从 Skill 自动生成：

```text
Skill definition -> Agent Tool definition
                 -> runtime validation
                 -> execution handler
```

Agent 可以直接调用 `pick(object="cup")`，但该调用仍然经过受控的 SkillRunner。Agent
直接调用 Tool 不等于 Agent 直接访问 Driver。

### 3.2 一个执行入口

顶层 Skill 和嵌套 Skill 都由同一个 `SkillRunner.run()` 执行。参数校验、资源互斥、超时、
取消和事件记录只实现一次。

### 3.3 快慢双系统通过异步任务协议连接

慢系统负责长程任务的理解、分解、记忆、恢复和根任务验证；快系统负责有界 Skill、VLA
闭环和机器人控制。两者只通过少量稳定消息交互：

```text
SkillCommand: submit / cancel
SkillEvent: accepted / running / progress / completed / failed
```

慢系统不等待每个底层动作，也不直接参与 VLA action loop。快系统不拥有长程任务规划和
根任务完成判断。

### 3.4 逻辑接口与传输解耦

- 同一套 `SkillClient` 可以由本地 async queue 或 NATS 实现；
- 分布式异步命令和事件使用 NATS；
- 有明确请求/响应语义的 ModelService 和远程 Robot Runtime 使用 RPC；
- durable state 使用 JSONL 或数据库。

业务模块依赖 `SkillClient`、`RobotClient`、`ModelClient` 和 `EventStore`，不直接依赖 NATS
subject、protobuf stub 或具体部署拓扑。

### 3.5 算法类型不是框架类型

Classic、VLA 和 Hybrid 都是普通 Skill 实现，不分别建设 `ClassicSkillRuntime`、
`VLASkillRuntime` 或 workflow engine。

### 3.6 安全和验证位于 Agent 之外

Robot Runtime 保持对动作限位、速度、碰撞、急停和观测帧一致性的最终控制。根任务完成
不能由动作成功或 VLA 的 `done` 单独决定。

### 3.7 需求出现后再抽象

新抽象至少需要两个真实用例和对应行为测试。若删除某个抽象不会导致行为测试失败，说明
它暂时不应成为核心运行机制。

## 4. 目标架构

```text
Slow system: Embodied Agent Harness
  TaskStore / Agent / Memory / Root Verifier
       -> pick(...) / place(...) / vla_act(...) / move_to(...)
       -> SkillToolAdapter
       -> SkillClient.submit(SkillCommand)
                         |
              local queue or NATS
                         |
Fast system: Skill execution
  SkillRunner
       -> Skill.execute(context, arguments)
           -> context.observe()
           -> context.robot.call()
           -> context.models.infer()
           -> context.run(child_skill)
       -> SkillEvent stream
                         |
              local queue or NATS
                         |
Slow system wakes on terminal/progress events
  -> observe / retry / replan / verify / complete

External boundaries:
  Skill -> gRPC ModelService
  Skill -> local or gRPC Robot Runtime
  Commands/events -> NATS in distributed deployment
  Durable task/trajectory state -> SQLite / JSONL / artifacts
```

目标目录：

```text
src/hey_robot/
  skills/
    models.py
    context.py
    registry.py
    runner.py
    client.py
    transport/
      local.py
      nats.py
    builtins/
      perception.py
      navigation.py
      manipulation.py
      safety.py
  cognition/tools/
    skill_tools.py
    task_tools.py
  events/
    jsonl.py
    store.py
  robot_runtime/
  foundation/
    clients/
    transport/grpc/
```

## 5. 核心模型

### 5.1 Skill

第一版只保留参与运行决策的字段：

```python
@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    parameters: dict[str, Any]
    execute: SkillHandler
    resources: tuple[str, ...] = ()
    timeout_sec: float = 60.0
```

以下字段不进入第一版核心模型：

```text
goal_effects
recovery_hints
cannot_satisfy
failure_modes
dependencies
feedback_mode
level
category
required_model_service
driver_primitives
```

必要的文档信息可以保存在开发文档中；出现实际调度或验证需求后，再将对应字段加入运行时。

### 5.2 SkillResult

```python
@dataclass
class SkillResult:
    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    error: str | None = None
```

`success` 表示当前 Skill 的后置条件是否满足，不自动表示根任务已经完成。根任务由独立
verifier 根据 evidence 判断。

### 5.3 SkillCommand 与 SkillEvent

跨快慢系统只使用两个核心协议对象。`SkillCommand` 表示不可变的执行请求：

```python
@dataclass(frozen=True)
class SkillCommand:
    run_id: str
    name: str
    arguments: dict[str, Any]
    robot_id: str
    trace_id: str
    deadline: float | None = None
```

`SkillEvent` 表示追加式状态变化：

```python
@dataclass(frozen=True)
class SkillEvent:
    run_id: str
    sequence: int
    phase: str
    timestamp: float
    progress: float | None = None
    result: SkillResult | None = None
```

`run_id + sequence` 用于去重和恢复。事件 phase 第一版仅保留：

```text
accepted
running
progress
completed
failed
cancelled
```

传输层不得重新定义另一套 Skill 生命周期。

### 5.4 SkillContext

```python
@dataclass
class SkillContext:
    robot: RobotClient
    models: ModelClient
    observe: Callable
    run: Callable
    emit: Callable
    cancelled: Callable
```

Context 提供少量通用操作，不为每个 Driver primitive 同时增加 Protocol 和 Port 方法。

### 5.5 SkillRegistry

Registry 是 Skill 集合的唯一事实来源，负责：

- 注册和名称唯一性；
- 按名称查找；
- 根据显式白名单生成 Tool definitions。

Registry 不负责资源调度、模型服务选择或执行状态。

### 5.6 SkillRunner

Runner 是唯一执行入口，只负责通用执行机制：

```text
lookup
-> validate arguments
-> acquire resources
-> emit started
-> execute with timeout and cancellation
-> normalize exception
-> emit finished
-> return SkillResult
```

Skill 组合直接使用 `context.run(name, arguments)`，第一版不引入 DAG 或 workflow engine。

### 5.7 SkillClient

慢系统只依赖异步客户端接口：

```python
class SkillClient(Protocol):
    async def submit(self, command: SkillCommand) -> str: ...
    async def cancel(self, run_id: str, reason: str) -> None: ...
    async def events(self, run_id: str) -> AsyncIterator[SkillEvent]: ...
```

当前只提供 `LocalSkillClient`：单进程测试和开发通过 async queue 连接 Runner。真机仍可使用
`RobotClient`/gRPC 边界，但不为 Skill execution 额外引入 NATS command transport。

## 6. Agent Tool 设计

每个 Agent 可见 Skill 生成一个独立 function Tool：

```python
def skill_to_tool(skill: Skill) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": skill.name,
            "description": skill.description,
            "parameters": skill.parameters,
        },
    }
```

模型调用：

```json
{
  "name": "pick",
  "arguments": {
    "object": "red cup"
  }
}
```

不再要求模型生成：

```json
{
  "name": "request_skill",
  "arguments": {
    "skill": "pick",
    "slots": {
      "object": "red cup"
    }
  }
}
```

Tool adapter 只负责把 Tool call 转成 `SkillCallProposal`，再由 `TaskCoordinator` 提交
`SkillCommand` 给 `SkillClient`。`ActionProposal` 已删除，不再作为模型接口或内部兼容 DTO。

Tool call 对模型表现为一次调用，但 Harness 内部是异步任务：提交后持久化 `run_id`，慢系统
挂起当前审议，收到 progress 或 terminal event 后再唤醒 Agent。这样不会让一次 LLM 请求
常驻等待整个机器人执行过程，也不会把快系统的每个动作塞进慢系统上下文。

非机器人执行工具单独保留，例如：

```text
complete_task
control_task
ask_user
```

## 7. Tool Surface

Agent 可见能力使用显式配置，不在多个运行时字段中推导。

产品模式提供稳定的语义能力：

```yaml
agent:
  tools:
    - inspect_scene
    - navigate_to
    - pick
    - place
    - manipulate
```

RPent/Harness 实验模式提供少量有界操作能力：

```yaml
agent:
  tools:
    - observe
    - move_to
    - move_pose
    - rotate_wrist
    - set_gripper
    - release
    - vla_act
    - verify
```

两种模式只改变 Tool 白名单，不改变 Registry、Runner 或 Robot Runtime。

当未来 Skill 数量过多时，应先根据机器人、任务和权限选择一个小的 Skill 子集，再生成独立
Tool；不应退回到无类型的通用 dispatcher。

## 8. Classic、VLA 与 Hybrid

### 8.1 Classic Skill

经典算法 Skill 在实现内部使用感知、几何、IK 和 RobotClient：

```text
observe -> locate -> plan -> robot.call -> verify
```

### 8.2 VLA Skill

`vla_act` 表示有界的局部操作 option，而不是一次裸模型 forward：

```text
observe
-> model.infer
-> bounded robot action
-> fresh observation
-> local termination check
-> repeat until done or max_steps
```

VLA 的单步动作或 action chunk 不直接暴露给 Agent。

### 8.3 Hybrid Skill

混合策略使用普通 Python 组合：

```text
approach_object
-> move_to pre-contact pose
-> vla_act local contact operation
-> analytic transport
-> verify
```

默认产品 Tool 名称保持稳定，例如始终使用 `pick`。具体实现通过部署配置选择：

```yaml
skills:
  implementations:
    pick: hybrid
    place: classic
```

需要研究 Agent 自主策略选择时，Harness 模式才同时暴露 `move_to`、`vla_act` 等 Tool。

## 9. 通信边界

### 9.1 保留 RPC

以下边界继续使用 gRPC：

- VLA/VLN ModelService；
- 独立 RoboCasa backend；
- 未来确实需要独立进程运行的 Robot Runtime。

Skill 只依赖 `ModelClient` 或 `RobotClient`，不直接依赖 protobuf 和 channel。客户端可以有
local 与 gRPC 两种实现。

### 9.2 传输与长程任务边界

当前核心 Harness 只支持单进程 `LocalSkillClient`。NATS 可以继续用于现有 Gateway、通知和
遥测，但不再承载 `SkillCommand`、`SkillEvent` 或 worker lifecycle；因此没有 NATS receipt、
JetStream、redelivery 和第二套 worker state machine。跨进程 worker 是独立部署需求，出现前不
保留实现或抽象占位。

### 9.3 最小持久化布局

第一版不建设通用 TaskStore/EventStore 平台，也不引入外部数据库。使用 SQLite 文件和普通
运行文件即可：

```text
runtime/<deployment_id>/
  harness.sqlite3
  skill_worker.sqlite3
  runs/
    <run_id>/
      events.jsonl
      result.json
      observations/
      artifacts/
```

职责划分：

| 数据 | 存储 | 原因 |
|---|---|---|
| active task、task step、submitted command、最后 event sequence | `harness.sqlite3` | 慢系统需要唯一约束、原子状态转换和按 ID 查询 |
| command receipt、active run、terminal result | `skill_worker.sqlite3` | 快系统需要独立的执行幂等和重启恢复 |
| Skill/VLA 完整轨迹 | `events.jsonl` | 追加式、可检查、便于离线分析 |
| 最终结果摘要 | `result.json` | 人和工具都可以直接读取 |
| 图片、视频、深度图、模型输出 | `artifacts/` | 避免大对象进入数据库或消息总线 |

SQLite 本身是一个无需服务进程的本地文件，仍然符合 file-system-as-memory 原则。不要为了
“文件化”把 active task 写成一个反复覆盖的 JSON 文件；这会重新实现文件锁、唯一性、原子
更新、查询和崩溃恢复。

当前已有的 `sustained_tasks.sqlite3` 可以在第一阶段继续使用，不要求为重构先迁移数据。
后续再把 conversation、submitted command 等慢系统 SQLite 文件合并为 `harness.sqlite3`；
worker receipt 合并为 worker 本地的 `skill_worker.sqlite3`。现有
RuntimeEventStore 和 SkillStore 的 JSONL 可迁移到每个 run 目录；全局 runtime event 仅用于
观测，不作为任务事实来源，允许按容量轮转或删除。

分布式第一版仍采用单写者：慢系统拥有 `harness.sqlite3`，Skill worker 拥有本机
`skill_worker.sqlite3` 并写 run artifact。两者通过 SkillCommand/SkillEvent 协作，不跨主机
共享 SQLite 文件；事件只返回 artifact reference。如果以后节点之间需要直接读取彼此的
artifact，再增加对象存储 adapter，不提前引入。

## 10. 渐进式迁移计划

状态快照下各阶段进展如下。详细文件证据、PR 0--9 映射和测试缺口以
[代码级重构计划](simple-tool-skill-refactoring-code-plan.zh-CN.md#120-当前迁移状态)为准。

| 阶段 | 当前状态 | 说明 |
|---|---|---|
| 0 行为基线 | 本轮完成 | 本轮使用现有 regression/smoke 覆盖简化主链；固定 RoboCasa/VLA benchmark 后续单独补 |
| 1 新内核 | 完成 | models/context/registry/runner/resource/local worker 已落地；不保留 legacy adapter |
| 2 直接 Tool 化 | 完成 | 独立 Tool 和 `SkillCallProposal` 已成为 Agent 主接口；`ActionProposal` 已删除 |
| 3 Builtin 迁移 | 完成 | 简单 builtin、classic、bounded VLA/VLN loop 与 oracle simulation dock 已迁移；legacy builtin/adapter/termination 已删除 |
| 4 Runner 唯一入口 | 完成 | local native 已切换；submit 失败回填、terminal resume 和 startup reconcile 已接通；旧 Controller 已删除 |
| 5 transport 收敛 | 已收敛 | local transport 是唯一新协议执行链；未使用的 NATS event-driven bridge 已删除 |
| 6 删除旧抽象 | 完成 | `skill_os`、Gateway、Controller、legacy catalog/transport 和旧 proposal/short command DTO 已删除 |

### 阶段 0：冻结行为基线

- 记录当前 classic、VLA 和 RoboCasa 代表任务结果；
- 固定 checkpoint、seed、任务和最大步数；
- 补齐 Tool schema、timeout、cancel、急停、VLA payload 和完成验证测试；
- 保存当前事件轨迹作为兼容对照。

### 阶段 1：增加新内核

新增 `skills/models.py`、`context.py`、`registry.py` 和 `runner.py`。这些模块当前已落地；本阶段
最初“不切换生产链路”的迁移约束仍用于解释提交顺序。

新内核首先覆盖：

- schema 校验和默认值；
- timeout；
- cancel；
- resource lock；
- `context.run()`；
- 异常归一化；
- JSONL 事件。

### 阶段 2：直接 Tool 化

本阶段主体已完成；以下条目保留为无回退要求：

- 从启用的 Skill 自动生成独立 Tool；
- Agent 停止默认使用 `request_skill`；
- Tool call 直接转为 `SkillCallProposal`；
- 底层执行和通信链路已切到 native local。

### 阶段 3：迁移 Builtin Skill

本阶段核心代码已完成。低风险 builtin、classic/VLA/hybrid、bounded VLN/VLA loop 和 oracle
simulation dock 已迁移；不能把这等同于 RoboCasa/VLA/真机 benchmark 已完成。

当前部署配置统一使用 native local。`configs/evaluation/robocasa365.agent.yaml` 只暴露
`inspect_scene`，因为 remote RoboCasa `manipulate` 的 native action contract 尚未完成验证。

按风险从低到高迁移：

```text
stop_motion
set_gripper
inspect_scene
navigation
classic pick/place
vla_act
hybrid pick/place
```

迁移期间允许使用单向 `LegacySkillAdapter`，但新 Skill 不得反向依赖旧 Controller。

### 阶段 4：Runner 成为唯一执行入口

native local 已使用 Runner；全局唯一入口尚未达成。

将 Controller 的职责分别移交给：

| 当前职责 | 目标位置 |
|---|---|
| Skill 查找 | `SkillRegistry` |
| 参数校验、timeout、cancel、resource | `SkillRunner` |
| VLA 调用 | `ModelClient` |
| Robot 动作 | `RobotClient` |
| 事件记录 | `EventSink` |
| 硬件安全 | `RobotRuntime` |

### 阶段 5：传输层收敛

只保留 `LocalSkillClient -> async queue -> SkillRunner`。未被部署配置使用的 NATS event-driven
bridge 已删除，避免维持两套 command/event 协议。真机 Robot Runtime 如需独立进程，使用
`GrpcRobotClient` 承载有明确响应语义的数据面调用。

### 阶段 6：删除旧抽象

本阶段已经完成，不保留兼容目录或正式执行路径。

所有 Builtin 和部署入口完成切换后，删除或合并：

```text
skill_os/
RobotExecutionGateway
SkillControllerService
ActionProposal
ShortOperationCommand
SkillCatalog/SkillSpec
legacy Skill transport
RequestSkillTool
```

合并关系：

```text
SkillSpec + SkillContract       -> Skill
SkillCatalog + ContractCatalog  -> SkillRegistry
SkillRuntime + Controller core  -> SkillRunner
Controller transport handling   -> LocalSkillClient + SkillRunner
lifecycle + event_sink          -> EventSink + JSONL
```

## 11. 安全边界

重构不得删除以下约束：

- Agent 不直接提交 Driver action；
- Robot Runtime 对所有来源的动作执行最终安全检查；
- emergency stop 和 cancel 不依赖模型判断；
- VLA 输出必须经过 action adapter 和 Robot Runtime；
- action success、Skill success 和 root task success 分开记录；
- verifier 和评测数据不允许被 Skill 或自改进逻辑覆盖；
- 所有执行都有 trace ID、参数、结果和 evidence 记录。

## 12. 验收标准

最终重构必须满足：

1. Agent Tool 使用 Skill 的真实 JSON Schema，不再依赖任意 `slots`；
2. Classic、VLA 和 Hybrid 通过同一个 SkillRunner；
3. 顶层和嵌套 Skill 使用同一执行入口；
4. 本地完整运行通过 LocalSkillClient 完成，不需要 NATS；
5. Agent 在终端事件到达后可从持久任务状态恢复审议；
6. VLA 和 RoboCasa gRPC 集成测试继续通过；
7. timeout、cancel、资源互斥、观测新鲜度和急停有独立测试；
8. 根任务完成仍由独立 verifier 根据 evidence 判断；
9. RoboCasa 相同 checkpoint 和 seed 的行为不低于重构前基线；
10. JSONL 轨迹足以复现一次 Skill 的输入、关键步骤和结果；
11. 不存在两套正式执行路径；Agent/native Skill 主链不再保留兼容 DTO。

### 12.1 当前验收结论

| 条目 | 当前状态 | 说明 |
|---|---|---|
| 1 独立真实 Tool schema | 已满足 | 配置 Skill 已投影为独立 Tool，不再向模型暴露任意 `slots` |
| 2 Classic/VLA/Hybrid 同一 Runner | 已满足 | native handler 使用同一 Runner；默认 legacy 路径已删除 |
| 3 顶层/嵌套同一入口 | 已满足 | native `context.run()` 已复用 Runner；builtin/部署已切到 native local |
| 4 本地无需 NATS | 已满足 | native local 装配使用 LocalSkillClient |
| 5 分布式同协议 | 不在当前范围 | 当前优先单进程 native 运行；跨进程 worker 需独立立项，不保留半成品 bridge |
| 6 terminal event 恢复 Agent | 已满足 | 持久 step、submit 失败回填、terminal resume 和 startup reconcile 已接通；跨进程 durable receipt 不属本轮范围 |
| 7 VLA/RoboCasa gRPC 行为 | 未验证 | 接口保留，但缺固定 checkpoint/seed 基线；RoboCasa native action contract 尚未闭合 |
| 8 独立机制测试 | 部分满足 | timeout/cancel/resource/safety 有单测；freshness、restart 和 emergency-stop lock 验收不完整 |
| 9 独立根 verifier | 已满足 | Skill success 未直接完成根任务，Agent 仍调用独立 verifier |
| 10 RoboCasa 不回退 | 未验证 | 没有状态快照要求的固定评测记录 |
| 11 可复现 JSONL 轨迹 | 部分满足 | 已有 events/result；完整 observation/model artifact 尚未统一归档 |
| 12 单一协议/正式路径 | 已满足 | 生产执行路径只保留 native local |

因此本轮简化重构可以按 **100%** 结项。完整 RoboCasa/VLA 固定 benchmark、真机长跑和跨进程
durable transport 需要作为后续独立验证项处理。

## 13. 非目标

本轮重构不建设：

- 通用 DAG 或 workflow engine；
- 多级 semantic/option/primitive 类型体系；
- 自动 capability negotiation；
- Skill 依赖解析和版本求解；
- 通用 memory manager；
- 可自修改的 Harness；
- 多机器人 fleet scheduler；
- 完整 termination plugin 框架。

以下内容明确不是非目标，必须在重构中保留：

- 长程任务持久状态和中断恢复；
- 快慢双系统的异步 command/event 边界；
- 单进程与分布式部署使用相同领域协议；
- Agent、Skill、ModelService 和 Robot Runtime 分层解耦。

这些能力只有在实验或产品需求形成至少两个重复用例后，才进入新的设计提案。

## 14. 推荐提交顺序

```text
1. characterization tests
2. minimal Skill kernel
3. direct Agent tools
4. migrate simple builtins
5. migrate classic manipulation
6. migrate vla_act and hybrid skills
7. introduce LocalSkillClient
8. retain gRPC data planes without adding a second Skill transport
9. remove compatibility adapters and old Skill OS
10. update architecture and operations documentation
```

这一顺序先改善模型接口，再替换执行内核，最后收敛通信实现。每次提交只改变一个主要变量，
便于定位 classic、VLA、Agent tool calling、异步恢复或分布式部署行为的回归。
