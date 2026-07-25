# 最小 Tool/Skill 收缩重构方案

> 状态：T0–T8 已实施
>
> 日期：2026-07-25
>
> 前置文档：`minimal-pi-shaped-durable-agent.zh-CN.md`
>
> 参考实现：Pi Agent Core、RPent Toolkit，以及 Hey Robot 当前代码

## 1. 决策

Hey Robot 的 Tool 和 Skill 需要重构，但本次重构不是删除 Skill，也不是把物理动作当成
普通的进程内函数执行。

目标边界是：

> Tool 是 Agent 看到的唯一接口；Skill 是 Tool 背后的持久化物理执行实现，不是第二套
> Agent Tool framework。

Agent Core 只认识统一的 Tool schema 和 tool call。执行器根据工具的执行类别选择：

```text
inline tool
    → 当前进程执行
    → ToolOutcome
    → 继续 model loop

physical tool
    → persist-before-submit
    → Skill Runtime
    → yield
    → terminal event
    → ToolOutcome
    → 恢复 model loop
```

重构必须是净收缩。除修复现有控制语义所必需的内容外，不同时加入 planner、task graph、
completion verifier、memory、remote worker framework 或新的 provider abstraction。

### 1.1 2026-07-25 实施状态

收缩已经完成：

- T0：锁定 durable submit、terminal replay、restart recovery 和 control plane 测试；
- T1：assistant text 恢复自然停止，删除 model-visible `complete_task` 和 `control_task`；
- T2：`SkillCallProposal` 收缩为 `PhysicalToolCall(name, arguments)`；
- T3：删除 `SkillTool`、`SkillToolCatalog` 和重复 catalog 投影；
- T4：将 `SkillResult → ToolOutcome` 归一化集中到唯一转换边界；
- T5：删除 `LocalSkillClient` 包装层，由 `SkillWorker` 直接实现 `SkillClient` port；
- T6：删除 `SkillContext.run()`、`Skill.dependencies` 和 dependency resolver；候选 composite
  Skill 改为普通函数组合，不再拥有第二个规划/调度层；
- T7：Robot Runtime admission 改为只消费最小 `RobotActionSpec(parameters, resources,
  motion)`；删除 `SkillContractCatalog`、native Skill contract 投影和无运行时用途的
  `SkillContract`；readiness、急停、资源冲突和确定性安全检查保持不变；
- T8：删除 amendment schema、旧 physical payload 兼容、dormant `look_around` 健康投影和
  过渡 adapter，更新测试与度量；
- observation：`inspect_scene` 统一走 Robot Runtime semantic perception path；
- 默认部署：Core 只暴露 `inspect_scene`，Mobile 只暴露 `inspect_scene`、`move_base`、
  `turn_base`；RoboCasa 专用评测继续使用 `inspect_scene + manipulate`。

候选 Skill 实现源码仍然保留，但它们不能通过 dependency framework 隐式扩张执行图；若
后续消融决定启用某个宏能力，应将它注册成一个边界清楚的 Tool/Skill。

验证结果：`poe style`、`poe lint` 和完整测试通过；完整测试为 711 passed，覆盖率
85.65%。生产代码与配置相对上次提交净减少约 714 行（包含新增的中立
`tool_schema.py`）。

## 2. 为什么需要重构

### 2.1 同一个物理能力被描述了多次

当前路径是：

```text
Skill
  ↓
SkillTool
  ↓
ToolSpec
  ↓
SkillCallProposal
  ↓
SkillCommand
  ↓
SkillContract
```

native `Skill` 已经拥有：

- `name`、`description`、`parameters`；
- `handler`；
- `resources`、`timeout_sec`；
- `supported_robots`、`required_actions`、`required_models`；
- `dependencies`。

但 cognition 又通过 `SkillTool` 复制其 schema，并增加：

- `intent_kind`；
- 从参数反推的 `objective`；
- 与 `name` 重复的 `skill_name`；
- 专用的 payload 序列化和反序列化。

这些字段没有形成新的稳定边界，反而让 observation 类型和 active task 状态进入 Agent
control flow。实际 runtime 中的重复观察就是这种耦合的结果之一。

### 2.2 Registry/Catalog 重复

当前存在：

```text
SkillRegistry
    ↓ select
SkillToolCatalog
    ↓ projection
ToolRegistry
```

`SkillToolCatalog` 只包装一个 `tuple[Skill, ...]`，没有独立执行、验证或持久化语义。
ToolRegistry 可以直接消费选中的 Skill，因此这层应删除。

### 2.3 Skill contract 重复

当前 Robot Runtime 还通过以下路径再得到一份 Skill 描述：

```text
Skill
  ↓ skill_contract_from_native()
SkillContract
  ↓
SkillContractCatalog
  ↓
SkillAdmissionGate
```

`SkillContract` 预留了大量当前 native Skill 不提供、runtime 也没有真正使用的字段，例如
`success_criteria`、`recovery_hints`、`goal_effects` 和 `cannot_satisfy`。

正确的职责应是：

- Skill Runtime 验证 semantic Skill 及其运行约束；
- Robot Runtime 验证实际提交到驱动的 primitive/action、机器人 readiness 和安全状态；
- Robot Runtime 不拥有另一份 Agent Skill catalog。

### 2.4 Result contract 重复

当前终态经过：

```text
SkillResult
  ↓ SkillEvent
  ↓ TaskCoordinator
ToolOutcome
```

`SkillResult` 和 `ToolOutcome` 都表达状态、摘要、数据和失败信息。转换层不仅增加代码，
还可能让 `success`、`status`、`retryable` 和 `failure_mode` 出现不一致。

### 2.5 本地执行 shell 过多

当前本地路径是：

```text
SkillClient Protocol
    ↓
LocalSkillClient
    ↓
SkillWorker
    ↓
SkillRunner
```

目前只有 local transport。`LocalSkillClient` 和 `SkillWorker` 都管理生命周期、事件和后台
任务，可以在不破坏 `SkillClient` port 的情况下折叠。

## 3. Pi 与 RPent 的取舍

### 3.1 从 Pi Agent Core 借鉴

Pi 的 `AgentTool` 同时拥有 model-facing schema 和 validated execution port。Agent loop 的
停止规则是：

```text
有 tool call → validate/execute → tool result → 下一轮
无 tool call → assistant text 自然结束
```

Hey Robot 应采用这个统一 Tool surface 和自然停止规则。

不能直接照搬的是 inline `execute()`：真实物理 Skill 必须先持久化、异步执行，并且可以
跨进程恢复。

### 3.2 从 RPent 借鉴

RPent 的 Toolkit 将每个 `schema` 与 `handler` 显式成对注册，Planner 只依赖：

```text
get_tools_spec()
execute_tool(name, input)
```

值得借鉴的是一个 canonical registry；不应照搬的是：

- 单进程 episode 的 inline 物理执行；
- 将 benchmark 操作策略大量写进通用 tool description；
- 用 `finish` 结束开放式对话。

RPent 的 `finish` 适用于封闭 benchmark episode。Hey Robot 的普通交互由无 tool call 的
assistant text 自然结束，不保留同类 `complete_task` 协议。

## 4. 目标数据模型

### 4.1 Canonical ToolSpec

整个 Agent tool surface 只保留一种 schema：

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
```

schema 不包含：

- system prompt；
- Agent workflow；
- completion policy；
- retry policy；
- benchmark-specific 操作教程。

### 4.2 Canonical ToolCall

参数验证后只生成一个最小调用：

```python
@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    execution: Literal["inline", "physical"]
```

不再保存：

- `intent_kind`；
- 从参数推测的 `objective`；
- `skill_name` alias；
- completion 状态。

用户原始目标只存在于 Conversation。`tool_call_id` 属于本次模型调用/持久化 step，不是
ToolCall 的领域字段。

### 4.3 InlineTool

```python
@dataclass(frozen=True)
class InlineTool:
    spec: ToolSpec
    handler: InlineToolHandler
```

handler 返回统一 `ToolOutcome`。它不创建 durable physical run。

### 4.4 Skill

```python
@dataclass(frozen=True)
class Skill:
    spec: ToolSpec
    handler: SkillHandler
    timeout_sec: float = 60.0
    resources: tuple[str, ...] = ()
    supported_robots: tuple[str, ...] = ()
    required_actions: tuple[str, ...] = ()
    required_models: tuple[str, ...] = ()
```

Skill 的 schema 是其唯一 Agent-facing schema。ToolRegistry 直接读取 `skill.spec`，不创建
`SkillTool`。

### 4.5 Result 与 Event

Agent 和 Skill terminal result 使用同一个 `ToolOutcome`：

```python
@dataclass(frozen=True)
class ToolOutcome:
    status: Literal["completed", "failed", "waiting", "accepted"]
    user_summary: str | None
    data: dict[str, Any]
    operation_id: str | None
    retryable: bool
```

SkillEvent 仍保留执行阶段和 sequence：

```text
accepted | running | progress | completed | failed | cancelled
```

只有 terminal event 携带 `ToolOutcome`。图像、artifact、frame ID 和 evidence reference
作为结构化 data/attachments 保存，不再通过第二套成功状态表达。

## 5. 目标模块边界

```text
cognition/tools/
├── models.py       # ToolSpec, ToolCall, InlineTool
├── registry.py     # 唯一 Agent tool surface + 参数验证
└── executor.py     # inline / durable physical 分流

skills/
├── models.py       # Skill, SkillCommand, SkillEvent
├── registry.py     # executable Skill lookup
├── runtime.py      # submit/cancel/status/events
├── runner.py       # timeout/resource/handler
├── context.py      # RobotClient + ModelRouter + progress/cancel
└── builtins/

robot_runtime/
├── clients.py
├── action_specs.py # driver-owned primitive capabilities
└── admission.py    # readiness + primitive safety gate
```

### 5.1 ToolRegistry

唯一职责：

- 注册 InlineTool 和 Agent-visible Skill；
- 检查重复名称；
- 发布 function definitions；
- 统一验证 arguments；
- 返回最小 ToolCall 及对应 binding。

它不负责：

- 执行物理动作；
- 创建 task/run；
- completion 判断；
- observation-specific policy。

### 5.2 ToolExecutor

只进行一次分流：

```text
inline   → handler → continue
physical → durable submit → wait
control/environment terminal → finish
```

普通 assistant text 不经过 ToolExecutor。

### 5.3 SkillRuntime

保留：

- persist-before-submit；
- stable run ID；
- background execution；
- status/events；
- cancellation；
- restart reconciliation；
- terminal event sequence。

它不理解用户的完整目标，也不决定 Agent 下一步调用哪个 Skill。

### 5.4 SkillRunner

baseline 保留：

- 防御性参数验证；
- timeout；
- cancellation；
- 每机器人执行互斥；
- progress/terminal event；
- 调用 handler。

baseline 暂不需要：

- nested Skill invocation；
- dependency graph；
- composite Skill 内部的第二套规划；
- 自动 recovery policy。

Long-horizon 默认由 Agent 根据每个原子 Skill 的真实结果逐步组成。

## 6. 删除、保留与延后

### 6.0 实现保留、最小启用

本轮重构不删除暂未启用的 Skill handler 实现。系统明确区分三种集合：

```text
available Skills = 代码库中保留、可供后续消融的候选实现
enabled Skills   = 当前部署实际加载和验证的实现
visible Tools    = 当前 Agent 可以调用的最小工具集合
```

候选实现可以继续保存在 `skills/builtins/`，但不得因为“代码仍然存在”而自动进入 Agent
surface、system prompt、durable control flow 或 Robot Runtime contract。

默认 Agent-visible baseline 按机器人能力组合：

```text
Core
└── inspect_scene

Mobile
├── inspect_scene
├── move_base
└── turn_base

Mobile + VLA Manipulation
├── inspect_scene
├── move_base
├── turn_base
└── manipulate
```

其中 `manipulate` 只在部署具备对应 model/action capability 时启用，并保持 bounded option；
baseline 默认 `max_steps=1`。普通对话、完成、pause/cancel/emergency stop 和 progress 不占用
Agent Tool surface。

以下候选实现本轮保留源码，但默认不进入最小 surface：

- `look_around`、`detect_marker`；
- `base_velocity_step`、`navigate_to`、`approach_object`；
- `pick`、`place`；
- `set_arm_pose`、`move_arm_joints`、`set_gripper`；
- `reset_posture`；
- dock-specific Skills。

后续启用其中任何一项都必须通过配置独立完成，并提供相对最小 baseline 的消融结果；不能
为它修改 Agent loop、ContextBuilder、durable 状态机或 provider adapter。

### 6.1 删除或折叠

- `SkillTool`；
- `SkillToolCatalog`；
- `SkillCallProposal.intent_kind`；
- `SkillCallProposal.objective`；
- `skill_name` alias；
- `skill_call_payload()` / `skill_call_from_payload()`；
- `CompleteTaskTool` / `CompleteTaskProposal`；
- model-visible `control_task`；
- 未使用的 `SkillCancel`；
- `SkillResult` 到 `ToolOutcome` 的重复状态模型；
- `LocalSkillClient` 与 `SkillWorker` 的重复生命周期 shell；
- bus projection health 对 local execution core 的侵入；
- 宽而未使用的 SkillContract 字段。

### 6.2 必须保留

- `SkillClient` 这一小型异步 port；
- `SkillRegistry`；
- `SkillRunner`；
- `SkillContext`；
- `SkillCommand` 和 `SkillEvent`；
- `RunStore`；
- persist-before-submit；
- terminal event 幂等；
- startup reconciliation；
- timeout、cancel、stop、emergency stop；
- RobotClient；
- 跨 wakeup 的 tool count/deadline；
- 一个机器人上的物理执行互斥。

### 6.3 延后到消融验证

- nested/composite Skill；
- 多资源并发调度；
- remote Skill worker/transport；
- Skill dependency resolver；
- 自动 recovery；
- schema-driven success verifier；
- richer evidence graph；
- dynamic tool discovery；
- MCP Tool bridge。

## 7. 分阶段重构

每一阶段必须独立通过 style、lint、type check 和完整测试；每阶段结束后统计生产代码净
变化。如果某阶段没有减少概念、分支或代码，应重新审查其必要性。

### T0：Characterization 与基线锁定

目标：在改类型前锁定当前不可丢失的物理可靠性。

新增/确认测试：

- Tool schema 名称和参数保持稳定；
- physical call 在 submit 前已有 durable step/run receipt；
- 相同 run ID 重复提交不重复执行；
- accepted/running/progress 不被当作 terminal；
- terminal sequence 重放不重复恢复 Agent；
- restart 后不会重放旧物理动作；
- cancel 和 emergency stop 能绕过模型；
- inline tool 不创建 physical task/run。

本阶段不修改结构。

### T1：先完成自然停止重构

依赖 `minimal-pi-shaped-durable-agent.zh-CN.md` Phase 1：

- 无 tool call 时 assistant text 自然结束；
- 删除 `CompleteTaskTool` 和 `CompleteTaskProposal`；
- active durable record 不覆盖自然停止；
- 删除 forced continuation 和 observation-specific completion prompt；
- pause/cancel/emergency stop 保留为 typed control plane。

先移除 completion protocol，避免后续 Tool 模型继续携带错误语义。

验收：

```text
inspect_scene → semantic ToolOutcome → assistant text → end
```

### T2：收缩 ToolCall，不改变执行路径

目标：删除 proposal 中不属于工具调用的字段。

修改：

- `SkillCallProposal` 替换为最小 physical ToolCall；
- 删除 `intent_kind`、`objective`、`skill_name`；
- task step 只持久化 `name + arguments`；
- Conversation 继续作为用户目标唯一事实源；
- 将 JSON schema 参数验证移到中立的 tool schema 模块，SkillRunner 只做防御性复验。

暂时可以保留兼容读取旧 task-store payload，但新写入只使用新格式。兼容代码应标明删除
期限，不形成永久双格式框架。

### T3：删除 SkillTool 与 SkillToolCatalog

目标：形成一个 canonical model-facing schema。

修改：

- `Skill` 持有 `ToolSpec`；
- ToolRegistry 直接注册选中的 Skill；
- 删除 `skill_tools.py`；
- 删除 `SkillToolCatalog`；
- composition root 将同一 SkillRegistry 的 selected view 交给 ToolRegistry；
- startup validation 继续基于同一批 Skill。

验收：修改一个 Skill description/schema 后，Agent definition、SkillRunner validation 和
startup validation 都读取同一个来源。

### T4：统一 terminal result

目标：消除 `SkillResult → ToolOutcome` 的重复状态翻译。

修改：

- Skill handler 直接返回 ToolOutcome，或返回不含第二套状态的最小 PhysicalResult；
- terminal SkillEvent 携带 canonical outcome；
- TaskCoordinator 不再推断 `success/status` 组合；
- observation semantic summary 直接作为 `user_summary`；
- `retryable` 只表示技术可重试，不触发自动重试。

如果 artifact/image 类型仍需要强类型，可作为 outcome attachments 保留，不重新引入
第二套 completion 状态。

### T5：折叠本地 Skill runtime

目标：将只有一个实现的本地执行路径从四层缩为三层。

修改：

```text
SkillClient Protocol
    ↓
LocalSkillRuntime
    ↓
SkillRunner
```

- 合并 `LocalSkillClient` 和 `SkillWorker`；
- 保留 RunStore、后台 task、subscriber 和 cancellation；
- bus event projection 改为外部可选 subscriber；
- projection 失败不得影响 durable run truth；
- 不增加 remote transport 实现。

### T6：收缩 Skill 内部组合

目标：让 baseline 的 long-horizon 只有一个规划者。

修改：

- 从默认工具面移除依赖 nested Skill 的 composite skills，但保留其候选实现源码；
- 删除 `SkillContext.run()`；
- 删除 `Skill.dependencies` 和 `resolve_dependencies()`；
- ResourceManager 改为最小的 per-robot execution lock；
- 保留由 Agent 显式组合 atomic Skill 的路径。

如果固定评测证明某个 composite macro 显著改善成功率、延迟或安全性，再作为独立消融项
恢复，而不是恢复一个通用 dependency framework。

### T7：收缩 Robot Runtime contract

目标：Robot Runtime 只验证实际执行的 primitive/action。

修改：

- 盘点 SkillAdmissionGate 真正使用的字段；
- primitive schema、resources、motion/safety 属性由 RobotActionSpec/driver capabilities 拥有；
- Skill Runtime 在提交 primitive 前完成 semantic Skill 验证；
- 删除 `skill_contract_from_native()` 投影；
- 删除或缩小 `SkillContractCatalog`；
- 不在 Robot Runtime 复制 Agent description、dependencies、model requirements。

该阶段最后执行，因为它跨越 Skill Runtime 与驱动安全边界。必须先有 action admission 和
readiness characterization tests，不能为了减少行数而削弱 emergency stop 或确定性安全
检查。

### T8：清理与度量

- 删除所有过渡 adapter 和旧 payload 写入；
- 更新 architecture/import boundary tests；
- 更新配置示例和文档；
- 运行 style、lint、type check 和完整测试；
- 运行一次真实/模拟交互 smoke test；
- 统计 Tool/Skill 生产代码净减少量；
- 记录默认 Agent-visible atomic Skill 列表。

## 8. 每阶段停止条件

出现以下情况时停止当前阶段，不继续扩大修改：

- 需要改变 Robot driver 的安全语义；
- 无法证明 restart 后不会重复物理动作；
- terminal event 丢失 sequence/idempotency；
- cancel/emergency stop 需要等待模型轮次；
- 为兼容未来 remote deployment 新增了当前没有使用的 abstraction；
- 为删除少量代码引入更复杂的通用框架；
- 测试只能通过降低真实约束或删除可靠性断言完成。

## 9. 验收矩阵

| 场景 | 期望 |
|---|---|
| 普通问答 | 无 Tool、无 durable run、文本自然结束 |
| 视觉问答 | 一个 canonical observation Tool，semantic result 后文本结束 |
| Inline Tool | 当前 loop 执行，不创建 physical receipt |
| Physical Tool | persist-before-submit，当前 wakeup 返回 waiting |
| 多步任务 | terminal 后恢复一次，由 Agent 决定下一原子 Skill |
| 重复 terminal event | 不重复恢复、不重复写 terminal outcome |
| 进程重启 | reconcile 已提交 run，不重放动作 |
| 用户 steer | 进入 Conversation，在 safe point 被下一决策看到 |
| Cancel | 不经过模型，取消 active run |
| Emergency stop | 不经过模型，直接进入 Robot control plane |
| Retryable failure | 告知模型可重试，但 harness 不自动重试 |
| Schema 修改 | Agent、Runner、startup validation 使用同一来源 |

## 10. 最终最小流程

Agent loop：

```python
while True:
    decision = await model(messages, registry.definitions)

    if not decision.tool_call:
        close_durable_turn_if_any()
        return decision.text

    call, binding = registry.prepare(decision.tool_call)

    if binding.inline:
        outcome = await binding.execute(call.arguments)
        messages += tool_result_messages(call, outcome)
        continue

    receipt = await durable_skills.submit(call)
    return Waiting(receipt.run_id)
```

Skill terminal：

```python
event = persist_terminal_event_idempotently()
outcome = event.outcome
resume_agent_once(outcome)
```

这个 baseline 只有一个工具选择者、一个 Agent tool surface、一个物理执行事实源和一个
Robot primitive safety owner：

```text
Agent       决定调用哪个 Tool
ToolRegistry 定义 Agent 能看到什么
SkillRuntime 保证物理调用可靠执行
RobotRuntime 保证 primitive/action 安全执行
```

这是适合 Hey Robot 当前阶段的简单而通用的 Tool/Skill 架构。后续能力只有在固定评测和
消融证明必要时，才逐项增加。
