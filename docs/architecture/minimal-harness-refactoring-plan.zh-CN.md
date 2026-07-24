# Hey Robot Tool/Skill 边界与 VLA 能力增量重构方案

## 1. 范围

本文是
[《Hey Robot 最小化 Embodied Agent Harness 设计与演进路线》](minimal-embodied-agent-harness.zh-CN.md)
在当前阶段的实施计划。本轮只解决两个相邻问题：

1. 重构现有 Tool/Skill 边界，使 Agent 既能调用普通 Harness Tool，也能通过同一套 Tool
   协议调用机器人 Skill；
2. 借鉴 RPent 的 VLA 实践，把 VLA 包装成 Agent 可以稳定调用、持续观察、有界执行并获得
   结构化反馈的机器人能力。

这两项工作共享 Tool schema、参数校验和执行分发边界，但不要求重写 Agent 推理循环。

### 1.1 实施状态（2026-07-24）

本文 Phase 0—7 已完成实现和回归验证：

- Tool contract、通用 Registry、typed prepared call 和最小 Dispatcher 已落地；
- 普通 Harness Tool 与 Robot Skill Tool 已能通过同一 Agent Tool Registry 注册和分发；
- Robot Tool schema 继续从 Skill 单向投影，classic/VLA/hybrid 切换不改变 schema；
- VLA 参数已收紧，父 Skill 到 `manipulate` 使用显式参数映射；
- `VLAOptionRunner`、`TerminationPolicy` 和 option/subgoal/task 完成语义已落地；
- execution trace 使用现有 FileRunStore/MediaStore 外置，Gateway 可按 trial 投影 option；
- RoboCasa365 `CloseFridge / seed=1000 / B1` 实际仿真通过官方成功谓词。

最终评估记录见
[《Tool/Skill 与 VLA 重构 RoboCasa365 评估记录》](../evaluation/robocasa365/tool-skill-vla-refactoring-evaluation-20260724.zh-CN.md)。

本轮明确不做：

- 不重写 `AutonomousAgentService` 和 cognition 主循环，只允许把现有硬编码分支委托给小型
  `ToolDispatcher`；
- 不引入另一套 Agent loop；
- 不移动 `cognition`、`skills`、`foundation`、`robot_runtime` 包；
- 不删除 `TaskCoordinator`、`SkillClient`、`SkillWorker`、`RunStore`；
- 不改变 Web、CLI、语音、飞书和 Gateway；
- 不为了单进程实验拆除 NATS、gRPC 或现有分布式部署；
- 不在第一批实现文件、网络、Shell 等高风险普通 Tool，只建立可扩展的普通 Tool 调用通道；
- 不在本轮统一 Skill/Capability 的所有术语。

这里的“简化”是控制新增设计和实验变量，不是拆除已经存在且经过测试的模块边界。

## 2. 从 RPent 借鉴什么

RPent 的 VLA 调用链是：

```text
Planner
  -> LiberoToolkit
  -> pi0_pick / pi0_doubled
  -> LiberoPrimitives
  -> VLAClient + EnvClient
  -> states / images / action video
```

`pi0_pick` 在一个 Tool handler 内执行多次 VLA action chunk，并根据末端高度、gripper opening
和环境 termination 停止。每次 Tool 完成后，Toolkit 保存状态、图片、耗时和动作片段，
再把新观察返回 Planner。

值得借鉴：

1. Agent 调用有界 VLA 能力，而不是直接处理 action tensor；
2. 能力输入包含语言 subgoal 和明确执行预算；
3. VLA 在短时域内连续执行 action chunk；
4. 每个 chunk 后检查 termination；
5. 能力结束后返回新观察、终止原因和诊断；
6. 保存图片、状态、动作、耗时和视频，支持复现与消融；
7. 环境和 VLA 可独立部署，由上层能力统一编排。

不借鉴：

- 不把 `pi0_pick` 等模型实现名作为默认 Agent Tool；
- 不默认向高层 Agent 暴露 `move_to`、`rotate_wrist` 等 primitive；
- 不把 RPent Planner、Toolkit 或文件 Tool 移植到 cognition；
- 不让 VLA Tool handler 绕过 Skill Runtime 和 Robot Runtime；
- 不用 RPent 的任意文件访问或无类型化 Tool 边界替换现有系统。

## 3. Hey Robot 当前基础

Hey Robot 已经有正确的主链：

```text
Agent Tool: manipulate / pick / place
  -> SkillCallProposal
  -> TaskCoordinator
  -> SkillClient / SkillWorker
  -> SkillRunner
  -> VLA Skill
  -> Foundation Model Router / gRPC ModelService
  -> action_chunk
  -> RobotClient / Robot Runtime
```

当前 `manipulate` 已实现 `task_prompt`、`max_steps`、逐步重新观察、Model Router 调用、
action chunk 执行、progress、frame id 和多种终止原因。因此不需要重造 RPent Toolkit，
只需要把现有 VLA Skill 收敛为稳定、可测试的 bounded option。

目标链保持现有系统边界：

```text
Agent semantic Tool
  -> existing proposal/coordinator/worker
  -> VLAOptionRunner
       observe -> infer -> validate -> robot execute
       -> reobserve -> terminate or repeat
  -> SkillResult + evidence/artifacts
  -> existing Agent continuation
```

## 4. Tool/Skill 边界重构

### 4.1 概念边界

`Tool` 是 Agent 可见的统一调用接口，是一个超集：

```text
Agent Tool
  ├── Harness Tool
  │     例如任务控制、artifact 读取、未来受限文件操作
  └── Robot Capability Tool
        由 Skill 自动投影，例如 pick、place、manipulate
```

因此：

- 不是每个 Tool 都是 Skill；
- 每个允许暴露给 Agent 的 Skill 都可以投影为 Tool；
- Skill 表示机器人能力及其运行约束，不能用普通 Tool handler 绕过 Skill Runtime；
- VLA、WAM、classic、hybrid 是 Skill 的实现方式，不是默认暴露给 Agent 的 Tool 名称。

RPent 的 `Toolkit(spec, handler)` 适合轻量实验，但 Hey Robot 不能把 Tool 与机器人执行能力
完全合并，否则会丢失现有的资源锁、取消、超时、事件、Worker 和远程执行边界。这里借鉴其
“schema 与可执行入口成对注册”的简洁性，不照搬其单层抽象。

### 4.2 统一 Tool 协议

在现有 `cognition/tools` 内建立最小协议，不移动包：

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, object]


class AgentTool(Protocol):
    @property
    def spec(self) -> ToolSpec: ...

    def prepare(self, arguments: dict[str, object]) -> "PreparedToolCall": ...
```

`PreparedToolCall` 使用显式类型，而不是无类型化字典：

```text
HarnessToolCall          -> 普通非物理 Tool handler
SkillCallProposal        -> 现有 TaskCoordinator / Skill Runtime
CompleteTaskProposal     -> 现有任务完成路径
ControlTaskProposal      -> 现有取消、暂停等控制路径
```

现有 `SkillTool` 保留职责：从 Skill 的 `name`、`description`、`parameters` 自动生成 ToolSpec，
校验 LLM 参数后生成 `SkillCallProposal`。普通 Harness Tool 则拥有自己的 schema 和 handler。

### 4.3 Registry 与 Dispatcher

将当前偏机器人语义的 `ToolRegistry` 收敛为通用注册表：

```text
LLM tool call
  -> ToolRegistry.prepare(name, arguments)
  -> typed PreparedToolCall
  -> ToolDispatcher.dispatch(call)
       ├── HarnessToolCall       -> constrained handler
       ├── SkillCallProposal     -> TaskCoordinator
       ├── CompleteTaskProposal  -> existing completion service
       └── ControlTaskProposal   -> existing control service
```

Registry 只负责发现、schema 发布、参数校验和生成 typed call；Dispatcher 只负责路由，不拥有
Agent loop、Skill 执行状态机或机器人通信。`AutonomousAgentService` 仅把已有 `isinstance`
分支委托给 Dispatcher，不改变 observe/reason/act/continue 的控制流。

为降低迁移风险，先在原模块保留兼容导出；测试和调用点迁移完成后，再决定是否把
`cognition/tools/robot.py` 更名为 `registry.py`。

### 4.4 Skill schema 是机器人 Tool 的唯一事实来源

机器人 Capability Tool 不再维护第二份参数定义：

```text
Skill parameters
  -> SkillTool projection
  -> Agent Tool JSON schema
  -> LLM arguments
  -> SkillCallProposal
  -> SkillRunner 再校验
```

双重校验是刻意保留的：Tool 边界尽早拒绝无效模型输出，Skill Runtime 边界防御来自 API、
回放或远程客户端的调用。两处校验必须基于同一份 Skill schema。

同一个语义 Skill 的 classic/VLA/hybrid/WAM 实现必须共享公开 schema。当前通过配置在 factory
阶段选择 handler 的机制可以继续使用，不需要第一批就重写整个 Skill Registry。只有当第二种
实现确实需要独立 requirements、resources 或 lifecycle 时，再把 `SkillSpec` 与
`SkillImplementation` 正式拆开。

### 4.5 参数如何到达 Skill

Agent 可以决定 Skill schema 明确公开的参数；Tool 层不吞掉这些参数：

```text
LLM arguments
  -> SkillTool.prepare(arguments)
  -> validate against Skill.parameters
  -> SkillCallProposal(arguments)
  -> TaskCoordinator / SkillCommand
  -> SkillRunner.validate(arguments)
  -> selected Skill handler
```

部署内部参数、模型 checkpoint、控制增益和安全限制不应为了“可调用”而暴露给 LLM。父 Skill
调用子 Skill 时也要显式做参数映射，不能把整个 arguments 字典透传。

## 5. Agent-facing 机器人能力

### 5.1 `manipulate`

建议将公开参数收紧为：

```python
MANIPULATE_PARAMETERS = {
    "type": "object",
    "properties": {
        "task_prompt": {"type": "string", "minLength": 1},
        "max_steps": {
            "type": "integer",
            "minimum": 1,
            "maximum": 300,
            "default": 1,
        },
    },
    "required": ["task_prompt"],
    "additionalProperties": False,
}
```

`model_timeout_sec`、fresh observation timeout 和 termination strategy 默认来自部署或实验
配置，不让 Agent 任意设置。如果需要消融，通过 evaluation config 控制。

这里保留 `maximum=300` 是为了兼容 RoboCasa365 B0 的单次完整根目标对照；常规 B1/B2 仍使用
短 option。真机部署可以通过部署 profile 使用更小上限，但不能让 LLM 绕过 schema 上限。

### 5.2 `pick` 和 `place`

保持语义接口：

```text
pick(object, grasp_hint?, task_prompt?, max_attempts?)
place(object?, target, placement_hint?, task_prompt?, max_attempts?)
```

实现继续由现有配置选择：

```text
classic | vla | hybrid | future: wam
```

Agent 不看到 Pi0.5、checkpoint 或 implementation id。当前 `vla_pick()` 和 `vla_place()`
无条件覆盖显式 `task_prompt`，应改为优先使用 Agent 参数：

```python
task_prompt = arguments.get("task_prompt") or f"grasp {target}"
```

同时不能继续用 `{**arguments}` 把 `object`、`target` 等父 Skill 参数全部透传给
`manipulate`。收紧 `additionalProperties` 后，semantic wrapper 必须显式映射子 Skill 参数：

```python
return await ctx.run(
    "manipulate",
    {
        "task_prompt": task_prompt,
        "max_steps": resolved_max_steps,
    },
)
```

这样可以保持 schema 严格，又不会把 parent-only 参数泄漏给 VLA backend。

### 5.3 参数所有权

| 参数 | 示例 | 所有者 |
|---|---|---|
| 语义目标 | object、target、task_prompt | Agent |
| 有界预算 | max_steps、max_attempts | Agent 在 schema 上限内决定 |
| termination 策略 | model/environment/success/budget | 部署或 evaluation |
| 模型实现 | VLA、WAM、classic、hybrid | Skill 配置 |
| timeout/freshness | 推理和观察超时 | 部署配置 |
| action scale/关节限制 | 控制参数 | Robot Runtime |
| observation/run id/deadline | 运行上下文 | Harness 注入 |

## 6. VLAOptionRunner

只从 `skills/builtins/vla.py` 抽取 VLA bounded loop，不改变 cognition 或 Skill runtime。

建议新增：

```text
src/hey_robot/skills/vla/
  __init__.py
  option.py
  termination.py
```

核心数据：

```python
@dataclass(frozen=True)
class VLAOptionRequest:
    task_prompt: str
    max_steps: int
    termination: str


@dataclass(frozen=True)
class VLAOptionResult:
    option_completed: bool
    subgoal_succeeded: bool | None
    termination_reason: str
    before_frame_id: int | None
    after_frame_id: int | None
    model_outputs: tuple[dict, ...]
    executed_actions: tuple[dict, ...]
    evidence_ids: tuple[str, ...]


class VLAOptionRunner:
    async def run(
        self,
        context: SkillContext,
        request: VLAOptionRequest,
    ) -> VLAOptionResult: ...
```

`skills/builtins/vla.py` 继续定义 `MANIPULATE` Skill；handler 只解析参数、构造 request、调用
runner 并转换为现有 `SkillResult`。这不是新增系统层，只是把 VLA 算法循环从注册文件中
抽出，便于独立测试和复用。

## 7. Termination

Hi-VLA 和 RPent 都说明控制权切换是关键变量。本轮只增加一个小型策略接口：

```python
class TerminationPolicy(Protocol):
    def evaluate(self, state: VLAOptionState) -> TerminationDecision: ...
```

第一阶段等价表达现有逻辑：

- `EnvironmentDoneTermination`；
- `ModelDoneTermination`；
- `NoActionTermination`；
- `BudgetTermination`；
- 上述策略的 composite。

后续实验再增加：

- `PickHeuristicTermination`：gripper closure、末端高度和 fresh image；
- `SuccessDetectorTermination`：独立 verifier；
- 连续多帧一致性，降低 false positive。

必须区分：

```text
option_completed    当前有界执行已结束
subgoal_succeeded   当前语言子目标有证据成功
task_completed      整个用户任务完成
```

达到 `max_steps` 只能证明 option 结束，不能自动证明 subgoal 或整个任务成功。Agent context
必须看到 `termination_reason` 和 `subgoal_succeeded`。

为兼容现有 `SkillResult`，`SkillResult.success` 表示 option 是否在没有模型、传输、控制或
安全错误的情况下完成执行；它不等同于 `subgoal_succeeded`。例如达到 budget 可以是
`SkillResult.success=True`、`option_completed=True`、`subgoal_succeeded=None`，随后由 Agent
重新观察。整个任务仍只能通过环境 done 或带 evidence 的 `complete_task` 完成。

## 8. Observation、结果与 artifact

每个 bounded step 固定执行：

```text
fresh observation
  -> ModelService infer
  -> validate action chunk
  -> Robot Runtime execute
  -> fresh observation
  -> termination evaluation
```

`SkillResult.data` 至少返回：

```json
{
  "termination_reason": "model_done",
  "option_completed": true,
  "subgoal_succeeded": true,
  "requires_reobservation": true,
  "before_frame_id": 10,
  "after_frame_id": 14,
  "steps_used": 2
}
```

图片、视频和较大模型输出通过现有 MediaStore/ArtifactRef 保存，不塞入 LLM context。

借鉴 RPent 的状态和视频记录，但不新建第二套全局 Trace 系统；在现有 `SkillEvent`、
`FileRunStore`、MediaStore 和 episode artifact 中补齐：

- 每个 bounded step 的 observation frame id；
- VLA request 的公开参数；
- 模型输出与 primitive 摘要；
- termination decision 和 elapsed time；
- 可选 step video artifact。

## 9. Foundation 与 Robot Runtime

保持 gRPC ModelService 和 Model Router，不把 VLA 模型加载进 Agent 进程。VLA request 收敛为
`task_prompt`、observation、policy session、step index 和 budget；response 收敛为
`action_chunk`、`task_done` 和 diagnostics。

VLA backend 只能返回声明过的 Robot Runtime action。VLA Skill 提交前继续校验结构和动作名，
Robot Runtime 继续负责 dimension、bounds、health 和 safety flags。

未来 WAM 是新的 `manipulate` Skill implementation 或 Foundation backend，不改变 Agent Tool
schema，也不进入 cognition。

## 10. 实施阶段

### Phase 0：锁定当前行为

先增加 characterization tests：

```text
tests/cognition/tools/test_tool_registry_characterization.py
tests/cognition/tools/test_skill_tool_argument_flow.py
tests/integration/test_vla_tool_argument_flow.py
tests/skills/vla/test_option_characterization.py
```

覆盖当前 Tool schema 顺序和内容、complete/control/Skill proposal 类型、LLM 参数到 SkillCommand
的传播、未知参数拒绝、逐步重新观察、action chunk 执行、各种终止、
cancel/timeout/stale observation，以及 classic/VLA/hybrid Tool schema 一致性。

退出条件：旧实现上新增测试和全量测试通过。

### Phase 1：统一 Tool contract 与 Registry

只在 `cognition/tools` 内完成等价重构：

- 引入 `ToolSpec`、`AgentTool` 和 typed `PreparedToolCall`；
- 让 `SkillTool`、`CompleteTaskTool`、`ControlTaskTool` 实现同一协议；
- 让 Registry 接收普通 Harness Tool，不再把返回值封闭为三种现有 proposal；
- 保留原 import 的兼容导出；
- 不改变发给模型的 Tool schema，不改变 SkillCommand。

退出条件：Phase 0 的 Tool 测试不改断言即可通过；Registry 可用 fake Harness Tool 证明扩展性。

### Phase 2：接入最小 ToolDispatcher

新增只做 typed call 路由的 Dispatcher，并让 `AutonomousAgentService` 委托现有执行分支：

- `SkillCallProposal` 仍进入 `TaskCoordinator`；
- complete/control 仍进入现有服务；
- `HarnessToolCall` 进入受约束 handler；
- 不改变 AgentRunner 的推理、重试和 continuation 语义；
- 暂不提供 Shell、网络或任意文件访问 Tool。

退出条件：现有端到端行为不变，并有一个无机器人副作用的 fake/in-memory Harness Tool
证明参数可以从 LLM 调用到 handler，再把结构化结果返回 Agent。

### Phase 3：稳定 Skill 投影并收紧公开参数

修改 `skills/builtins/vla.py` 和 `skills/builtins/tabletop.py`：

- 确认机器人 Tool schema 只来自 Skill，不允许 Registry 覆盖；
- 明确 Agent 参数和上限；
- 内移运行时参数；
- `additionalProperties` 改为 `false`；
- 修复 `task_prompt` 被覆盖；
- semantic wrapper 显式映射 child Skill 参数，不再透传整个 parent arguments；
- 保证实现切换不改变 Tool schema。

退出条件：未知/内部参数在 SkillCommand 前被拒绝，合法链路不变。

### Phase 4：等价抽取 VLAOptionRunner

新增 `skills/vla/option.py`，只迁移 bounded loop：

- 不改 Agent、TaskCoordinator、SkillClient、Worker、Robot Runtime；
- 不改 SkillEvent phase；
- 不改 gRPC；
- 不同时加入新 termination 算法。

退出条件：Phase 0 测试不改断言即可通过。

### Phase 5：显式 TerminationPolicy

新增 `skills/vla/termination.py`，明确 option/subgoal/task 三种完成语义并结构化保存原因。

退出条件：每种策略有纯单测和 fake model/runtime 集成测试。

### Phase 6：补齐现有 artifact

使用现有 MediaStore、FileRunStore 和 SkillEvent 保存 bounded step 证据，不新增数据库或第二套
Trace 系统。

退出条件：给定 run id，可重建 observation/inference/action/termination 时间线。

### Phase 7：实现对照实验

在同一语义 Tool 下比较 generic VLA、`pick/place` 的 classic/VLA/hybrid，后续再增加 WAM。
继续使用现有 `skills.implementations` 配置，不重构整个 Skill Registry。

退出条件：schema 不变，并输出 success、steps、latency、termination、recovery 指标。

## 11. 文件级决策

| 文件 | 本轮动作 |
|---|---|
| `cognition/runtime/agent_runner.py` | 仅适配通用 prepared call 类型，不改推理循环 |
| `cognition/autonomous_agent_service.py` | 最小改动：把现有执行分支委托给 Dispatcher |
| `cognition/tools/models.py` | 新增统一 ToolSpec、协议和 typed call |
| `cognition/tools/registry.py` | 新增或由 `robot.py` 渐进更名；统一注册和 prepare |
| `cognition/tools/dispatcher.py` | 新增小型执行路由，不拥有 Agent loop |
| `cognition/tools/robot.py` | 迁移期保留兼容导出，确认调用点后再删除 |
| `cognition/tools/skill_tools.py` | 保持 Skill 自动投影，适配统一协议 |
| `cognition/runtime/task_coordinator.py` | 不改 |
| `skills/runner.py` | 复用，不改执行架构 |
| `skills/worker.py`、`skills/transport/local.py` | 不改 |
| `skills/models.py`、`skills/registry.py` | 第一批不拆；只在真实多实现需求出现后演进 |
| `skills/builtins/vla.py` | 收紧为 adapter，抽取 bounded loop |
| `skills/builtins/tabletop.py` | 修复参数传递，保持 semantic wrapper |
| `skills/vla/option.py` | 新增 VLA bounded option |
| `skills/vla/termination.py` | 新增 termination 策略 |
| `foundation/*` | 保持边界，仅收敛 VLA payload |
| `robot_runtime/*` | 不重构，继续执行安全校验 |
| `app/runtime_components.py` | 不改部署组合 |
| NATS/gRPC/deploy | 保留，不做简化迁移 |

## 12. 测试与第一批工作

重点新增：

```text
test_registry_accepts_harness_and_skill_tools
test_skill_tool_schema_is_projected_from_skill
test_skill_tool_arguments_reach_skill_command
test_harness_tool_arguments_reach_handler
test_dispatcher_preserves_existing_skill_route
test_llm_arguments_reach_vla_option
test_pick_preserves_explicit_task_prompt
test_vla_option_reobserves_after_action
test_vla_option_rejects_unknown_agent_parameters
test_vla_output_cannot_bypass_robot_runtime
test_budget_end_is_not_automatically_task_success
test_termination_reason_is_persisted
test_pick_implementation_switch_keeps_tool_schema
```

每阶段运行 `poe style`、`poe lint` 和 `poe test`。

实际实施遵循上述顺序：先锁定当前参数/结果链路，再统一 Tool contract 和 Registry；随后接入
Dispatcher、稳定 Skill 参数投影，最后抽取 `VLAOptionRunner` 并完成 RoboCasa365 门禁。真实
仿真额外发现并修复了 fresh-frame 门槛和 Gateway option 统计投影两个单元测试未覆盖的问题。

这能修正 Hey Robot 当前 Tool Registry 只认识 Skill/complete/control proposal 的局限，也保留
“Tool 是 Agent 接口、Skill 是机器人能力”的必要分层；同时以最小改动吸收 RPent 在 VLA
能力封装上的价值，并保持 Hey Robot 已有 cognition 主循环、Skill Runtime、Robot Runtime
和分布式部署结构稳定。
