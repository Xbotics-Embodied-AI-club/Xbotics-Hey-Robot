# nanobot：它算 Autonomous Agent System 吗？——基于当前代码的 Long-Horizon Task 分析

> 分析对象：`D:\agent_robot\nanobot`，提交 `a7b8a9ed46304063c7ca0eb08cd1e34a4cc32c16`（2026-07-13 读取）。
>
> 证据优先级：Python 实现和测试优先于 README/文档。本文所说的“任务”均指一个 chat session（会话），而不是跨会话的全局任务编排。

## 先给结论

**广义上，nanobot 是一个 Autonomous Agent System；更精确地说，它是一个以 LLM tool-use loop 为核心、拥有会话状态、外部工具和有限持续执行能力的单智能体系统。**

它不是传统工作流引擎，也不是强规划型的多智能体平台：没有一个独立的、确定性的 planner 把长期目标分解为 DAG，再由 scheduler 可靠地恢复每个子任务。它的“规划—行动—观察—再规划”主要由同一个 LLM 在循环中临场完成。因此，它的自主性是**受限的、以会话为范围的、由模型驱动的自主性**。

nanobot 对 Long-Horizon Task（长时程任务）的支持与 Agent Loop **直接相关，且是对 loop 的终止语义和跨 turn 状态的扩展**。它由三层协作而成：

```text
长期目标的持久化（session metadata）
             │  每轮注入目标到 Runtime Context
             v
单次 AgentRunner 工具循环 ──达到 iteration 预算──> 隐式 continuation turn
       │  模型说“完成”但目标仍 active                    │（最多 12 次）
       └────────── 注入“继续/或 complete_goal” ────────────┘
             │
             v
complete_goal 把目标置 completed，loop 才能自然结束
```

因此不要把 `long_task` 误解为“后台任务队列”或“可靠的项目管理器”。它首先是一个**持久化目标标记（marker）**；真正推进工作的是普通工具调用循环，真正决定完成的是模型调用 `complete_goal`。

## 1. 判断一个系统是否是 Autonomous Agent System

一个实用的最小判据是：系统能否在接收目标后，使用环境反馈反复选择动作，而不是只生成一次文本。形式化地说，它近似在执行：

\[
o_t \rightarrow \pi_{LLM}(H_t, o_t, G) \rightarrow a_t \rightarrow E(a_t) \rightarrow o_{t+1}
\]

其中 `G` 是目标、`H_t` 是历史/记忆、`a_t` 是工具调用、`E` 是文件系统、shell、Web、MCP 等外部环境。只要 `a_t` 的结果会回到模型并影响下一步，便构成 agentic feedback loop。

nanobot 满足该判据：

| Agent 要素 | nanobot 的实现证据 | 含义 |
|---|---|---|
| 目标/输入 | `InboundMessage` 进入 `MessageBus`，再由 `AgentLoop` 处理 | 用户消息、cron、trigger 都可成为一次 turn 的起点 |
| 决策器 | `nanobot/agent/runner.py` 的 `AgentRunner` 调 provider | LLM 决定直接回答还是调用何种工具 |
| 动作 | `ToolRegistry` 注册文件、shell、web、MCP、cron、spawn 等工具 | 模型输出被转换成可执行的副作用 |
| 环境反馈 | tool result 作为 `role: tool` message 追加，进入下一次模型调用 | 行动不是盲发，下一步能基于结果调整 |
| 记忆/状态 | `SessionManager` 的 session JSONL/metadata、`MEMORY.md`、context consolidation | 状态可跨模型调用、部分跨进程保留 |
| 自主终止 | 没有 tool call 时 Runner 原本结束；长期目标 active 时被改写 | 终止不是单纯由一次自然语言回答决定 |

但它也缺少很多“强自主系统”通常具备的部件：没有硬编码的任务 DAG/依赖图、没有独立 verifier、没有基于 world state 的显式 replanning 状态机、没有 durable job queue 来保证长期任务在重启后自动继续。故最恰当的表述不是“全自主通用系统”，而是：

> 一个可配置、可扩展、会使用工具并可在单个会话中持续推进目标的 LLM Agent Runtime。

## 2. 整体执行路径：AgentLoop 管生命周期，AgentRunner 管思考—行动循环

两层的职责分离是理解 Long Horizon 的钥匙。

```text
Channel / WebUI / CLI / API
          │ InboundMessage
          v
MessageBus ──> AgentLoop (`agent/loop.py`)
                    │ session lock、状态恢复、上下文构建、保存、响应
                    v
              AgentRunner (`agent/runner.py`)
                    │ LLM request <-> tool calls/results
                    ├── ToolRegistry（filesystem / shell / web / MCP / spawn ...）
                    └── ContextGovernor
                    │
                    v
              SessionManager（history + metadata + checkpoint）
```

### 2.1 `AgentLoop`：一次产品级 turn 的外壳

`TurnState` 在 `nanobot/agent/loop.py` 中定义了清晰的状态机：

```text
RESTORE -> COMPACT -> COMMAND -> BUILD -> RUN -> SAVE -> RESPOND -> DONE
```

含义分别是恢复中断状态、自动压缩、处理命令、构建上下文、运行模型工具循环、保存会话、发送回复和结束。它还负责：同一 session 串行锁、不同 session 并发、pending 消息队列、流式输出、hook、会话早落盘和 checkpoint 恢复。

所以 `AgentLoop` 不是“for 循环调 LLM”的同义词，而是**一次可恢复、可观测的会话事务协调器**。Long-Horizon continuation 最终也是由它把一个新的内部消息塞回同 session 的 pending queue，再以新 turn 执行。

### 2.2 `AgentRunner`：真正的 ReAct 风格循环

`AgentRunner._run_core()`（`nanobot/agent/runner.py`）执行：

```python
for iteration in range(spec.max_iterations):
    messages_for_model = context_governor.prepare_for_model(...)
    response = await provider.chat_with_retry(...)
    if response.should_execute_tools:
        messages.append(assistant_tool_call)
        results = await execute_tools(...)
        messages.extend(tool_results)
        continue
    # 无工具调用：通常视为最终回答
```

这是典型的 Observe/Think/Act 反馈回路：工具结果被写成 `role="tool"`，下一轮模型自然能“看见”观察结果。默认 `AgentDefaults.max_tool_iterations = 200`（`config/schema.py`）；注意它限制的是**模型—工具迭代次数**，不是工具调用总数，某一轮可以并发执行多个工具。

`ContextGovernor` 在每个 provider 请求前对**给模型的副本**做协议修复、工具结果截断/归档和历史裁剪；持久化原始对话不被它就地修改。这对长任务很重要：模型上下文可以被压缩，但审计历史仍保留。

## 3. Long-Horizon 的第 1 层：`long_task` 把目标变成持久状态

实现位于 `nanobot/agent/tools/long_task.py`，状态辅助函数在 `nanobot/session/goal_state.py`。

当模型调用：

```json
{"goal": "在 docs/ 下完成……并验证链接", "ui_summary": "完善文档"}
```

`LongTaskTool.execute()` 会将下面的对象写到当前 session 的 `metadata["goal_state"]` 并立即 `sessions.save()`：

```json
{
  "status": "active",
  "objective": "……",
  "ui_summary": "……",
  "started_at": "ISO-8601 时间"
}
```

它有几个值得注意的边界。

- 一条 session 同时只能有一个 active goal；已有 active goal 时，注册新目标会报错。
- `long_task` 本身不执行工作、不拆解子任务、不创建后台 worker。工具返回的文字也明确要求“using ordinary tools”。
- 活跃目标在每次 `ContextBuilder.build_messages()` 时通过 `goal_state_runtime_lines()` 追加进 Runtime Context；目标文本最长进入上下文 4000 字符。这样即使历史被压缩，模型仍能读到核心目标。
- `ui_summary` 主要是 UI 标签；它不应承载验收条件。真正可恢复的约束要写在 `goal` 本身。
- WebUI 会收到 `GoalStateChanged` 事件，显示当前目标；这是展示同步，不是调度机制。

配套的内置 `skills/long-goal/SKILL.md` 要求模型尽快登记目标，并把目标写成幂等、自包含、有范围、可验收的 end-state。这个 prompt/skill 设计在工程上很合理：continuation 或压缩后模型可能只剩目标文本，若目标写成“先做第一步再做第二步”，恢复时很容易重复副作用；写成“确保最终状态满足 X，并先检查”则更安全。

## 4. Long-Horizon 的第 2 层：防止模型“口头完成”导致 loop 提前结束

普通 agent loop 中，如果模型不再返回 tool call，只返回一段文本，Runner 就结束。对于长期任务，这很脆弱：模型可能在只做了一部分时写“我完成了”。

nanobot 的改法很直接。在 `AgentLoop._run_agent_loop()` 创建 `AgentRunSpec` 时传入：

```python
goal_active_predicate=lambda: sustained_goal_active(session.metadata)
goal_continue_message=_goal_continue
```

然后 `AgentRunner._try_drain_injections()` 在拿到“最终回答”后按此顺序工作：

1. 先读取真实的 pending 用户消息/子 agent 结果；
2. 若没有真实注入，且 `goal_active_predicate()` 仍为真，则生成一条合成 user message；
3. 内容大意是：“你还有 active sustained goal；继续用工具推进，或在真正完成时调用 `complete_goal`”；
4. 将刚才的 assistant 文本和该合成消息追加到 `messages`，`continue` 到下一次 LLM 调用。

也就是说：**`active` 目标改变了“没有工具调用”这个停止条件。** 只有模型调用 `complete_goal`，将 `status` 改为 `completed`，下一次检查才允许正常结束。`CompleteGoalTool.execute()` 会保存完成时间与模型给出的 `recap`。

这是 Agent Loop 和 Long Task 最直接的耦合点：目标状态并非只用于 UI，而是实际参与了 Runner 的控制流判断。

## 5. Long-Horizon 的第 3 层：单个 iteration budget 耗尽后，内部切片续跑

即使模型一直在正确地调工具，单次 `AgentRunner` 也受 `max_tool_iterations` 限制。普通任务达到上限后，会生成“达到最大迭代次数”的最终回复。活跃长期目标则走 `nanobot/session/turn_continuation.py` 的不同分支。

执行链路如下：

1. `AgentRunner` 用完当前 slice 的 `max_iterations`，返回 `stop_reason="max_iterations"`。
2. 在有 active goal、存在 pending queue 的条件下，`should_finalize_on_max_iterations()` 返回 `False`，避免额外花一次无工具 finalization 调用。
3. `AgentLoop._state_run()` 调用 `maybe_continue_turn(ctx)`。
4. 该函数构造一个 `sender_id="system:continuation"` 的内部 `InboundMessage`，其中包含目标和“从已保存上下文继续、不要向用户提到边界、完成时调用 complete_goal”的提示。
5. 它把消息放入**同一 session 的内存 pending queue**，设置 `suppress_response=True`，并移除这个 slice 因预算耗尽生成的终端 assistant 文本后再保存历史。
6. 当前 turn 不向用户发回复；`AgentLoop` 随后从 queue 取到内部消息，按完整状态机开始下一 slice。

实现限制非常具体：`_MAX_GOAL_CONTINUATION_ROUNDS = 12`。初始 slice 之后最多安排 12 个 continuation slice；在默认每 slice 200 次迭代的情况下，理论上限约为 **13 × 200 = 2600 次模型迭代**（实际取决于配置、错误、取消和是否提前完成）。这说明它并非无限运行，而是一个有保险丝的持续执行策略。

这里的“内部 turn”与用户重新发消息不同：它带 `_internal_continuation` 元数据，`should_persist_user_message()` 会避免把它伪装成用户输入写进历史；`_save_skip_for_turn()` 也保证保存边界正确。对用户而言，多个 slice 尽量呈现为一次连续执行。

## 6. 任务恢复、上下文压缩与子 agent：它们能帮什么，不能帮什么

### 6.1 中断恢复：恢复对话协议，不是重放动作

Runner 在工具调用前后通过 `checkpoint_callback` 记录三类 checkpoint：

```text
awaiting_tools  -> 已有 assistant tool-call，尚无结果
tools_completed -> tool results 已齐全
final_response  -> 生成最终文本前后的边界
```

`AgentLoop._restore_runtime_checkpoint()` 会把 checkpoint 物化回 session history；对于未完成的 tool call，它追加：`Error: Task interrupted before this tool finished.`。**它不会自动重新执行该工具。** 这是一个重要的幂等性/安全选择：系统能让模型看到中断事实并自行重新规划，却不盲目重放有副作用的 shell、写文件或外部 API 动作。

此外，用户消息会先落盘并标为 `pending_user_turn`；进程在回答前崩溃时，恢复逻辑会补一条中断错误，而不是悄悄丢失用户输入。

### 6.2 压缩与记忆：解决“可放进上下文”，不等于理解任务进度

`AutoCompact` 和 `Consolidator` 会在 token 压力/空闲条件下对会话进行归纳；`ContextGovernor` 则按每次请求预算裁剪模型副本。它们解决的是上下文窗口有限的问题。活跃 goal 另存于 session metadata，因而不会依赖某一段历史是否还在。

不过它们没有维护结构化的“子任务 A 已完成、B 失败、C 等待外部事件”状态。模型从工具结果、文件系统和目标文本中推断进度；这也是它在非常长、复杂、多依赖任务上容易漂移的根源。

### 6.3 子 agent：是并行工具，不是长期任务调度器

nanobot 有 `spawn`/`SubagentManager`。子 agent 的结果会通过 pending queue 注入父 Runner，且当父 Runner 暂无消息但本 dispatch 创建的子 agent 仍在运行时，`_drain_pending()` 最多可等待 300 秒以接收结果。这有助于“研究/实现分工”。

但子 agent 并没有取代 goal continuation：父 session 的 `goal_state` 才是长期目标的唯一主状态；没有跨 agent 的任务图、依赖解析或 exactly-once 调度。

## 7. 一个实际时序例子

假设用户要求“审查仓库、修复测试、写报告”。理想执行如下：

```text
用户消息
  -> AgentLoop BUILD：载入历史、memory、skills、active goal
  -> Runner：模型读 long-goal skill，调用 long_task(G)
  -> goal_state=active 持久化
  -> Runner：read/rg/edit/pytest ...；每个结果返回模型
  -> 模型暂时回答“修复好了”
  -> Runner 发现 G 仍 active，注入“继续或 complete_goal”
  -> 模型验证并调用 complete_goal(recap)
  -> goal_state=completed
  -> 下一次无工具文本回答，Runner 正常退出，AgentLoop 保存并响应
```

若中间用尽 200 次迭代：

```text
Runner max_iterations
  -> 不把预算耗尽回复给用户
  -> pending queue 入队 internal continuation
  -> 保存当前 slice 历史
  -> 新 AgentLoop turn 从已保存上下文继续
  -> 最多重复 12 次；仍未完成则最终暴露预算边界
```

## 8. 这套机制的优点与关键局限

### 优点

1. **极少的额外抽象。** 目标只是 session metadata，复用现有 Runner、工具和 session 设施，不引入另一套难以同步的 task engine。
2. **目标跨压缩可见。** history 可能被摘要，但 `Goal (active)` 每 turn 都重新注入。
3. **避免明显的假完成。** active goal 让“模型说了一句 done”不再自动结束。
4. **预算切片对用户透明。** 长执行可跨多个 Runner slice，且通过最大轮数防止无限烧 token。
5. **恢复策略保守。** checkpoint 把未完成工具明确变成错误结果，不自动重放副作用。

### 局限与工程风险

1. **完成判断仍信任模型。** `complete_goal` 没有强制 verifier；模型可过早调用它，或永远不调用它。测试通过、产物存在、URL 可访问等验收，主要是 prompt/skill 约束，而非硬策略。
2. **不是 durable background job。** continuation message 放在进程内 `asyncio.Queue`。目标 metadata 会落盘，但“下一 slice 已入队”本身没有持久化为可重启的 job；进程在队列消费前退出时，不会仅凭 active goal 自动唤醒继续执行。
3. **不主动触发。** active goal 不会自己定时醒来。它只能在当前 dispatch 的内部 continuation 中继续，或由用户、cron、trigger 等外部输入再次触发 turn。
4. **只保证会话级串行。** 这适合个人助手；若任务需要跨 session 资源锁、分布式执行、外部回调编排，必须另建 scheduler/lease/状态机。
5. **理论预算仍可能很大。** 默认上界的 2600 次迭代可能带来成本、延迟和副作用累计；系统需要产品层的总 token、wall-clock、工具权限和人工取消保护。
6. **“幂等目标”不是幂等执行。** 文档要求 check-then-act，但是否真的幂等取决于具体工具/API；`exec`、网络写操作、外部系统调用仍需工具层的确认、去重键和权限控制。
7. **恢复不是断点续算。** 恢复的是对话证据与错误标记，模型仍要重新判断环境现状。对非幂等操作这是正确取舍，但会牺牲自动完成率。

## 9. 最终回答：它与 Agent Loop 到底是什么关系？

可以把两者关系浓缩为一句话：

> Agent Loop 提供“模型能反复调用工具”的短程闭环；`long_task` 给闭环加上跨上下文的目标不变量，`complete_goal` 定义显式退出，continuation 则把一个受预算限制的闭环串成有限多个内部 turn。

因此，Long-Horizon Task **不是** Agent Loop 的替代物，也不是仅靠 memory 就能实现；它正是建立在 Agent Loop 之上，对以下三个问题分别给出机制：

| 长任务问题 | nanobot 的回答 | 类型 |
|---|---|---|
| 如何记住“还在做什么”？ | `session.metadata[goal_state]` + 每 turn Runtime Context | 持久目标 |
| 如何不因一次文本回答就停？ | `goal_active_predicate` 注入继续消息 | 修改 Runner 终止条件 |
| 单轮工具预算用完怎么办？ | `maybe_continue_turn()` 入队内部 turn，最多 12 轮 | Loop 外层切片续跑 |

如果把 nanobot 用于真实的长周期生产任务，建议把它看作“智能决策与工具操作层”，外面再配一个确定性的任务控制层：持久化 job queue、阶段状态/验收器、幂等键、预算和超时、人工审批以及外部事件订阅。这样能保留 LLM 的灵活性，同时把**何时继续、何时完成、哪些副作用可以重试**从模型自觉提升为可审计的工程规则。

## 代码索引（复核入口）

- `nanobot/agent/loop.py`：`TurnState`；`_run_agent_loop()` 注入 goal predicate/continuation；`_state_run()`、`_state_save()`；checkpoint 恢复。
- `nanobot/agent/runner.py`：`AgentRunSpec`；`_run_core()` 的 iteration/tool loop；`_try_drain_injections()` 的 sustained-goal continuation。
- `nanobot/agent/tools/long_task.py`：`LongTaskTool` 与 `CompleteGoalTool` 的唯一 active goal 读写逻辑。
- `nanobot/session/goal_state.py`：goal 是否 active、Runtime Context 渲染、long-goal 取消 LLM wall timeout。
- `nanobot/session/turn_continuation.py`：达到 iteration 上限后入队内部 continuation、12 轮上限、保存边界。
- `nanobot/agent/context.py`：将 active goal 加入构造给模型的 messages。
- `nanobot/skills/long-goal/SKILL.md`：给模型的目标书写、幂等性与完成约束。
- `tests/agent/test_runner_goal_continue.py`、`tests/agent/tools/test_long_task.py`、`tests/session/test_goal_state.py`：以上关键语义的可执行测试。
