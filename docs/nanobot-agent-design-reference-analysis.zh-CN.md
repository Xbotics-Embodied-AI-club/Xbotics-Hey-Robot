# nanobot Agent 设计深度分析及其对 Hey Robot 的参考价值

> 分析对象：`D:\agent_robot\nanobot`  
> nanobot 代码版本：`a7b8a9ed46304063c7ca0eb08cd1e34a4cc32c16`（2026-07-07）  
> 对照项目：当前工作区 Hey Robot 代码  
> 分析原则：以实际代码和测试为准，项目文档仅作为设计意图的辅助证据。  
> 文档角色：参考分析和决策依据，不作为实施规范。统一重构请只执行 `docs/architecture/autonomous-agent-refactor-plan.md`。
>
> **当前统一 V1 决策（2026-07-12）**：自主 RobotAgent 只看到 `request_observation` 和 `request_skill`，普通 UserTurn 没有物理工具；两个工具只产生 ActionProposal，真实 SkillIntent 由 Supervisor 唯一提交；V1 没有 Agent memory/RAG、自动 reobserve/retry、approval、timer、mid-turn correction、ProgressDetector、模型最终报告或崩溃续跑；物理状态未知会形成跨 Goal 的执行锁。本文后续出现的 memory consolidation、自动重新感知、阶段目录或恢复建议均只代表研究候选，不能直接作为当前实施步骤。

## 1. 结论先行

nanobot 对 Hey Robot **有较高参考价值，但不适合整体移植**。

对于以架构研究为目标的 Hey Robot，最值得借鉴的不是 nanobot 的自动修复和长任务续跑，而是三项更基础的工程结构：

1. **AgentLoop 与 AgentRunner 分层**：把产品 turn 生命周期和通用模型工具循环分开。
2. **进行中 turn 的细粒度 checkpoint 与事件**：准确记录模型决定、已完成工具和未完成工具，但不自动重发物理动作。
3. **可复用的 Hooks、SDK 和观测接口**：完整记录运行时间线、token、延迟、工具结果和失败位置。

ContextGovernor、Runner 内中途注入和 nanobot sustained goal 仍值得研究，但当前只作为设计案例，不直接迁移。Hey Robot 的持续目标采用 GoalStore + Supervisor + 真实事件唤醒，不等同于 nanobot 的隐藏自动续跑。尤其不应引入静默上下文修复、内部 continuation 或 provider fallback，因为这些机制会掩盖 Hey Robot 正在探索的真实失败。

Hey Robot 已经吸收了 nanobot 的部分思想。代码中有明确证据：

- `RobotAgentLoop` 注释说明其 turn lifecycle 借鉴了 nanobot 的分层方式；
- OpenAI Responses 转换和解析模块由 nanobot provider 层移植并适配；
- Hey Robot 也已经拥有显式 turn 状态机、工具协议、并发工具批处理、checkpoint、pending turn 和 provider 兼容层。

但是，两者的核心安全假设不同：

> nanobot 默认面对数字世界，工具失败通常可以重试；Hey Robot 面对物理世界，动作可能不可逆，不能把“自动重试、上下文修复、内部续跑”原样套到机器人动作上。

因此，推荐的总体策略是：

> 借鉴 nanobot 的模块边界、checkpoint 和可观测性；保留 Hey Robot 的 Skill OS、物理证据、机器人租约和安全边界；默认显式失败，不用兼容和自动修复维持“看起来还能运行”。

### 1.1 研究型系统的优先原则

本文后续建议均服从以下原则：

```text
Fail explicitly.
Recover deliberately.
Never hide failure.
```

- 可以接受任务失败，但失败必须结构化、可观察、可复现；
- 安全 gate 可以阻止危险动作，但 fallback 不能把失败伪装成正常结果；
- 非法工具协议、缺失参数、上下文超限和模型空响应默认终止当前实验；
- 恢复只能来自少量明确策略，不能依靠文本猜测和层层兼容；
- 旧配置通过一次性迁移脚本处理，主运行时只接受当前 schema；
- 架构优化目标是减少职责和分支，不是提高“无论如何都返回一句话”的成功率。

## 2. nanobot 的准确定位

nanobot 是一个面向个人数字助理和通用工具 Agent 的 Python 框架，包含 React/TypeScript WebUI。它的核心不是复杂的多 Agent 编排框架，而是一个较小的模型工具循环，加上围绕它建立的会话、上下文、渠道、provider、安全和持续任务设施。

当前仓库大约包含：

- 197 个 Python 源文件，约 6.6 万行；
- 266 个 Python 测试文件，约 8.2 万行；
- 多种聊天渠道、provider、MCP、WebUI、OpenAI-compatible API 和 Python SDK。

从代码所有权看，可以分为五部分：

```text
Channels / API / SDK
        |
        v
MessageBus + AgentLoop
        |
        v
AgentRunner
        |
        +-- Provider
        +-- ToolRegistry
        +-- ContextGovernor
        +-- Hooks
        |
        v
Session / Memory / Goal / Automation
```

与 Hey Robot 不同，nanobot 没有独立的 Skill OS 和 Robot Runtime。对它而言，`exec`、`read_file`、`web_fetch`、`spawn` 等工具就是执行边界。

## 3. nanobot 的核心设计原则

`.agent/design.md` 明确给出三个值得重视的原则。

### 3.1 核心保持小，能力向边缘扩展

`agent/loop.py` 和 `agent/runner.py` 是关键路径。新能力优先放在：

- Channel adapter
- Tool
- Markdown Skill
- MCP server
- Hook

它避免把 Telegram、WebUI、GitHub、定时任务等产品逻辑直接塞进模型循环。

### 3.2 少结构，多智能

nanobot 不追求把每个概念都抽象成框架层。它允许 provider 和 channel 之间存在少量重复，以换取单文件可读性。

这个原则对 Hey Robot 有条件适用。通用 Agent 层可以保持简单，但机器人技能合同、资源调度和安全检查不能为了“少结构”被压回 prompt 或工具描述中。

### 3.3 显式优于隐式

配置必须进入 Pydantic schema，provider 解析路径必须可以追踪，非法输入应给出明确错误，而不是静默修正。

nanobot 的工具参数仍会做安全的 schema-driven cast，例如把字符串数字转为数值，但不会把未知工具名自动改成另一个工具名，只提供建议。

## 4. AgentLoop 与 AgentRunner 的分工

这是 nanobot 最重要的结构设计。

### 4.1 AgentLoop：产品层 turn 生命周期

`AgentLoop` 负责：

- 从 MessageBus 获取消息；
- 计算 session key；
- 控制同一 session 串行、不同 session 并发；
- 构造上下文；
- 管理 session、memory、自动压缩和 checkpoint；
- 处理 slash command；
- 接收中途消息；
- 组织 streaming、progress 和最终回复；
- 驱动 cron、trigger、subagent 等外围能力。

它使用显式状态机：

```text
RESTORE
  -> COMPACT
  -> COMMAND
  -> BUILD
  -> RUN
  -> SAVE
  -> RESPOND
  -> DONE
```

各状态含义如下：

| 状态 | 责任 |
|---|---|
| `RESTORE` | 恢复未完成 turn、pending user turn 和 session |
| `COMPACT` | 处理过期 session 和上下文压缩 |
| `COMMAND` | 优先处理 slash command 和快捷路径 |
| `BUILD` | 构造 system prompt、历史、memory、skill 和 runtime context |
| `RUN` | 调用 AgentRunner |
| `SAVE` | 保存新消息、checkpoint 和会话元数据 |
| `RESPOND` | 组装并发送 OutboundMessage |

这与 Hey Robot 的 `restore -> build -> run -> save` 相似，但 nanobot 多了独立的 `COMPACT`、`COMMAND` 和 `RESPOND` 阶段。

### 4.2 AgentRunner：与产品无关的模型工具循环

`AgentRunner` 只关心：

1. 准备模型输入；
2. 请求 provider；
3. 解析 reasoning、文本和 tool calls；
4. 执行工具；
5. 追加 tool results；
6. 判断继续还是返回；
7. 统计 usage 和 stop reason；
8. 在关键位置发 checkpoint 和 hook 事件。

它不直接拥有 Channel、SessionManager、WebUI 或具体业务状态。这使同一 Runner 可以被：

- 主 AgentLoop 使用；
- SubagentManager 使用；
- Python SDK 使用；
- Dream memory consolidation 使用。

Hey Robot 也有 `RobotAgentCore` 和 `AgentRuntime` 的分层，但 `AgentRuntime` 已经包含任务合同、证据账本、感知 grounding、机器人技能完成语义等具身逻辑。因此它不是纯通用 Runner。

## 5. 一次 nanobot turn 的真实执行过程

### 5.1 消息调度

Channel 把输入包装成 `InboundMessage`，放入进程内 `MessageBus` 的 asyncio queue。

AgentLoop 的调度策略是：

- 同一 session 使用 asyncio lock 串行；
- 不同 session 可以并行；
- `/stop` 等优先命令不等待普通 turn；
- session 已有活动 turn 时，普通追问进入 pending queue；
- cron 和 local trigger 在 session 忙碌时延后。

这与 Hey Robot 的并发模型不同：Hey Robot 同时锁 episode 和 robot，因为不同用户会话也可能争用同一台实体机器人。

### 5.2 用户消息提前落盘

nanobot 在正式执行前先持久化触发 turn 的用户消息，并在 session metadata 写入 `pending_user_turn`。

如果进程此后崩溃，下次恢复时会发现只有用户消息、没有 Assistant 回复，从而补入：

```text
Error: Task interrupted before a response was generated.
```

这避免用户输入在崩溃窗口中凭空消失。

### 5.3 上下文构造

System prompt 由下列部分组合：

- 内置 identity 和平台策略；
- workspace 中的 `AGENTS.md`、`SOUL.md`、`USER.md`；
- tool contract；
- 长期 `MEMORY.md`；
- always-on skills；
- 其他 skill 摘要；
- recent history；
- 已归档的 session summary。

当前时间、channel、chat ID、sender ID、活动目标、MCP 和 CLI App 等动态信息被放入一个标记为 metadata-only 的 runtime context，并追加在当前 user content 后面。

追加而不是前置有一个细节收益：用户内容前缀较稳定，有利于 provider prompt cache 命中。

### 5.4 模型工具循环

AgentRunner 每次迭代：

1. 生成只用于模型请求的消息副本；
2. 对该副本执行 ContextGovernor；
3. 调用模型；
4. 若模型请求工具，则保存 Assistant tool-call 消息；
5. 执行工具并保存 tool result；
6. 检查中途注入；
7. 继续下一次模型调用；
8. 无工具时生成最终回复。

空回复会有限重试，输出因 token 长度被截断时会要求模型继续。达到最大工具迭代数后，会尝试一次禁用工具的 finalization 请求。

### 5.5 保存和响应

保存时会：

- 丢弃无内容且无 tool call 的 Assistant 消息；
- 丢弃 orphan tool result；
- 截断大 tool result；
- 把内嵌 base64 图片替换为文件占位符；
- 移除 runtime metadata，避免下一轮把动态元数据当成历史指令；
- 记录 turn latency；
- 原子替换 session JSONL 文件。

## 6. ContextGovernor：适合分析，不适合当前直接迁移

Hey Robot 当前的 `MessageWindowPolicy` 主要做两件事：限制消息数量、截断 tool result；`message_protocol` 则严格拒绝 orphan 或缺失 tool result。

nanobot 的 `ContextGovernor` 更完整，它在每次模型请求前依次执行：

1. 移除无意义的 Assistant placeholder；
2. 移除没有合法工具名的畸形 tool call；
3. 丢弃 orphan tool result；
4. 为缺失的 tool result 补入“调用中断或丢失”的错误结果；
5. 将过大的 tool result 保存到 workspace 文件，只在 prompt 中放引用；
6. 对当前 turn 内的大型查询结果做 micro-compaction；
7. 根据 context window、输出预算和工具 schema 成本裁剪历史；
8. 再次修复工具调用边界。

最关键的设计是：

> ContextGovernor 只修改交给模型的副本，不原地修改持久化会话。

这同时保留了两种事实：

- 审计层知道历史中真实发生过什么；
- 模型层得到合法、紧凑、可继续执行的上下文。

### 对 Hey Robot 的价值

ContextGovernor 揭示了长会话中常见的协议污染问题，但当前 Hey Robot 不应迁移它的自动修复行为。研究阶段更需要一个严格的 `RobotContextInspector`：

- 发现 orphan tool result 时产生 `MODEL_PROTOCOL` 失败；
- 发现 missing tool result 时保存完整上下文并终止；
- 发现无效 tool call 时记录 provider、request ID 和原始响应；
- 统计 system、history、memory、robot state、tool schema 的 token 占用；
- 上下文超过预算时产生 `CONTEXT_BUDGET_EXCEEDED`，不静默裁剪；
- 验证 TaskContract、EvidenceLedger 和 RobotStatus 是否出现在预期位置。

等实验数据证明某类污染高频且修复语义明确后，再把对应修复作为显式策略加入，并分别统计“原始成功率”和“恢复后成功率”。

nanobot 的“只修改模型副本、不改审计原文”仍值得保留为未来修复机制的设计原则，但不能让修复过程没有事件和指标。

## 7. 进行中 turn checkpoint

nanobot 在三个关键阶段保存 checkpoint：

```text
awaiting_tools
tools_completed
final_response
```

checkpoint 包含：

- iteration；
- model；
- Assistant tool-call 消息；
- 已完成 tool results；
- 尚未完成的 tool calls。

当任务被 `/stop` 取消或进程中断时，恢复逻辑会：

1. 把 Assistant tool-call 决策恢复进历史；
2. 保留已经完成的 tool results；
3. 为未完成工具补入“Task interrupted before this tool finished”；
4. 清除 runtime checkpoint。

它不会自动重跑未完成工具。这一点非常适合机器人场景。

### Hey Robot 当前差距

Hey Robot 的 `RobotAgentCheckpoint` 当前主要保存：

- episode ID；
- phase；
- skill ID；
- pending turns。

它能恢复任务级状态，但不能精确还原 LLM 已发出的 tool calls、哪些结果已写入上下文，以及中断发生在模型调用前还是工具执行中。

### 当时的 checkpoint 建议（已由统一方案替代）

以下两层方案用于说明问题来源。当前 V1 已删除 TaskRunManager，并改为 Supervisor 的 autonomy.sqlite3、RobotAgent 的 DeliberationStore 与 Skill OS 的 SkillCommandStore 三个明确所有者，不按下面的旧 TaskRun 方案实施。

当时建议把 checkpoint 分成两层：

1. **Task checkpoint**：继续使用现有 `RobotAgentCheckpoint` 和 TaskRun，描述物理任务状态。
2. **Model-turn checkpoint**：新增只用于恢复模型协议的轻量记录，保存 Assistant tool call、完成结果和 pending call。

恢复规则必须比 nanobot 更严格：

- 未完成的只读查询可以标为 interrupted 后重新规划；
- 未完成的 `request_skill` 不能自动重发；
- 先查询 SkillStore、RobotStatus 和 skill lease，判断动作是否已受理或已完成；
- 对状态不确定的物理动作进入 `UNKNOWN/BLOCKED`，不把重新感知当作动作是否已执行的证明，也不自动重发；
- checkpoint 中只保存引用，不复制大图像。

## 8. Mid-turn Injection：中途追问如何进入当前推理

nanobot 为每个活动 session 建立 pending queue。AgentRunner 在以下边界读取新消息：

- 工具执行完成后；
- 模型给出最终文本后但尚未结束 stream 时；
- 工具错误后；
- LLM 错误后；
- 空回复后；
- max iterations 后。

如果读到用户追问或 subagent 结果，它会追加为新消息并让模型继续，而不是结束当前 turn 再开一个竞争 turn。

系统对注入数量和循环次数设有上限，未消费的消息会重新发布到 MessageBus，避免静默丢失。

### Hey Robot 当前行为

Hey Robot 已能识别机器人忙碌时的：

- read-only 查询；
- correction；
- follow-up；
- retry；
- interrupt / emergency stop。

但 correction 和 follow-up 主要写入 durable pending turn，在当前活动 skill 结束后作为新 turn 重放。`RobotTurnInjector` 的“mid-turn”更准确地说是“下一安全 turn 的上下文合并”，不是 nanobot 那种同一 AgentRunner 循环内注入。

### 是否值得增强

值得，但只能部分采用。

统一方案只允许在以下安全边界处理 correction/follow-up：

- SkillResult/Observation 事件结束当前等待，并准备新的 deliberation 时；
- 活动物理 skill 已终止、取消或状态明确以后；
- 新 slice 尚未发起下一项主动观测或物理技能以前；
- final response 发送以前。

不在同一个等待中的 AgentRunner 内续跑，也不允许把普通 correction 直接注入正在运行的 driver primitive 或 VLA action chunk。紧急停止继续走独立高优先级通道，不依赖 LLM 注入。

## 9. Sustained Goal 与内部 continuation

nanobot 的 `long_task` 工具把目标写入 session metadata：

```json
{
  "status": "active",
  "objective": "...",
  "ui_summary": "...",
  "started_at": "..."
}
```

模型完成目标后必须调用 `complete_goal`。

如果活动目标在一轮达到 max iterations，`turn_continuation` 会：

1. 不把预算耗尽回复发给用户；
2. 生成一个不可见的 system continuation turn；
3. 从已保存上下文继续；
4. 最多连续 12 个 continuation round；
5. 最终由 `complete_goal` 或预算上限结束。

AgentRunner 还支持 `goal_active_predicate`。如果模型试图在目标仍 active 时提前返回最终文本，Runner 会插入“继续完成目标”的消息，让模型继续调用工具。

### 对 Hey Robot 的价值

它说明长任务为什么需要跨模型迭代预算推进，但不建议引入 nanobot 的内部 continuation。内部 continuation 会改变任务成功率、失败分布和执行时间，并把 AgentRunner 重构与自治调度混在一起。Hey Robot 当前要实现的是 Supervisor 根据真实事件启动新的有限 slice。

但 nanobot 对 sustained goal 会关闭默认 LLM wall timeout，这在机器人场景中不可接受。统一 V1 的 Goal 只保留以下硬预算：

- 最大 deliberation 次数；
- 总墙钟时间；
- 最大物理技能数；
- 最低电量。

移动距离、转角、温度和 observation freshness 继续由 SkillSpec / safety gate 约束，不再复制为 Goal 级启发式预算。用户取消和 emergency stop 是类型化控制事件，不是预算。

Hey Robot 当前选择把持续目标实现为 Supervisor 管理的事件驱动 Goal，而不是复制 nanobot 的 session-only goal 或 TaskRun 内部 continuation：

```text
Goal WAITING
  -> persist Goal + Action + evidence
  -> receive matching SkillResult
  -> WakePolicy evaluates hard budgets and state
  -> create a new finite DeliberationRequest
  -> run one new Agent slice
```

普通 RobotObservation 只更新 snapshot，不唤醒；V1 没有 timer goal 或普通 correction wake。cancel/emergency 只改变控制状态，不调用 LLM。这不是隐藏 continuation：没有匹配 SkillResult 就不进行下一次物理决策。

## 10. Session 与 Memory 设计

### 10.1 Session durability

nanobot 每个 session 使用 JSONL 文件。保存方式是：

1. 写临时文件；
2. 可选 flush + fsync；
3. `os.replace` 原子替换；
4. graceful shutdown 时对缓存 session 执行 durable flush。

加载逻辑还会修复损坏或旧版 session，并限制文件最大消息数。

可借鉴的只有原子提交、flush 和清晰 cursor。统一 V1 的权威状态改用 SQLite；schema 不匹配、损坏或校验失败直接停止，不引入运行时 corruption repair，也不把所有状态统一成 nanobot Session。

### 10.2 两阶段 memory

nanobot 的 memory 分两步：

1. **Consolidator**：当 context 超预算或 session 空闲时，把旧消息总结到 `history.jsonl`。
2. **Dream**：定期读取未处理 history，用受限文件工具更新 `SOUL.md`、`USER.md` 和 `MEMORY.md`，并通过 Git diff 形成审计记录。

这个设计解决了两个问题：

- 热上下文不能无限增长；
- 长期记忆不能只是聊天记录堆积。

### 对 Hey Robot 的价值

Consolidator 对数字 Agent 很有价值，但统一重构已将 consolidation 和模型 memory mutation 全部后置；Dream 不迁移。

仅作为未来 memory 实验的参考原则：

- 如未来增加 episode consolidation，必须单独开关和评估；
- summary 必须带源 episode、turn、skill 和 evidence ID；
- 结构化的 skill experience、scene anchor、task lesson 继续作为事实源；
- 如未来允许 LLM 提出 memory mutation proposal，必须由 schema validator 和规则层提交；
- 不允许 Dream 自动修改物理 SkillSpec、安全策略、driver 参数或机器人身份 prompt；
- 用户偏好与机器人操作经验分库存储，避免“用户喜欢快速”被错误解释成突破运动限制。

### 10.3 nanobot 当前有没有 RAG

按当前代码，nanobot **没有把经典向量 RAG 作为 Agent memory 主路径**。

代码证据：

- nanobot/agent/memory.py 的 MemoryStore 是文件 I/O，核心文件是 MEMORY.md、history.jsonl、SOUL.md 和 USER.md；
- Consolidator 负责把会话历史压缩为文本记录；
- Dream 读取历史和现有文件，再提出受限文件修改；
- 主代码没有 FAISS、Chroma、Qdrant、Milvus、pgvector、embedding pipeline 或向量召回器；
- nanobot/apps/cli/service.py 中出现的 chromadb 只是 CLI Apps 品牌目录项，不是 Agent memory backend。

所以它更准确的描述是：

~~~text
session history
  -> text consolidation
  -> durable Markdown/JSONL memory
  -> prompt/file-tool access
~~~

而不是：

~~~text
documents
  -> embeddings
  -> vector index
  -> top-k retrieval
  -> augmented prompt
~~~

这类文件记忆对具身 Agent 的最大参考价值是 provenance、cursor、审计和压缩边界，不是语义检索算法。Hey Robot V1 连这套长期文件记忆也不迁移，只保留当前 Goal 的类型化 EvidenceFact。

## 11. Markdown Skills 与 Hey Robot Skill OS 不是同一概念

nanobot 的 Skill 是 Markdown + YAML frontmatter，本质是按需加载的操作说明、领域知识或工作流提示。它通常告诉模型如何组合现有工具。

Hey Robot 的 `BaseSkill/SkillSpec` 是可执行合同，包含资源、安全、超时、支持机器人和成功证据。

两者不应合并命名。建议在 Hey Robot 中引入不同概念，例如：

```text
Agent Playbook / Task Playbook
```

适合放入 Playbook 的内容：

- “先观察再导航”的策略；
- 特定房间搜索顺序；
- 失败恢复经验；
- 某类任务如何组合 semantic skills；
- 用户交互和确认规范。

不适合放入 Playbook 的内容：

- 电机范围；
- 安全阈值；
- required resources；
- 是否允许执行；
- 物理完成判定。

后者必须继续由代码和 SkillSpec 决定。

## 12. Subagent 设计（未来研究，不属于 V1）

nanobot 的 SubagentManager 有几个良好约束：

- 子 Agent 使用独立 ToolRegistry；
- 只继承 exec、web、file 等有限工具配置；
- 有并发数量限制；
- 按 session 跟踪，可被 `/stop` 一并取消；
- 结果通过 MessageBus 注入主 Agent，而不是直接修改主会话；
- tool error 时返回已完成步骤和失败摘要。

### Hey Robot 后续是否值得单独研究

V1 不引入 Subagent。未来若单独立项，它也只适合非物理决策的并行任务，例如：

- 分析多张场景图；
- 查询说明书或地图元数据；
- 比较多个恢复方案；
- 总结长 episode；
- 离线生成任务报告。

未来候选中的子 Agent 不得拥有：

- `request_skill`
- `request_observation`
- RobotAction
- Skill OS 提交接口
- emergency stop 以外的机器人控制能力

当前 V1 没有 `propose_skill` 工具，也没有任何子 Agent 注册入口。RobotAgent 只产生一个 ActionProposal，Supervisor 是唯一物理提交者。

## 13. Hook 与 SDK

nanobot 定义了完整的生命周期 hook：

- `before_run`
- `after_run`
- `on_error`
- `on_finally`
- `before_iteration`
- `on_stream`
- `on_stream_end`
- `before_execute_tools`
- `after_iteration`
- `finalize_content`

CompositeHook 对大多数观察性 hook 做错误隔离，避免一个监控插件拖垮 Agent。Python SDK 则提供：

- `run`
- `run_streamed`
- event stream
- session client
- memory client
- runtime client
- per-run model override

### 对 Hey Robot 的价值

Hey Robot 已有 ToolHook、AgentRuntimeHook、事件总线和 Web progress，但接口分散。可以增加统一的只读观测 hook，用于：

- provider latency 和 token usage；
- tool/skill timeline；
- task evidence 变化；
- recovery decision；
- robot observation freshness；
- benchmark recorder。

安全 hook 不应使用 nanobot 的“异常隔离后继续”策略。任何安全 gate 异常必须 fail closed；只有日志、指标和 UI hook 可以隔离失败。

## 14. Tool 插件和安全边界

nanobot 的 ToolLoader 通过包扫描和 entry point 自动发现工具，支持 core/subagent scope。当前内置工具包括文件、Shell、搜索、Web、MCP、定时任务、子 Agent、图片生成和持续目标。

因为工具权限很大，它建立了：

- workspace path containment；
- read/write 分离的额外允许目录；
- Shell working directory 检查；
- 可选 bubblewrap sandbox；
- SSRF 防护；
- 重复 workspace violation 和外部查询节流；
- wildcard API host 必须配置 API key；
- Channel sender allowlist / pairing。

Hey Robot 当前 Agent 工具面更窄，因此不需要照搬整套数字工具安全系统。但如果未来加入 MCP、Shell、网页或自修改能力，应先移植安全边界，再开放工具，不能反过来。

## 15. Provider 与流式交互

nanobot provider 层比 Hey Robot 更广，支持 Anthropic、OpenAI-compatible、Responses API、Azure、Bedrock、GitHub Copilot 等，并覆盖：

- provider retry 和 Retry-After；
- reasoning/thinking block；
- prompt cache marker；
- token usage 与 cached token；
- stream idle timeout；
- Responses API circuit breaker；
- 多种 provider 的 role/tool 协议差异。

对 Hey Robot 最有价值的是 provider capability matrix、usage 统计、超时分类和原始 retry metadata。统一 V1 只记录 provider 给出的 Retry-After 等诊断信息，不执行 provider retry；一次 deliberation 的 provider 失败直接结束该 Goal。

nanobot 的 streaming 设计也值得参考。它把一次含多轮工具调用的回答拆成多个 stream segment，并通过 `resuming` 告诉 UI 当前文本结束后是否还会继续执行工具。Hey Robot Web/Voice 可以用类似语义区分：

- 当前自然语言片段结束；
- Agent 仍在运行；
- 技能正在执行；
- 最终任务回复结束。

## 16. nanobot 自身的局限和架构风险

nanobot 提供了很多成熟机制，但代码也暴露出值得 Hey Robot 避免的问题。

### 16.1 “核心保持小”是目标，不完全是现状

当前 `AgentLoop` 接近 1900 行，`AgentRunner` 接近 1500 行。Loop 同时处理 session、命令、streaming、自动化 turn、MCP、subagent、checkpoint、文件状态和 Web runtime event；Runner 同时处理 provider、上下文治理、工具执行、注入、安全错误分类、流式输出和 finalization。

这些逻辑仍有清晰的局部函数和测试，但核心修改的回归面已经很大。Hey Robot 不应把 nanobot 的所有 continuation、checkpoint、hook 和 context 逻辑直接塞进现有 `AgentRuntime`，而应保持独立组件和单向依赖。

### 16.2 进程内 MessageBus 不提供持久性

`MessageBus` 本质是两个 asyncio queue。进程崩溃时未落盘消息会消失，跨进程和多节点也不可用。nanobot 用 session checkpoint 缓解 turn 丢失，但它不等于 durable event bus。

Hey Robot 已有 NATS 和本地任务状态，不应降级为这一模型。真正需要借鉴的是 pending queue 和注入语义，而不是总线实现。

### 16.3 工具能力面很宽，默认限制并非最严格

nanobot 可以给模型 Shell、文件写入、Web、MCP 和子 Agent。`restrict_to_workspace` 默认是 `False`；没有 bubblewrap 时，Shell workspace 检查也只是应用层防护，不是进程隔离。

这对个人本地助理可能是可接受的易用性选择，但不能成为机器人控制主机的默认权限模型。Hey Robot 若接入通用数字工具，应将其放在独立低权限服务或容器中，并与机器人控制网络隔离。

### 16.4 ToolRegistry 重名时会覆盖

`ToolRegistry.register()` 直接按名称赋值，后注册工具可以覆盖同名工具。MCP 和 entry-point 插件扩大了供应链与命名冲突风险。

Hey Robot 的 SkillRegistry 会拒绝 duplicate skill，这个策略更适合物理能力。未来接入 MCP 时也应拒绝重名，或要求带稳定 namespace，不能允许插件静默替换安全相关工具。

### 16.5 Context 自愈可能掩盖上游缺陷

ContextGovernor 的 repair 很实用，但修复只发生在模型副本，持久化原文可能长期保留畸形记录。如果没有指标，系统可能每一轮都在静默修同一个问题。

Hey Robot 应同时：

- 修复模型副本以保证可用性；
- 发出结构化 corruption event；
- 记录首次出现位置和 provider；
- 提供离线修复工具；
- 对物理工具协议异常触发 recovery，而不是只有 prompt repair。

### 16.6 Sustained goal 的预算对数字任务更宽松

活动目标可以跨最多 12 个 continuation round，并关闭默认 LLM wall timeout。即便没有物理风险，也可能带来不可预测的费用和运行时间。

Hey Robot 必须使用多维硬预算，不能只限制 continuation 次数。

### 16.7 Dream 和外部 Skill 存在提示供应链风险

Markdown Skill、workspace 指令、MCP 描述和 Dream prompt 都会进入模型上下文。恶意或低质量内容可能污染长期行为。nanobot 已意识到 context pollution，但基于文本的能力扩展天然难以获得与代码合同同等的可信度。

Hey Robot 的 Playbook 应区分来源、签名或安装权限，并永远不能覆盖系统安全规则和 SkillSpec。

### 16.8 Subagent 结果仍然是模型生成的非可信输入

子 Agent 结果通过 system inbound message 注入主 Agent。它能提高并行度，但结果本身不是事实证明。主 Agent必须重新验证关键结论。

在机器人场景中，子 Agent 对“看到杯子”“路径安全”“抓取成功”的判断不能直接成为动作或完成证据，必须回到受信任的 perception/skill evidence。

## 17. 两个项目的核心差异

| 维度 | nanobot | Hey Robot |
|---|---|---|
| 目标场景 | 数字个人助理 | 具身机器人 Agent |
| 消息总线 | 进程内 asyncio queue | NATS，可拆分服务 |
| Agent 核心 | 通用 tool loop | 一次模型请求、最多一个 ActionProposal 的有限 slice |
| 执行边界 | ToolRegistry | SkillGateway + Skill OS + Robot Runtime |
| 完成判定 | 最终文本或 `complete_goal` | TaskContract + EvidenceLedger + SkillResult |
| 并发单位 | session | episode + robot + skill resource |
| 中途输入 | 同一 Runner 内注入 | V1 只有 Goal cancel / emergency control，不做普通 correction |
| 长任务 | sustained goal continuation | GoalStore + Supervisor + matching SkillResult 事件唤醒 |
| checkpoint | 模型消息和工具级 | AutonomyStore / DeliberationStore / SkillCommandStore 分权持久化 |
| 上下文治理 | 修复、offload、micro-compact、裁剪 | 自包含 request + 严格校验；超限直接失败 |
| 长期记忆 | Consolidator + Dream + 文件 | V1 无 Agent memory/RAG；只有当前 Goal 的 EvidenceFact |
| 子 Agent | 内置后台 subagent | 无通用 subagent |
| 物理安全 | 不涉及 | 多层 deterministic gate |
| 资源调度 | read-only/concurrency-safe/exclusive | camera/base/arm/gripper 资源合同 |
| 自治边界 | 可跨 max-iteration 内部续跑 | Supervisor 独占 SkillIntent 提交，失败或未知即停止 |

## 18. 哪些设计应直接借鉴

### P0：优先级最高

#### 18.1 AgentLoop 与 AgentRunner 分层

理由：当前 `cognition/runtime/runner.py` 同时处理模型循环、任务合同、证据、grounding、fallback 和最终回复，真实失败容易被后续策略覆盖。应让 Runner 只返回模型与工具真实发生的结果。

#### 18.2 统一结构化失败模型

理由：Provider error、协议错误、工具错误、技能失败和任务证据不足必须能够被明确区分，不能继续依靠字符串解析和 fallback 文本。

#### 18.3 Model-turn checkpoint

理由：物理任务耗时长，模型或进程中断的代价高。应精确记录 tool-call/result 协议状态，并在恢复时把未完成动作标为 interrupted，而不是重发。

### P1：有明显收益

#### 18.4 Token-aware context inspector

只测量和报告上下文组成、预算和协议合法性。超限直接失败，不做自动裁剪和补齐。

#### 18.5 统一只读 observability hooks

为实验记录 provider 请求、tool/skill timeline、TaskEvaluation、token、延迟和失败，但 observer 不得修改控制流或吞掉异常。

#### 18.6 删除运行时兼容和 fallback

废弃配置直接报错，旧配置使用一次性脚本迁移；模型未完成就记录 unfinished，不根据最后一个工具结果自动合成答案。

### P2：未来独立研究，不进入统一 V1

#### 18.7 Agent Playbook

只有仿真实验证明稳定工作流知识确有价值后才单独研究，并且必须与可执行 SkillSpec 分离。当前不实现。

#### 18.8 安全边界上的用户修正

统一 V1 只支持 Goal cancel 和 emergency stop，不实现 durable correction event，也不把普通用户消息合并到 active Goal。若未来研究 correction，必须作为独立变量重新设计协议。

#### 18.9 Bounded continuation 和认知子 Agent

作为独立研究变量，不与当前核心重构同时引入。子 Agent 永远不获取物理执行权。

## 19. 哪些设计不应直接照搬

### 19.1 不要用进程内 MessageBus 替换 NATS

nanobot 的 queue 简单有效，但不适合 Hey Robot 已有的多服务、GPU 模型和机器人进程边界。

### 19.2 不要让所有能力都退化成通用 Tool

物理动作必须继续走 SkillGateway、Skill contract、资源调度和 Robot Runtime。不能把 `move_base` 做成一个与 `read_file` 同级的普通 Tool。

### 19.3 不要采用“目标 active 就强制继续”的单一规则

机器人还必须判断：相机是否新鲜、状态是否健康、恢复是否完成、用户是否撤销、资源是否可用。

### 19.4 不要关闭长任务 LLM 超时

nanobot 对 sustained goal 可取消 wall timeout。机器人系统必须始终有硬上限和 watchdog。

### 19.5 不要让 Dream 自动改安全和技能代码

自修改对个人数字助理有实验价值，对真实机器人则可能改变动作语义和安全边界。最多允许生成提案和测试，不应自动部署。

### 19.6 不要用纯文本 final answer 取代物理完成证据

Hey Robot 的 EvidenceLedger 和 execution feedback 是比 nanobot 更适合具身场景的设计，必须保留。

## 20. 设计推导（非实施规范）

> 本节是形成统一方案时的中间推导，组件名称、协议字段和 checkpoint 结构已经被 `docs/architecture/autonomous-agent-refactor-plan.md` 替代，不应据此创建文件或接口。

建议重构 Agent/Cognition 层，同时保持现有四层架构不变。

```text
RobotAgentService
  |
  +-- RobotTurnLifecycle
  |     restore / build / run / evaluate / save
  |
  +-- RobotContextInspector         [新增]
  |     strict validation / token accounting / diagnostics
  |
  +-- ModelTurnCheckpointStore      [新增]
  |     assistant calls / completed results / pending calls
  |
  +-- AgentRunObserver              [新增]
        timeline / metrics / failure events

Strict AgentRunner
  |
  +-- 只负责 LLM <-> Tool 严格循环
  +-- 不负责任务完成判定和用户回复 fallback

RobotDecisionPolicy
  |
  +-- TaskContract / EvidenceLedger / grounding
  +-- 输出 SATISFIED / UNSATISFIED / INCONCLUSIVE

SkillGateway
  +-- 唯一物理执行入口

Skill OS / Foundation / Robot Runtime
  +-- 不改变所有权边界
```

### 20.1 Strict AgentRunner 的建议处理顺序

```text
build request
  -> inspect context budget
  -> validate provider messages
  -> request model
  -> validate provider response
  -> validate tool name and arguments
  -> execute tool
  -> persist exact result
  -> continue or return structured failure
```

遇到 orphan result、missing result、无效工具、参数错误、空响应、provider timeout 和 max iterations 时直接失败，不自动修复。

### 20.2 ModelTurnCheckpoint 建议字段

```json
{
  "episode_id": "...",
  "turn_id": "...",
  "iteration": 2,
  "phase": "awaiting_tools",
  "assistant_message": {},
  "completed_tool_results": [],
  "pending_tool_calls": [],
  "active_skill_id": "...",
  "observation_frame_id": 123,
  "updated_at": 0.0
}
```

恢复时还要比较当前 `RobotStatus.frame_id`、active skill 和 checkpoint observation，防止用旧状态继续规划。

### 20.3 统一失败模型

建议使用一个结构化失败对象：

```python
@dataclass(frozen=True)
class AgentFailure:
    stage: FailureStage
    code: FailureCode
    component: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = field(default_factory=dict)
```

最小阶段集合：

```text
CONTEXT_BUILD
MODEL_REQUEST
MODEL_PROTOCOL
TOOL_VALIDATION
TOOL_EXECUTION
SKILL_EXECUTION
TASK_EVALUATION
PERSISTENCE
```

`retryable` 只是实验标签，默认不触发自动重试。

## 21. 历史实施建议（已被统一 Phase 0～7 替代）

> 以下顺序只保留用于解释决策演进；实际重构不得执行本节，必须执行统一重构文档的 Phase 0～7。

### 第一阶段：建立严格失败基线

1. 定义 `AgentFailure`、`AgentRunResult` 和统一 error code；
2. 为当前主链路建立 golden trace 和故障注入测试；
3. 删除 provider error、空响应和 max iterations 的伪正常回复；
4. 删除基于自然语言错误文本的 failure mode 推断；
5. 记录当前真实失败分布。

这是后续重构的实验基线，不能省略。

> 以下阶段顺序是分析形成过程，不是当前施工顺序；实际 Phase 0～4 只见统一实施文档。

### 第二阶段：拆分 Runner 与机器人策略

1. 把纯 LLM/tool loop 提取为 Strict AgentRunner；
2. 将 TaskContract、EvidenceLedger 和 grounding 移到 `RobotDecisionPolicy`；
3. 将用户回复呈现移出 Core；
4. Core 只负责依赖组装和一次 turn 调用；
5. 保证 SkillGateway 仍是唯一物理出口。

### 第三阶段：checkpoint、上下文诊断和 observer

1. 增加 model-turn checkpoint；
2. 增加 token-aware ContextInspector；
3. 增加只读 AgentRunObserver；
4. checkpoint 恢复只标记 interrupted，不重发动作；
5. 对上下文超限和协议污染直接失败并保存完整证据。

### 第四阶段：删除兼容

1. 删除 `mode`、`type`、`resources` 等被忽略的旧配置字段；
2. 删除旧 skill 名称和 legacy protocol 文本；
3. Provider 差异只允许存在于 `providers/`；
4. 旧配置使用一次性迁移脚本；
5. 不增加 `legacy_mode` 或双运行路径。

### 第五阶段：事件驱动自主目标；复杂恢复和内部 continuation 后置

持续目标按统一重构文档实现为 `GoalStore + AutonomySupervisor + WakePolicy`，不采用 nanobot continuation。只有在积累足够失败数据后，才选择高频且语义明确的失败增加恢复。例如确认大量失败来自 observation stale 后，再单独实验一次 `REOBSERVE`。恢复前后的成功率必须分开统计。复杂恢复和 Runner 内 continuation 不进入当前主路径。

## 22. 测试证据

本次在 nanobot 仓库运行了与 Agent 设计直接相关的测试：

```powershell
python -m pytest -q `
  tests/agent/test_loop_runner_integration.py `
  tests/agent/test_runner_governance.py `
  tests/agent/test_runner_injections.py `
  tests/agent/test_runner_goal_continue.py `
  tests/agent/test_stop_preserves_context.py `
  tests/agent/test_session_atomic.py `
  tests/agent/test_subagent.py `
  tests/session/test_turn_continuation.py `
  tests/session/test_goal_state.py `
  tests/security
```

结果：

```text
164 passed, 1 skipped in 13.67s
```

这些测试验证了：

- Loop/Runner 分层执行；
- ContextGovernor 的协议修复和裁剪；
- tool result micro-compaction；
- mid-turn injection 及队列清理；
- sustained goal continuation；
- checkpoint 取消恢复；
- session 原子写；
- subagent 工具隔离；
- workspace 和网络安全边界。

本次没有运行 nanobot 全部测试，因此不对整个项目当前测试状态作“全绿”结论。

## 23. 最终评价

nanobot 的 Agent 设计优势不在于提出了一个新颖的循环，而在于把实际运行中最棘手的边界处理得较完整：

- 历史会污染；
- provider 协议会不一致；
- tool result 会过大；
- 用户会在执行中改变需求；
- Agent 会在预算边界被迫停止；
- 进程会在 tool call 中间崩溃；
- 子任务结果会异步返回；
- 会话需要可恢复、可压缩、可审计。

Hey Robot 已经拥有比 nanobot 更适合真实机器人的执行架构：Skill OS、Foundation Model 边界、Robot Runtime、物理资源合同、任务证据和多层安全 gate。它当前更需要解决的是 Agent 核心职责过多、兼容与 fallback 分支累积，以及失败被 presentation 和恢复逻辑覆盖的问题。

因此最专业、也最务实的结论是：

> nanobot 可以作为 Hey Robot Agent Runtime 分层方式的参考实现，当前优先借鉴 AgentLoop/AgentRunner 分离、model-turn checkpoint 和 hook/SDK 可观测接口；ContextGovernor 自动修复、Runner 内 safe-boundary injection、bounded internal continuation、自动重试和自修改不迁移。长任务由 GoalStore、Supervisor 和真实事件驱动的新 deliberation 实现。

若按统一重构文档的优先级实施，Hey Robot 可以在不削弱物理安全和四层架构的前提下，成为一个核心路径简洁、失败原因明确、实验数据可信的自主具身 Agent 研究平台。复杂恢复、内部 continuation 和认知子 Agent 仍是后续独立实验，不进入本轮默认路径。

## 24. 历史代码映射（非实施清单）

> 本节的保留/移出分析仍可用于理解当前代码，但目录、协议、工具集合和阶段安排均以统一重构文档为准。

### 24.1 `cognition/runtime/runner.py`

保留：

- provider request；
- 严格工具循环；
- 参数校验；
- 纯只读工具并发批处理；主动观测和物理 skill 不并发，提交后结束当前 slice；
- iteration 和时间预算；
- 结构化 stop reason。

移出：

- TaskContract 和 EvidenceLedger；
- 任务完成判定；
- grounding perception policy；
- 用户回复合成；
- provider/tool 失败 fallback；
- 内部协议文本猜测；
- 从 execution feedback 文本解析结构化字段。

### 24.2 `cognition/core.py`

目标形态：

```python
class RobotAgentCore:
    async def handle_turn(self, payload: AgentTurnInput) -> AgentCoreResult:
        context = self.context_builder.build(payload)
        run = await self.runner.run(context)
        return self.task_evaluator.evaluate(payload, run)
```

Core 不再维护多套 fallback reply，也不根据最后一次工具调用猜测任务是否完成。

### 24.3 `cognition/task_runtime.py`

逐步拆出：

- `TaskStateStore`：TaskRun 读写；
- `TaskEvaluator`：合同和证据判断；
- `PendingTurnStore`：忙碌期间的用户输入；
- `RecoveryPolicy`：极少量确定性恢复；
- `TaskReportBuilder`：展示和报告。

不要为了保持旧调用接口增加长期 facade。每完成一次迁移就删除旧入口。

### 24.4 Provider 与 presentation

- Provider 兼容只存在于 `providers/`；
- Agent 层只消费统一 `ReasoningResponse`；
- 用户可见文本由 presentation 层根据结构化结果生成；
- presentation 不得改变 `failed`、`interrupted`、`inconclusive` 等状态；
- Provider timeout、空回复和非法协议不得转换成普通完成回复。

### 24.5 历史恢复候选（已被统一 V1 否决）

以下内容只保留为早期分析记录，不属于当前实施：

```text
NONE
REOBSERVE
OPERATOR_REQUIRED
SAFE_ABORT
```

早期草案曾考虑两个确定性恢复：

- observation stale -> `REOBSERVE`；
- 进程中断且机器人状态明确 -> 从 checkpoint 重新构造上下文。

统一 V1 已进一步收敛为：不自动 REOBSERVE，不续跑中断的模型调用，不自动重试运动技能，也不由 LLM 生成恢复类型。状态明确时由新 deliberation 正常继续；状态未知时 BLOCKED。

### 24.6 当时的目录草案（已废弃）

> 不按此目录创建文件；这里只保留分析过程。

```text
cognition/
  service/
    robot_agent.py
    task_supervisor.py
  turn/
    loop.py
    context.py
    checkpoint.py
    result.py
  runtime/
    runner.py
    model_loop.py
    tool_executor.py
    protocol.py
    budget.py
    hooks.py
  policy/
    task_evaluator.py
    grounding.py
    safety.py
  task/
    contract.py
    evidence.py
    state.py
  memory/
    long_term.py
    scene.py
    runtime.py
```

依赖方向应保持：

```text
service -> turn -> runtime
               -> policy
               -> task

runtime 不导入 RobotAgentService、TaskRuntime 或 Skill OS
policy 读取 task 和结构化 run result
SkillGateway 是唯一物理执行出口
```

## 25. 关键源码索引

### nanobot

- `nanobot/agent/loop.py`：产品层 turn lifecycle、session 并发和中途注入
- `nanobot/agent/runner.py`：通用模型工具循环
- `nanobot/agent/context_governance.py`：模型上下文修复和预算治理
- `nanobot/agent/context.py`：system prompt、memory、skill 和 runtime context
- `nanobot/agent/memory.py`：Consolidator、Dream 和 history cursor
- `nanobot/agent/subagent.py`：受限后台子 Agent
- `nanobot/agent/tools/long_task.py`：持续目标工具
- `nanobot/session/turn_continuation.py`：跨预算内部续跑
- `nanobot/session/manager.py`：session 合法边界、原子持久化和修复
- `nanobot/session/goal_state.py`：持续目标状态
- `nanobot/agent/hook.py`：生命周期 hook
- `nanobot/nanobot.py`：Python SDK facade
- `nanobot/security/`：workspace 和网络安全

### Hey Robot 对照点

- `src/hey_robot/cognition/robot_agent.py`
- `src/hey_robot/cognition/loop.py`
- `src/hey_robot/cognition/core.py`
- `src/hey_robot/cognition/runtime/runner.py`
- `src/hey_robot/cognition/runtime/message_window.py`
- `src/hey_robot/cognition/runtime/message_protocol.py`
- `src/hey_robot/cognition/checkpoint.py`
- `src/hey_robot/cognition/injection.py`
- `src/hey_robot/cognition/task_runtime.py`
- `src/hey_robot/cognition/autonomy.py`
- `src/hey_robot/cognition/skill_gateway.py`
- `src/hey_robot/skill_os/controller.py`
