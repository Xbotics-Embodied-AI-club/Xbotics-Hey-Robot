# Hey Robot 双目标优化建议：Long-Horizon Task 与多通道可交互性

> 状态：V2 增量设计建议，**不是当前实施规范**。`docs/architecture/autonomous-agent-refactor-plan.md` 所定义的 V1 最小自主内核已基本落地；本文以该内核为既有前提，只定义 Long Horizon 和多通道交互的后续增量。若批准落地，应将选定条目合并到正式实施规范。
>
> 代码基线：Hey Robot `d8ed6b26a7cca225dbc9351710e73ec9ac59e167`；nanobot `a7b8a9ed46304063c7ca0eb08cd1e34a4cc32c16`。
>
> 目的：在不削弱物理安全、可审计性和失败显式化原则的前提下，同时推进两个产品目标：可控的 Long-Horizon Task，以及 Web / 语音 / 飞书等渠道中的自然、连续交互。

## 结论

Hey Robot 不应该把 nanobot 的 `long_task -> AgentRunner 内部续跑` 原样移植过来。你现在已有的架构方向更适合具身长程任务：**Goal 是持久化的、物理执行由 Supervisor 单点提交、下一次决策由真实事件唤醒、完成由证据而不是模型文本决定。**

两个目标不是彼此独立的功能。长程任务使机器人在用户暂时不说话时仍能在**明确的安全事件边界**内推进；交互性使用户随时知道它在做什么，并能查询、取消、修正或替换目标。真正需要优化的不是“让模型连续多想几轮”，而是让一个 Goal 可以在较长时间、跨服务重启、跨感知和执行边界下，始终满足下面这条不变量：

```text
没有新的可信事实，就不做新的物理决策；
没有可验证的完成证据，就不宣布完成；
任何动作是否已发生不确定，就不自动重发。
```

nanobot 最值得借鉴的是它对**目标持久化、循环预算、上下文边界、checkpoint 和可观测性**的工程意识；最不应照搬的是它用 prompt 驱动的隐式 continuation、模型自报完成和进程内队列续跑。

同样，“用户可以让机器人做任何事情”应理解为：用户可以用任意自然语言提出请求，系统会在**已注册能力、可验证合同、当前权限和安全策略**内把它转成行动、追问或明确拒绝；它不能也不应表示用户的任意一句话可绕过安全边界直接变成机器人动作。

## 1. 当前基线：V1 最小自主内核已是 V2 的前提

`autonomous-agent-refactor-plan.md` 的重点不是“让模型多调用工具”，而是建立可审计的具身自主内核：`GoalStore`、不可变 `TaskContract`、`EvidenceLedger`、`ActionLedger`、`RobotExecutionGate`、一次一个 deliberation、一次一个外部 intent、失败显式化。当前代码已经实现了这条主链路。

因此，当前实现不是普通的聊天 Agent，而是一个事件驱动的具身任务系统：

```text
GoalCommand
  -> AutonomySupervisor + AutonomyStore (SQLite)
  -> DeliberationRequest
  -> AutonomousRobotAgentService
  -> StrictAgentRunner（一次模型请求，至多一个 ActionProposal）
  -> AutonomySupervisor
  -> ActionLedger 持久化后发布 SkillIntent
  -> SkillResult / RobotStatus / RobotObservation
  -> EvidenceLedger + 下一次 DeliberationRequest
```

关键代码证据：

| 已有能力 | 当前实现 | 对长程任务的意义 |
|---|---|---|
| 持久化目标 | `cognition/autonomous/store.py` 的 `goals`、`autonomy.sqlite3` | 目标不依赖某一段 LLM 对话，也不会因进程退出消失 |
| 不可变合同 | `TaskContract` + `contract_hash` | “完成”有稳定的可审计定义，不随模型临场改写 |
| 证据驱动完成 | `TaskEvaluator` 只接受新鲜 `EvidenceFact` | 不能通过自然语言“我做完了”结束物理任务 |
| 单步决策 | `StrictAgentRunner`：一次 provider 请求、至多一个 tool call | 每一步的因果链清楚，避免一个上下文中连发多个物理动作 |
| 事件唤醒 | `SkillResult` 成功后 `Supervisor._schedule()` 下一次 deliberation | `观察/动作 -> 结果 -> 决策` 是真实闭环 |
| 动作去重与不确定锁 | `ActionLedger`、`publishing -> unknown`、`RobotExecutionGate` | 进程/网络中断时不重放可能已发生的物理动作 |
| 有界运行 | Goal 的 wall time、deliberations、skills、电量预算 | 任务不会无限消耗机器人、时间或 token |
| 中断策略 | `DeliberationStore.interrupt_incomplete()` 把未终结模型调用变成失败 | 不把无法确认的模型/物理状态伪装成“已恢复” |

这已经对应了 nanobot 长任务方案中更可靠的部分，而且在物理安全上更严格。此前针对 `tests/autonomy` 与架构边界的检查中，129 个用例通过；整次 pytest 的非零退出来自覆盖率阈值（选测导致总覆盖率 33% 小于全局 85%），不是功能断言失败。

## 2. nanobot 和 Hey Robot 的正确对应关系

两者都在做“目标持续存在 + 多轮行动”，但外部世界不同，控制权必须不同。

| 关注点 | nanobot 的做法 | Hey Robot 当前做法 | 结论 |
|---|---|---|---|
| 目标 | `session.metadata[goal_state]`，由 `long_task` 写入 | `GoalCommand` 创建 immutable `TaskContract` + Goal snapshot | 保留 Hey Robot；合同更强 |
| 一轮内动作 | Runner 可连续执行普通数字工具 | Runner 只产生一个 `ActionProposal` | 保留 Hey Robot；物理技能不可与 shell 同等处理 |
| 触发下一步 | tool result 进入同一 Runner；必要时内存 queue 续跑 | terminal `SkillResult` 触发 Supervisor 调度下一 deliberation | 保留 Hey Robot；真实事件比合成 prompt 更可信 |
| 完成 | 模型调用 `complete_goal` | `TaskEvaluator` 以 `EvidenceFact` 满足 all-of criteria | 保留 Hey Robot；不能让模型自证完成 |
| 崩溃 | 恢复对话，未完成工具补错误，不重放 | 未终结 deliberation 失败；发布中动作变 UNKNOWN 并锁机器人 | 保留 Hey Robot；不确定状态应优先安全 |
| 预算耗尽 | 内部 continuation slice，最多 12 次 | fail/stop，依赖显式下一事件 | V2 可以增强“等待/恢复”，不能自动重发物理动作 |

因此，Long Horizon 不等于“一个超长 Agent Loop”。对于机器人，更合理的单位是：

```text
Goal（长寿命、持久化）
  = 多个 Deliberation（短、一次模型决策）
  = 多个 Skill / Observation（有明确外部终态）
```

这相当于把 nanobot 的连续 loop 拆成了**事件驱动的离散闭环**。拆分带来少量调度复杂度，却换来了更清楚的动作归属、恢复边界和安全控制。

## 3. 第二个目标：把多通道交互变成受控的任务控制面

### 3.1 当前已有入口，以及尚未闭合的部分

当前代码已经有良好的通道基础：

```text
Web HTTP / WebSocket ─┐
Voice ASR             ├─> Channel -> UserTurn -> GatewayService
Feishu 文本/图片/音频 ─┘                            │
                                                     v
                           presentation provider: route_interaction
                            ├─ conversation -> AgentReply
                            └─ goal         -> GoalCommand -> Supervisor
```

- `channels/web.py` 提供 `/turn`、WebSocket、近期回复/事件、历史和任务 cockpit API；
- `channels/voice.py` 把 ASR 文本变为 `UserTurn`，并把最终回复或重要通知 TTS 播报；
- `channels/feishu/channel.py` 把文本、图片、音频、文件转换为统一的 `UserTurn` / `MediaRef`；
- `GatewayService._reply_to_presentation_turn()` 通过一个 `route_interaction` tool 将非物理对话和物理 Goal 分开；物理请求只有形成 `objective + success_criteria` 后才会发布 `GoalCommand`；
- 旧的 `interaction/InteractionStateStore` 已因未接入生产控制链而移除；跨渠道事实统一以 Goal owner、identity binding 和 Gateway receipt 为准。

这个分层方向是对的：nanobot 的 channel 适配器也只负责把外部消息转换为统一入口，不能直接调用工具。Hey Robot 需要再进一步：把“自然语言输入”转为**类型化的用户控制事件**，再由 Goal/Supervisor 的确定性规则决定是否影响物理世界。

### 3.2 必须区分的两条平面

```text
交互/呈现平面（快、可对话、渠道相关）
  Channel -> Identity/Episode -> Interaction Router -> Reply / UI / TTS
                                      │ 只产生已验证的意图
                                      v
自主控制平面（慢、持久化、渠道无关）
  GoalStore -> Supervisor -> SkillGateway -> Skill OS -> Robot Runtime
```

交互平面可以使用 LLM 理解自然语言、做澄清、格式化回复、适配卡片/TTS；自主控制平面不能根据聊天文本直接执行动作。两条平面只通过下列明确类型交接：

| 用户语义 | 交互层输出 | Supervisor 的处理 |
|---|---|---|
| 普通聊天、能力咨询 | `ConversationRequest` / `AgentReply` | 不触碰 Goal 或动作 |
| “看看前面”“把杯子放回去” | `CreateGoalRequest` | 校验能力/实体/合同后创建 Goal |
| “现在进度如何” | `GoalQuery` | 读取 Goal、Action、Evidence、Gate，返回事实快照 |
| “停止当前任务” | `CancelGoalRequest` | 走持久化 cancel + `SkillControl`，不经 LLM |
| “急停” | `EmergencyStopRequest` | 高优先级确定性通道，绝不等待模型路由 |
| “不是左边，是右边” | `AmendGoalRequest`（候选） | 不修改 active contract；安全边界后取消旧 Goal，再创建新 Goal 或要求确认 |
| “是的，继续” | `HumanConfirmation` | 仅满足某个持久化 `WakeCondition`，不直接发动作 |
| 对账/恢复请求 | `ReconcileRequest` | 仅用权威 idle 证据解除 UNKNOWN，保留审计 |

`ConversationRequest` 等是建议的协议名；实现时也可扩展现有 `GoalCommand`，但必须避免把这些语义重新塞回 `metadata` 或 prompt 文本。

### 3.3 “任何事情”应是能力协商，不是无限权限

用户体验可以是自然语言的：例如“去厨房看一下有没有水”“帮我把杯子拿到桌上”。系统应尽力理解，但不能假设每个请求都可执行。

推荐交互流程：

```text
自然语言请求
  -> Interaction Router 识别：对话 / 查询 / 控制 / 物理 Goal
  -> 若为物理 Goal：校验已注册 Skill、实体、机器人与可验证合同
      -> 能构造确定合同：创建 Goal
      -> 缺实体、目标或成功条件：提出一条具体澄清问题
      -> 能力不存在或权限不足：明确拒绝，并说明可支持的替代
```

不能让 presentation LLM 自由发明 skill、实体 ID、成功条件或安全例外。现有 `route_interaction` 要求 `success_criteria`，再由 `SkillGateway` 校验已注册 Skill、由 Supervisor 校验实体与 `TaskContract`，已足以构成当前阶段的安全收口。为保持架构简单，暂不引入独立的 `CapabilityCatalog` 或 `contract_template`；高频任务模板只在未来确有重复产品需求时再评估。

### 3.4 活跃长任务期间的用户交互语义

这是两个目标真正会相撞的地方。不能把新用户消息直接注入正在运行的物理 Agent Loop；应在事件边界处理：

```text
WAITING_SKILL：
  status/query       -> 立即读状态回复，不影响物理执行
  emergency_stop     -> 立即控制通道
  cancel             -> 持久化终止并中断当前 skill
  correction/new task-> 记录为 pending amendment；等待 skill terminal 或 stop 确认

ACTIVE：
  status/query       -> 立即回复
  cancel             -> 阻止晚到 proposal，并终止 Goal
  correction         -> 取消/替换旧 Goal；不得修改 immutable contract
  new task           -> 若同一机器人已有 non-terminal Goal，要求“取消并替换”或拒绝排队

WAITING_CONDITION：
  confirmation       -> 精确匹配 condition_id 后唤醒
  status/query       -> 说明正在等待什么、何时过期、预算余量
```

V1 不做 pause/resume 是合理的。V2 若要提供“暂停”，必须先把它定义为哪一种：仅停止后续调度、向正在执行的 skill 发 interrupt、还是保持机器人姿态等待。三者安全语义完全不同，不能只靠一个自然语言 `pause` 词解决。

### 3.5 身份、会话、消息幂等与跨渠道回复

Long Horizon 的 Goal 生命周期通常长于一次 WebSocket、一次语音会话或一条飞书消息。因此，以下标识必须分开：

```text
principal_id    谁拥有/有权控制任务（经 identity binding）
interaction_id  这一次渠道消息（channel + account + message_id）
episode_id      对话与展示上下文
goal_id          持久化的物理任务
robot_id         被控制的实体机器人
trace_id         单条因果链的观测/调试关联
```

当前 Gateway 会分配 episode 并能解析 identity，但 `GoalCommand.command_id` 在自然语言路由和 slash 命令路径中使用 `uuid4()`。这意味着同一渠道消息被上游重复投递时，不能仅靠 GoalStore 的 `command_id` 唯一性去重。V2 应把 command id 设计为稳定派生值，例如：

```text
hash(deployment_id, principal_id, channel, account_id, message_id, semantic_action)
```

没有稳定 `message_id` 的语音输入，必须在 Gateway receipt store 生成并持久化一个输入 receipt，再生成 command id。不要使用“相同文本”去重，因为用户可能合理地连续说两次相同命令。

跨渠道状态应以 `(principal_id, robot_id, interaction_scope)` 关联，而不能仅靠 channel-local `episode_id`。Goal 的所有权、取消权和权限校验必须以持久化 principal/role 为准；渠道偏好若未来需要，应作为 Gateway 的独立展示数据实现，不能成为执行真相。

回复投递也应分两类：

- **请求回复**：默认回到发起渠道和对应 `reply_to_id`；语音保持短句，Web/飞书可展示证据、图片和 timeline。
- **任务事件通知**：按用户显式订阅的渠道投递（例如飞书完成通知、语音只播报 warning/critical），并用 `(goal_id, event_kind, state_version)` 去重，避免多通道和重连导致重复播报。

### 3.6 紧急命令必须绕过 LLM，但不能绕过审计

Gateway 对明确的 stop、cancel、status 与 confirm 已有确定性路由；其他自然语言才进入 presentation provider。旧的关键词意图分类器已移除，避免出现第二套未接入 Supervisor 的控制语义。

建议在所有 Channel 归一成 `UserTurn` 后、调用任何 LLM 前，设置一个确定性的 `SafetyCommandRouter`：

```text
ASR/text input -> normalize -> SafetyCommandRouter
  emergency-stop phrase / authenticated hardware trigger
      -> EmergencyStopRequest -> Supervisor/SkillControl immediately
  otherwise -> Interaction Router / presentation LLM
```

它必须保留原始消息、身份、时间和 receipt，仍由 Supervisor 写入可审计状态；“绕过 LLM”绝不等于“绕过 RobotExecutionGate 或动作账本”。语音通道尤其要有更高的 ASR 置信度、唤醒词/确认策略和本地硬件急停作为最终兜底，以防把环境噪声误识别为停止或动作命令。

### 3.7 交互性路线图和验收

| 阶段 | 新能力 | 关键约束 |
|---|---|---|
| I0 | 统一事实状态卡：目标、当前 skill、等待原因、证据、预算、Gate | 所有渠道展示同一 Goal snapshot，不从模型文本拼进度 |
| I1 | 自然语言 create/query/cancel + 确定性 emergency router | create 必须形成合同；cancel/stop 不经 LLM；消息 receipt 幂等 |
| I2 | 跨 Web/语音/飞书订阅与 identity binding | 任务所有权按 principal，不按某个 WebSocket/频道 |
| I3 | pending amendment、HumanConfirmation、`WAITING_CONDITION` | 修正不会注入正在执行的 skill；合同替换是显式 cancel + create |
| I4 | TaskContract + SkillGateway | “做任何事”被收敛为“尽可能理解、在已注册技能、已验证实体和证据合同内执行或清晰澄清” |

必须增加的测试：

1. 同一飞书/Web message 重投不创建两个 Goal；同文本但不同 message id 可以创建两个不同 Goal。
2. 用户从语音发起 Goal、在 Web 查询、在飞书收到完成通知时，三处看到同一个 `goal_id` 和状态版本。
3. WAITING_SKILL 时 status 查询不改变 ActionLedger；cancel 能阻止晚到 proposal；correction 不能直接改变 active contract。
4. 自然语言急停在 provider 不可用时仍能到达 Supervisor；未授权用户不能取消其他 principal 的 Goal。
5. presentation model 生成未知 skill、越权实体或非法 criterion 时，只得到澄清/拒绝，绝不发布 `GoalCommand`。
6. 语音断连、WebSocket 重连、飞书重复 callback 下，通知不会重复播报，控制命令不会重复执行。

## 4. 从已完成的 V1 到 V2：Long-Horizon 增量能力

V1 的“失败即停”对研究非常正确，但它刻意没有覆盖一部分真正的长程能力。以下不是缺陷清单，而是进入 V2 前需要做出的明确产品/安全决策。

### 3.1 任务进展只有“事实”，缺少“阶段”

目前 `EvidenceLedger` 能证明 criteria 是否满足，`ActionLedger` 能说明做过什么；但没有持久化的阶段语义，例如“已定位目标、等待人移开障碍物、正在充电、等待视觉稳定”。模型每次从完整上下文重新推断进度。

风险不是不能执行，而是任务变长后容易反复观察、选择重复动作，且 UI/运维难以解释“为什么现在没有继续”。

建议：**先不引入 LLM 生成的任务 DAG**，而是增加一个小而确定性的 `GoalProgress` 投影：

```text
ACTIVE            可调度下一次 deliberation
WAITING_SKILL     等待某个有明确 skill_id 的 terminal SkillResult
WAITING_CONDITION 等待具名外部条件（充电完成、人员确认、目标到场）
NEEDS_REVIEW      有证据冲突/重复无进展/策略不允许自动继续
BLOCKED           动作状态不确定、机器人离线或安全锁
TERMINAL          COMPLETED / FAILED / CANCELLED
```

其中 `GoalProgress` 不是新的完成判定来源；完成仍由 `TaskContract + EvidenceLedger` 唯一决定。它只回答“当前为什么没有新决策、什么事件允许再次决策”。

### 3.2 事件类型不足以表达“安全地等待”

当前连续性主要靠 `SkillResult` 成功后立即调度；skill 失败/未知通常使 Goal failed/blocked。对于真实长任务，还会遇到：用户尚未确认、机器人需要充电、资源被占用、物体暂时不可见、外部系统尚未响应。

建议新增**显式 wake condition**，而不是 nanobot 的“再塞一段继续工作提示”：

```text
WakeCondition
  kind: skill_result | robot_status | observation | human_confirmation | timer
  correlation_id: skill_id / confirmation_id / condition_id
  earliest_wake_at: optional timestamp
  expires_at: timestamp
  policy: manual_only | auto_reobserve_once | auto_resume_no_action
```

注意 `timer` 的含义只能是“允许 Supervisor 重新评估/发起一次主动观察”，不能直接等价于“再次执行上一动作”。`auto_resume_no_action` 只允许重建上下文或调用模型；模型若提出物理动作，仍需经过现有 Gate 和 admission。

### 3.3 “无进展”目前只靠硬预算兜底

`max_deliberations`、`max_skills` 很重要，但它们只能在多次重复以后停止。长程任务还需要区分：任务正在合理地多步推进，还是已经卡在“观察相同画面—做相同动作—没有新证据”的循环中。

建议先做确定性的 `ProgressMonitor`，不直接判定成功，只输出 `NO_PROGRESS`：

- 连续 N 次 observation 的语义 evidence 指纹没有新增；
- 连续 N 次同一 skill + 相同规范化参数，且没有新增满足 criterion 的事实；
- 一个 criterion 的证据已超过 `max_age_sec`，但没有可用的重新观测路径；
- 同一可恢复错误码超过固定次数。

V2 的默认动作应是 `NEEDS_REVIEW` 或 `FAILED(NO_PROGRESS)`，不是自动试到成功。只有经过单独验证的、只读 observation 可配置一次有限 `auto_reobserve_once`。

### 3.4 证据的“新鲜度”需要任务级策略

现在 `TaskEvaluator` 已按 criterion 的 `max_age_sec` 检查 freshness，这是很好的基础。但长期任务需要明确不同事实的失效逻辑：机器人位置和视觉对象关系的半衰期不同；一个抓取成功的 SkillResult 也不等于物体仍在手中。

建议：

1. 将 criterion 的 freshness 保留为合同字段；不要由模型在运行时改写。
2. 对“会被环境改变”的 object relation，完成前要求一次**独立的后验 observation 证据**，而不是只接受执行 skill 的 self-report。
3. 在 `EvidenceFact` 中可选增加 `confidence`、`sensor_id`、`calibration_version`，但阈值和融合在 perception 层完成；Evaluator 只消费已合格的类型化事实。
4. 以 evidence/source 的 provenance hash 侦测冲突；冲突进入 `NEEDS_REVIEW`，绝不选取“看起来更合理”的一条。

### 3.5 重启后的“目标仍存在”与“可安全继续”应解耦

当前设计正确地区分了：Goal 可持久化，但中断中的 deliberation 不续跑，发布中的 physical action 变 UNKNOWN。下一阶段不要为了提高表面完成率而把两者重新耦合。

建议重启恢复矩阵：

| 重启时状态 | 自动动作 | Goal 去向 |
|---|---|---|
| `ACTIVE`，没有 active deliberation/action | 仅恢复为可展示的 ACTIVE；等待明确 wake policy | 不自动调用模型 |
| 未终结 deliberation | 写 `AGENT_PROCESS_INTERRUPTED` | FAILED（保持 V1） |
| `WAITING_SKILL`，已有 terminal SkillResult | 正常消费幂等结果 | 可事件驱动继续 |
| `PUBLISHING/PUBLISHED/UNKNOWN` action | 不重新发布；查询/人工对账 | BLOCKED，执行锁保留 |
| `WAITING_CONDITION` | 重建条件订阅；条件满足后按 policy 唤醒 | 仍 WAITING，直到可信事件到达 |

这里可以借鉴 nanobot 的 checkpoint “保留事实、不重放工具”原则，但不能借鉴其内部 queue continuation。

## 5. 推荐目标架构（V2）

在不改变 Agent/Skill OS/Robot Runtime 所有权边界的前提下，增加一个**确定性的 continuation policy 层**，放在 Supervisor 内，而不放进 LLM Runner：

```text
                              ┌──────────────────────────────┐
GoalCommand / trusted event ->│ AutonomySupervisor            │
                              │  GoalStore + ActionLedger     │
                              │  EvidenceLedger               │
                              │  ProgressMonitor (pure)       │
                              │  ContinuationPolicy (pure)    │
                              └───────┬───────────────┬───────┘
                                      │               │
                         schedule one │               │ persist/subscribe
                         deliberation │               │ WakeCondition
                                      v               v
                         RobotAgentService       external event / timer
                         (one model request)          │
                                      │ ActionProposal │
                                      v                │
                              existing SkillGateway <──┘
                              + RobotExecutionGate
```

两个纯函数是核心：

```python
def evaluate_progress(goal, actions, evidence, now) -> ProgressAssessment: ...

def decide_continuation(goal, trigger, progress, gate, budget) -> ContinuationDecision:
    # SCHEDULE_DELIBERATION | WAIT | NEEDS_REVIEW | FAIL | BLOCK
    ...
```

它们不得调用 provider、不得发布 `SkillIntent`、不得从自然语言解析状态。这样每一次“为什么继续/为什么停止”都可离线重放。

## 6. 分阶段路线图

### Phase 0：V1 收尾、基线冻结与可观测性补强

目标不是重做 V1，而是冻结其行为边界、补足质量门槛，确保它能作为长程系统的地基。

- 给 `AutonomyStore`、`DeliberationStore` 加 migration version 和关闭/资源管理测试，处理现有测试中大量 SQLite `ResourceWarning`。
- 为每个 Goal 生成可下载的 timeline：GoalCommand、deliberation、proposal、action 状态、SkillResult、EvidenceFact、Gate 状态和 termination reason。
- 增加 property/fault tests：重复消息、乱序结果、重启点、`PUBLISHING` 竞态、取消与晚到 proposal、证据冲突。
- 增加 Gateway receipt 和跨渠道测试：同一渠道消息重投、跨渠道状态查询、自然语言 stop 的 provider-down 路径。
- 记录指标：每 Goal deliberation 数、skill 数、token/延迟、每 criterion 首次满足时间、UNKNOWN 率、人工介入率、无进展率。

验收：一次任务能从 trace 中无歧义地回答“哪条证据导致完成/失败、哪个动作可能已执行、为何没有继续”；V1 的状态机、工具边界和不自动恢复原则保持不变。

### Phase 1：显式等待与可观察进展

目标是支持“任务尚未失败，但当前不应行动”。

- 新增 `WAITING_CONDITION` 与持久化 `WakeCondition`，仅支持 `human_confirmation`、`robot_status`、`observation` 三类；timer 后置。
- 把 Goal 的事实状态卡接入 Web cockpit、飞书通知和语音短播报；状态、证据和预算都来自 Supervisor 的 snapshot。
- 在 GoalEvent/UI 中显示 waiting 原因、下一可唤醒条件、预算剩余和最近可信证据。
- 引入只读 `ProgressMonitor`，先只记录/告警，不改变状态；收集足够轨迹再决定 `NO_PROGRESS` 阈值。
- 给完成类合同加可选 `require_post_action_observation` 策略，防止把“技能成功”直接当作世界状态成功。

验收：重启、等待、用户确认和新 observation 都可清晰地恢复/唤醒；没有任何路径会自动重发物理 skill。

### Phase 2：受控 continuation，而非隐式 Agent Loop

目标是增加有限自主恢复，同时维持物理动作的安全不变量。

- 将 `ProgressMonitor` 变为确定性的 `ContinuationPolicy` 输入。
- 对**只读、无副作用的主动观察**允许一次、显式配置的自动重新观测；每次必须有新的 `deliberation_id` 和 trace reason。
- 对 `WAITING_CONDITION` 的可信 wake event，允许调度一次新的 deliberation；它只能重新规划，不能复用/重发前一 `SkillIntent`。
- 让 `HumanConfirmation` 和已绑定用户的 status/query/cancel 成为正式控制输入；correction 走显式 cancel-and-replace，而不是中途注入模型。
- 同一 Goal 增加 `max_auto_wakes`、`max_reobservations`、`max_no_progress_cycles`；超过即 `NEEDS_REVIEW` 或 FAILED。
- 在每次自动唤醒前再次检查 `RobotExecutionGate`、电量、机器人状态、wall-time 和技能预算。

验收：可演示“等待人确认后继续”“视觉暂时不可用后一次重观测”“动作不确定后永不自动重发”；所有自动唤醒都有独立可解释理由。

### Phase 3：仅在必要时引入静态组合任务（最后再做）

复杂家庭任务通常需要多个完成条件，但不必立刻引入通用 DAG。

- 当前不引入模板目录或通用 DAG；用一个不可变 `TaskContract`、现有 Skill 注册表和单步事件循环覆盖常见任务。
- 只有当同一类任务的合同、参数、后验验证在多个机器人配置中稳定重复时，才评估静态阶段图；每个阶段仍有自己的合同片段和可验证出口。
- 禁止模型动态创建新的成功条件、扩大允许技能集或修改预算。

验收：复杂任务的长期结构可被配置、测试和回放；模型仍只决定当前一步的候选动作。

## 7. 具体实施蓝图

本节将前述路线图拆成可独立提交、可单独验收的代码切片。实施顺序刻意是“先让用户可靠地控制和看懂任务，再增加自动 continuation”，而不是一开始扩大 LLM 自主权。

### 7.1 切片 A：冻结 V1，并建立迁移和可观测性基线

> 实施状态：增量已实现。`GoalView` 现在包含 Gate、预算、待满足 wake condition、最近 action/evidence/review；Web 任务接口提供从持久化 Goal、Action、Evidence 组装的 timeline。等待条件过期会转为 `NEEDS_REVIEW(WAKE_CONDITION_EXPIRED)`，不会自动继续。

目标：不改变当前自主行为，只确保 V1 可以安全承载后续 schema 增量。

修改区域：

- `cognition/autonomous/store.py`：增加显式 SQLite schema/migration version；为 store 增加 close/资源释放路径。
- `cognition/runtime/deliberation_store.py`：同样增加关闭和 migration 测试。
- `gateway/service.py`：输出可查询的 Goal timeline 投影。
- `tests/autonomy/`：增加乱序、重复、取消、重启、`PUBLISHING -> UNKNOWN` 的回归覆盖。

每个 Goal 的 timeline 至少包含：GoalCommand、deliberation、ActionProposal、Action 状态、SkillResult、EvidenceFact、RobotExecutionGate、预算和 termination reason。

验收：能仅从 trace/timeline 回答“哪条证据导致完成或失败”“哪个动作可能已经执行”“为什么当前没有继续”；V1 的一次一个 deliberation、一次一个外部 intent、UNKNOWN 不重发不改变。

### 7.2 切片 B：入站消息 receipt 和稳定的控制命令幂等

> 实施状态：已实现。Gateway `InteractionReceiptStore`、稳定 command id 与重复 transport message 回归测试已具备；语音循环会在进入 Gateway 前为每个 utterance 分配 UUID，并写入 `Envelope.message_id`，因此 Gateway 能以同一 receipt 去重而不会按文本去重。

目标：Web、语音、飞书的网络重投或断线重连不能创建重复 Goal、重复取消或重复动作。

在 Gateway 侧建立持久化 `interaction_receipts` 表：

```text
interaction_id       PRIMARY KEY
principal_id
channel
account_id
message_id
received_at
payload_hash
handled_result
```

规则：

- Web/飞书的 `interaction_id` 从渠道 `message_id` 稳定派生。
- 语音没有上游稳定 message id 时，Gateway 必须先持久化 receipt，再生成本地 interaction id。
- 同一 receipt 重放时直接返回 `handled_result`，不得重新路由或创建 Goal。
- 不能用“文本相同”去重；用户可能合理地连续下达相同文本命令。
- `GoalCommand.command_id` 改为从 `(deployment_id, principal_id, interaction_id, semantic_action)` 稳定派生，替代 Gateway 当前路径中直接生成的 `uuid4()`。

验收：同一飞书/Web 回调重投只创建一个 Goal；同文本但不同 message id 可创建两个独立请求；语音重连不重复执行控制命令。

### 7.3 切片 C：LLM 前置的安全控制路由

> 实施状态：首版已实现。Gateway 对明确的 emergency stop、cancel current task 和 status query 在 presentation provider 之前路由；VoiceChannel 可按渠道配置的 `activation.min_command_confidence` 拒绝明确的低置信 ASR，唤醒词/session 仍由 VoiceSessionRouter 守护。硬件急停联调仍属于真机验收范围。

目标：急停、取消、状态查询不应依赖 presentation provider 成功调用。

新增 Gateway 侧的 `SafetyCommandRouter`：

```text
UserTurn -> normalize / ASR confidence check -> SafetyCommandRouter
  emergency stop -> EmergencyStopRequest / GoalCommand(emergency_stop)
  cancel         -> CancelGoalRequest / GoalCommand(cancel)
  status query   -> GoalQuery
  otherwise      -> presentation LLM 的 route_interaction
```

实现约束：

- 急停必须通过 Supervisor/`SkillControl` 和账本审计；“不经 LLM”不等于绕过 Gate。
- 初期只识别少量高置信度明确短语；模糊的“等等”“别动”要求澄清，不能误操作。
- 语音路径需要额外的 ASR 置信度、唤醒词或确认策略；本地硬件急停仍是最终兜底。
- `GoalQuery` 只读取持久化 Goal/Action/Evidence/Gate 状态，不能触发新 deliberation。

验收：provider 不可用时，自然语言急停仍能到达 Supervisor；status query 不改变 ActionLedger；cancel 能阻止晚到的 ActionProposal。

### 7.4 切片 D：任务所有权、跨渠道状态和通知

> 实施状态：首版已实现。Goal 已持久化 owner、来源 interaction/channel 和创建者，Supervisor 会拒绝非 owner 的 cancel；基于 Store 的 `GoalView` 已接入 Web cockpit、任务详情和 status query。Gateway 已订阅 `goal.event`，向 identity binding 的跨渠道目标投递去重的 Goal 状态通知。

目标：同一用户可以语音发起、Web 查看、飞书接收通知，而不同用户不能越权控制任务。

给 Goal 的持久化记录增加：

```text
owner_principal_id
origin_interaction_id
origin_channel
created_by
```

身份与关联规则：

```text
principal_id    任务所有者/权限主体（identity binding）
interaction_id  单次渠道输入 receipt
episode_id      对话和展示上下文
goal_id          持久化物理任务
robot_id         被控制机器人
trace_id         单条因果链诊断关联
```

新增 Supervisor 的只读 `GoalView` 投影：

```text
goal_id, objective, status, current_skill, gate_state,
waiting_reason, budget_remaining, satisfied_criteria,
missing_criteria, latest_evidence, latest_failure, version
```

渠道呈现策略：

- 请求回复默认回到发起渠道和 `reply_to_id`。
- Web 显示完整 timeline、证据和媒体。
- 飞书发送状态卡和完成通知。
- 语音只播报短状态与 warning/critical。
- 通知去重键为 `(goal_id, goal_version, event_kind, target_channel)`。

验收：语音发起、Web 查询、飞书通知关联同一个 `goal_id`；未授权用户不能取消其他 principal 的 Goal；WebSocket 重连和飞书 callback 重复不会重复播报。

### 7.5 切片 E：`WAITING_CONDITION` 和人工确认

> 实施状态：首版已实现。`waiting_condition`、持久化 `wake_conditions`、owner 校验的 `GoalCommand(confirm)` 以及 `/goal confirm {"goal_id": ..., "condition_id": ...}` 已具备；`robot_status` 与 `observation` 可满足持久化的对应 wake condition，并只触发新的 deliberation。当前存在待确认的人类条件时，明确的“确认/confirm/yes”也会转为类型化 confirm 命令。UI 卡片和其他 wake kind 仍待实现。

目标：让长任务能够安全地等待用户、状态或观测，而不是把所有非立即完成的情况视为失败。

将 Goal 状态机扩展为：

```text
PENDING -> ACTIVE -> WAITING_SKILL -> ACTIVE
                  -> WAITING_CONDITION -> ACTIVE
                  -> NEEDS_REVIEW
                  -> BLOCKED
                  -> TERMINAL
```

新建持久化 `wake_conditions` 表：

```text
condition_id         PRIMARY KEY
goal_id
kind                 human_confirmation | robot_status | observation
expected_payload
created_at
expires_at
policy               manual_only | auto_resume_no_action | auto_reobserve_once
status               pending | satisfied | expired | cancelled
```

第一版只实现 `human_confirmation`：Supervisor 持久化 condition 后将 Goal 置为 `WAITING_CONDITION`；用户从任何已绑定渠道发送带 `condition_id` 的确认；Supervisor 校验 owner、状态、payload 和有效期后，才调度一个全新的 deliberation。

确认只能让系统“重新思考一次”，不能复用、重发或暗中恢复旧 `SkillIntent`。

过期 condition 由 Supervisor watchdog 标记为 `expired`，对应 Goal 进入 `NEEDS_REVIEW`；这同样不产生新的 deliberation 或动作。

验收：确认可以跨渠道完成；过期/错误确认无副作用；进程重启后等待条件仍存在且可查询。

### 7.6 切片 F：进展检测和有限 continuation

> 实施状态：已实现受控首版。纯 `ProgressMonitor` 与 `ContinuationPolicy` 有确定性测试；Supervisor 在 terminal SkillResult 后记录进展。部署可通过 `autonomy.enable_no_progress_review: true` 启用无可信证据的重复等价操作转 `NEEDS_REVIEW`；启用 `autonomy.enable_auto_reobserve_once: true` 时，`inspect_scene` 可自动补发一次全新的只读观察，第二次无进展会停在 review。默认关闭这两个 V2 策略，以维持旧部署只靠硬预算停止的兼容语义；无论配置如何，都绝不重试移动、抓取、放置或 UNKNOWN 动作。

目标：识别合理的多步推进与重复无进展的区别，同时保持不确定动作不重发。

新增纯函数模块：

```text
cognition/autonomous/progress.py
cognition/autonomous/continuation_policy.py
```

输入只允许是 `GoalSnapshot`、`ActionLedger`、`EvidenceLedger`、`RobotExecutionGate`、`BudgetState` 和明确的 wake trigger；输出只能是：

```text
SCHEDULE_DELIBERATION | WAIT | NEEDS_REVIEW | FAIL | BLOCK
```

初始规则：

- 连续 N 次观察没有新增 EvidenceFact；
- 连续 N 次相同 skill 且规范化参数相同、没有新增 criterion；
- 等待条件超时；
- 来源/内容冲突的证据。

先以 observe-only 模式记录 `NO_PROGRESS`，不要立刻改变状态。收集仿真轨迹后，再默认转入 `NEEDS_REVIEW` 或 FAILED。只有明确标为只读的 observation skill 才可在单独配置后启用一次 `auto_reobserve_once`。

绝对禁止：自动重发抓取、移动、放置等 physical skill；自动重放 UNKNOWN 动作；仅因 Goal active 而在重启后自动调用 LLM。

验收：能演示“视觉暂时不可用后自动重观测一次”；同一执行不确定动作永远保持 `BLOCKED`，直到权威对账完成。

### 7.7 切片 G：保持 Skill 与合同边界简单

> 实施状态：已采用。独立的 `CapabilityCatalog` / `ContractTemplateCatalog` 已撤回；当前由注册的 `SkillCatalog`、`SkillGateway`、Supervisor 的实体合同校验及不可变 `TaskContract` 共同收口。

目标：将“用户可以自然语言让机器人做事”实现为已注册 Skill 与可验证 TaskContract 的组合，而不是让模型发明任意 skill/合同。

自然语言“把杯子放桌上”必须产生明确的 `objective + success_criteria`；对象、目的地、可用 Skill 或后验验证不明确时，系统只提出具体澄清问题，不创建模糊 Goal。Skill 是否存在由 `SkillGateway` 校验，实体与完成条件由 Supervisor/`TaskContract` 校验。

验收：presentation LLM 生成未知 skill、越权实体或非法 criterion 时，只得到澄清/拒绝，绝不发布 `GoalCommand`。

### 7.8 推荐提交顺序

每个切片独立 PR、独立迁移、独立故障测试：

1. `feat(interaction): durable inbound receipts and stable command IDs`
2. `feat(interaction): deterministic emergency/cancel/query router`
3. `feat(autonomy): goal ownership and read-only GoalView`
4. `feat(channels): cross-channel task status and deduplicated notifications`
5. `feat(autonomy): waiting conditions and human confirmation`
6. `feat(autonomy): progress monitor in observe-only mode`
7. `feat(autonomy): bounded auto-reobserve policy`
8. `feat(autonomy): retain small TaskContract + SkillGateway boundary`

## 8. 明确“不做”的事项

以下机制即使 nanobot 支持，也不建议作为 Hey Robot Long-Horizon 的默认能力：

- 不在 `StrictAgentRunner` 中做多工具循环或并发物理工具调用；
- 不让模型调用类似 `complete_goal` 的 bookkeeping tool 自行宣布完成；
- 不保存“继续工作”的隐藏 prompt 并在内存 queue 中反复喂给模型；
- 不依据 assistant 文本、tool summary 或最近结果猜测任务进度；
- 不自动重试、重发、重放 `SkillIntent`；
- 不因 Goal active 就在重启后自动唤醒模型；
- 不把 memory/RAG 当作物理世界状态来源；
- 不让 Supervisor 调 provider，也不让 Agent 直发 `skill.intent`；
- 不用“提高成功率”的 fallback 覆盖 `UNKNOWN`、证据缺失或协议错误。
- 不以“任何事都能做”为由允许自然语言绕过已注册 Skill、合同、权限或急停审计。

## 9. 如何评价 Long-Horizon 与交互性是否成功

不要只用任务最终成功率评价。对具身系统，更重要的是正确的停止和可解释失败。

| 维度 | 建议指标 |
|---|---|
| 安全 | 未确认动作自动重发次数（必须为 0）；UNKNOWN 后错误解锁次数（必须为 0） |
| 正确性 | 无证据完成次数（必须为 0）；完成时每个 criterion 的有效证据覆盖率 |
| 长程能力 | 平均 deliberation/skill 深度；跨等待条件成功完成率；预算内完成率 |
| 抗扰动 | 消息重复、乱序、服务重启、SkillResult 丢失、机器人离线下的状态一致性 |
| 效率 | 每成功 Goal 的 token、wall time、观测次数、无进展循环数 |
| 可运维性 | `NEEDS_REVIEW` 的原因分布、人工介入后恢复成功率、trace 可重放率 |
| 交互性 | 自然语言请求形成合法合同的比例、澄清轮数、跨渠道状态一致性、重复控制命令去重率 |
| 交互安全 | emergency command 端到端延迟、未授权控制拒绝率、纠正/取消的晚到动作阻断率 |

应分别报告：

```text
原始成功率
安全终止率（BLOCKED / FAILED / NEEDS_REVIEW 是否正确）
有限自动恢复后的成功率
需要人工对账/确认的比例
```

否则自动重试可能会掩盖真实的不确定性，让“成功率”看起来变高而安全性变差。

## 10. 最小决策清单

在写 V2 代码前，建议先确认以下八项：

1. `WAITING_CONDITION` 是否进入目标状态机，以及首批允许的 wake kind 是哪些？
2. 哪些 observation 被证明是只读且可以有限自动 reobserve？默认次数是多少？
3. 哪些合同必须有 post-action observation 才能完成？
4. `NO_PROGRESS` 的初始策略是 `NEEDS_REVIEW` 还是直接 `FAILED`？
5. 真机阶段是否只允许“人工确认/新外部事件唤醒”，把任何自动 wake 限制在仿真？
6. 是否将 `principal_id` 作为跨 Web/语音/飞书的 Goal 所有权主键，并为每条入站消息建立 durable receipt？
7. 哪些自然语言紧急短语可以在不调用 LLM 的情况下进入 emergency/interrupt 通道？它们的 ASR 置信度和身份门槛是什么？
8. 哪些已注册 Skill 与可验证合同会作为“用户可直接要求”的第一批稳定能力？

我的建议是：先在仿真启用 Phase 1 和 Phase 2；真机先只启用 `WAITING_CONDITION`、人工确认、预算和观测证据增强。等到 UNKNOWN、无进展和后验验证的统计稳定后，再逐步开放有限的自动重新观测。

## 总结

nanobot 说明了“持久目标 + 有限 Agent Loop + context 管理”足以让数字 Agent 做较长的工作；Hey Robot 已经把其中最关键的思想改造成了更适合物理世界的形式：

> 长程任务不是让 LLM 一直循环，而是让一个不可变目标在一串可验证、可中断、可审计的事件边界之间持续推进。

下一步最有价值的优化是 `GoalProgress + WakeCondition + ProgressMonitor + ContinuationPolicy`，以及与它们对应的 `Interaction Router + durable receipt + principal-based ownership`。前者让系统持续且安全地推进目标，后者让用户可从任意渠道清楚地发起、查询、取消、确认和替换目标；两者都必须由 Supervisor 的确定性规则收口。这样系统会获得真正的长程能力和可交互性，同时不牺牲当前 V1 最有价值的特性：不确定就停、动作不重发、证据不足不完成。
