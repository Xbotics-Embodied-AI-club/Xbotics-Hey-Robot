# 最小 Pi-shaped Durable Agent 架构

> 状态：已实施的最小 baseline 与重构决策
>
> 日期：2026-07-25
>
> 适用范围：`src/hey_robot/cognition` 以及与它直接相连的 conversation、Skill
> execution 和 durable recovery 边界

## 0. 2026-07-25 实施状态

收缩重构已经完成：

- Phase 0：durable submit、terminal replay、restart recovery、steer 和 control plane
  characterization tests 已锁定；
- Phase 1：assistant text 自然停止，`complete_task`、`control_task` 和 forced continuation
  已从模型路径删除；
- Phase 2：`inspect_scene` 已统一到 Robot Runtime semantic observation path，
  `look_around` 不再进入默认配置或健康能力投影；
- Phase 3：Conversation 成为用户意图的唯一事实源，已删除 `task_amendments`、
  `effective_objective()` 和 waiting assistant transcript；progress/waiting 只保留为运行状态；
- Phase 4：旧 physical-call payload 兼容读取已经删除，文档、边界测试和代码度量已更新。

`AgentTask` 暂不改名为 `DurableTurn`：重命名不会减少状态或分支，只会制造迁移噪声。
这里保留的是名称，不是旧 completion protocol；它仍只是物理执行 ledger。

验证结果：`poe style`、`poe lint` 和完整测试通过；完整测试为 711 passed，覆盖率
85.65%。

## 1. 决策

Hey Robot 采用以下最小架构：

> Pi 风格、可自然停止的 Agent Core，加上只负责物理执行可靠性的 durable bridge。

Agent Core 使用一个通用停止规则：

```text
有 tool call  → 执行工具并继续
无 tool call  → assistant 文本是最终回复，结束
```

durable task/run ledger 只记录物理执行事实，不再决定 Agent 必须继续调用工具，也不再
要求模型通过 `complete_task` 宣告结束。

这套 baseline 只服务两个产品要求：

1. 可交互：用户可以获得 streaming/progress，可以在推理或物理执行期间 steer、cancel
   或 emergency stop；
2. long-horizon：任务可以跨多个工具、物理等待和进程重启继续，同时不重复执行旧动作。

除此以外的 planner、task graph、completion verifier、memory、entity cache、自动
reobservation、provider framework 和 self-improvement 均不进入 baseline。

## 2. 当前问题及其证据

2026-07-25 的实际 runtime 中，用户只问了“你看到了什么”，系统却依次执行了：

```text
inspect_scene  completed
look_around    completed
inspect_scene  completed
look_around    timed out
inspect_scene  completed
look_around    timed out
inspect_scene  completed
```

第 2 步已经返回足够回答用户的语义场景描述，但 task 仍保持 `active`，step count 达到
7，并继续等待下一次恢复。

这不是 provider 失控，主要原因是当前 harness 的控制语义：

1. 第一次 Skill 调用会创建 active task；
2. active task 存在时，模型返回的普通文本不能结束，Agent 会把文本重新塞回上下文；
3. continuation message 明确要求模型“继续观察或执行一个有界 Skill”；
4. observation outcome 又统一声明“这次观察不自动完成用户请求”；
5. 模型只能额外调用 `complete_task` 才能结束；
6. 每个物理 Skill 都在 terminal event 后创建一个新 wakeup，单 wakeup 模型轮次限制会
   重置，实际只能依靠 24 个 Skill 或 1 小时的默认上限停止。

此外，当前两条观察路径语义不一致：

```text
inspect_scene → ctx.observe() → 通常只返回 "scene inspected"
look_around   → RobotRuntime + SceneCaptioner → 返回真实场景语义
```

因此必须同时修正 Agent 停止语义和 observation result contract，不能只调整 system
prompt。

## 3. 设计原则

### 3.1 模型输出决定认知循环是否继续

Agent 不建立第二套隐藏的“必须继续”判断：

- assistant response 含 tool call：执行工具；
- assistant response 不含 tool call：返回文本并结束；
- assistant response 同时含文本和 tool call：tool call 表示尚未结束，先执行工具；
- trusted environment terminal：可以由宿主直接关闭，不要求 LLM 再确认。

不根据 active task、observation 类型或成功 step 数量覆盖这条规则。

### 3.2 Ledger 是执行账本，不是 Agent policy

Ledger 回答：

- 哪个物理请求已经持久化？
- 是否已经提交？
- 当前 pending run 是哪个？
- terminal event 是否已经处理？
- 重启后是否应恢复一次模型决策？
- 是否达到 tool count 或 deadline？

Ledger 不回答：

- 模型还应不应该继续观察？
- 当前文本是不是“阶段性文本”？
- 用户的完整语义目标是否真的完成？
- 应该选择哪个 Skill？

### 3.3 Conversation 是用户意图的唯一自然语言记录

用户初始请求和后续 steer 都按顺序持久化进 conversation。baseline 不再维护另一份
自然语言 amendment/effective-objective 事实源。

物理运行期间收到 steer 时：

1. 持久化用户消息；
2. 不并发提交第二个物理动作；
3. 当前动作到 terminal/safe point 后恢复模型；
4. 模型从最新 conversation 和真实 tool outcome 继续。

### 3.4 Progress 不进入对话历史

以下内容属于 UI/runtime state：

- accepted；
- running；
- progress；
- “已提交机器人操作，正在等待执行结果”。

它们通过 `RuntimeEvent` 或 channel progress 呈现，不写入 conversation transcript。
conversation 只保存：

- 用户消息；
- 模型最终自然语言；
- 构造模型上下文所需的规范 tool call/result。

### 3.5 先使用一个 canonical observation tool

baseline 只暴露一个能够返回完整语义的 `inspect_scene(question)`：

```text
inspect_scene
    ↓
RobotRuntime refresh
    ↓
SceneCaptioner
    ↓
self-contained semantic ToolOutcome
```

不同时维护“只采图的 inspect_scene”和“内部转动多次的 look_around”两套重叠语义。
如需环视，Agent 可以显式组合 `turn_base` 和 `inspect_scene`。只有固定评测证明
`look_around` 宏显著改善成功率/延迟后，才通过消融重新加入。

## 4. 目标运行流程

### 4.1 普通问答

```text
UserTurn
   ↓
AgentRunner
   ↓
assistant text（无 tool call）
   ↓
写入 conversation
   ↓
结束
```

不创建 durable physical task。

### 4.2 视觉问答

```text
“你看到了什么”
        ↓
模型调用 inspect_scene(question)
        ↓
persist physical receipt
        ↓
submit + yield
        ↓
terminal event
        ↓
幂等写入 semantic ToolOutcome
        ↓
恢复一次模型决策
        ↓
“我看到木盒、彩色积木、镜子和门……”
        ↓
无 tool call，结束
```

正常情况只需要一次 observation Skill 和两次模型决策。

### 4.3 多步物理任务

```text
UserTurn
   ↓
model → physical tool A
   ↓
persist-before-submit
   ↓
yield

terminal A
   ↓
model → physical tool B
   ↓
persist-before-submit
   ↓
yield

terminal B
   ↓
model → assistant text
   ↓
close durable turn
```

模型可以根据真实结果继续任意多个有界步骤，但每一步仍受 durable tool count 和
deadline 限制。

### 4.4 物理运行期间 steer

```text
physical run active
        ↓
用户发送修正
        ↓
持久化 conversation user message
        ↓
不启动第二个 physical run
        ↓
当前 run terminal
        ↓
用最新 conversation + tool outcome 恢复模型
```

### 4.5 Cancel 和 emergency stop

```text
typed control message
        ↓
直接进入 Skill/Robot control plane
        ↓
cancel pending run / emergency stop
        ↓
更新 durable state
```

这条路径不经过 LLM，也不受 Agent 是否正在 streaming 影响。

## 5. 最小模块边界

```text
Agent
├── AgentRunner
├── AgentContextBuilder
├── ToolRegistry
├── AgentToolExecutor
├── ConversationStore
└── DurableTaskStore + RunStore
```

### 5.1 `Agent`

唯一职责：

- 接受 prompt/resume；
- 调用模型；
- 有 tool call 时交给 Executor；
- inline result 后继续；
- deferred physical receipt 后 yield；
- 无 tool call 时返回最终文本。

`Agent` 不知道：

- bus topic；
- OpenAI SDK 细节；
- Skill terminal event sequence 的 SQL 实现；
- observation/completion 领域策略；
- 某个具体工具名。

### 5.2 `AgentRunner`

唯一职责：执行一次 provider decision。

输入：

```text
ModelMessage[] + ToolDefinition[] + deadline
```

输出：

```text
assistant text | one validated tool call | normalized failure
```

baseline 继续保持每次决策最多一个工具调用，避免并行物理动作。

### 5.3 `AgentContextBuilder`

只投影事实：

- system policy；
- recent conversation；
- 当前 durable turn 的原始目标；
- 最近规范 tool outcomes；
- pending physical run（如有）；
- 用户在等待期间追加的消息。

它不再生成：

- “不要把阶段性文本当最终回复”；
- “继续观察或执行 Skill”；
- “observation 不自动完成请求”；
- 任何具体工具选择建议。

### 5.4 `ToolRegistry`

只负责：

- 当前工具的 function definitions；
- 参数验证；
- 从 tool call 构造 typed proposal。

不维护 prompt instructions、policy resolver 或 provider 能力矩阵。

### 5.5 `AgentToolExecutor`

只返回三种通用 directive：

```text
continue  → inline 工具已完成，结果加入当前 loop
wait      → deferred physical run 已提交，当前进程让出控制
finish    → trusted control/environment terminal 直接结束
```

普通 assistant text 不经过 Executor，也不需要 `finish` tool。

### 5.6 Durable stores

保留两类事实：

1. Agent turn/step：session、route、objective、tool count、deadline、pending run、resume
   sequence；
2. Physical run：command、events、terminal result、sequence。

长期可以把 `AgentTask` 重命名为 `DurableTurn`，但命名迁移不是第一阶段要求。首先纠正
控制语义，避免为了重命名扩大 patch。

## 6. 最小状态语义

不要把“认知循环关闭”和“世界目标经独立验证成功”混为一谈。推荐的运行状态是：

```text
running
waiting_physical
closed
blocked
cancelled
```

- `closed`：assistant 已给出最终文本，当前 durable turn 结束；
- `blocked`：预算耗尽、不可恢复失败或必须等待人工输入；
- `cancelled`：用户取消或 emergency stop；
- trusted `environment_done` 可以附加 `success=true`，但 baseline 不需要为所有开放任务
  维护一个虚假的强完成证明。

过渡期可以继续使用现有数据库的 `completed` 字段表示 `closed`，但代码和文档不得将它
解释为独立 verifier 已证明世界目标成功。

## 7. Tool result contract

每个 Agent 可见工具结果必须 self-contained。最低要求：

```python
ToolOutcome(
    status="completed" | "failed" | "waiting",
    user_summary="模型下一轮可以直接使用的事实摘要",
    data={...},
    operation_id="...",
    retryable=False,
)
```

### 7.1 Observation outcome

成功 observation 的 `user_summary` 必须说明看到了什么，不能只写：

```text
scene inspected
camera captured
operation completed
```

正确示例：

```text
前方中央有一个木盒，盒上有红、绿、蓝色积木；背景右侧有门，墙上有镜子。
```

图像 URI、frame ID 和 entity ID 放在 `data`，语义摘要放在 `user_summary`。Agent 不需要
自行读取本地图片路径才能知道工具结果。

### 7.2 Failure outcome

失败结果明确包含：

- 实际发生的 failure mode；
- 是否 retryable；
- 已获得的部分事实；
- 不暗示必须自动重试。

retryable 只表示“技术上可以再试”，不表示 Agent 必须再调用一次。

## 8. 删除与保留清单

### 8.1 删除

- `CompleteTaskTool`；
- `CompleteTaskProposal`；
- `CompletionCheck`；
- `check_completion()` / `complete_task()`；
- active task 下拒绝 assistant text 的分支；
- `continuation_message()` 的强制继续语义；
- observation-specific completion prompt；
- `intent_kind=observation` 对 Agent control flow 的影响；
- 默认工具面中的 `look_around`；
- conversation 中重复的 waiting assistant messages；
- 如果 conversation 已覆盖需求，则删除 `task_amendments` 和
  `effective_objective()` 第二事实源。

### 8.2 保留

- AgentRunner 的一次模型决策边界；
- persist-before-submit；
- stable physical run receipt；
- terminal event sequence 幂等；
- startup reconciliation；
- terminal-undeliberated wakeup；
- 一个 session 最多一个 active physical run；
- durable tool count 和 deadline；
- pause/cancel/emergency stop；
- ConversationStore；
- ToolRegistry 的 function schema；
- ToolOutcome 自动持久化；
- `environment_done` 的 trusted terminal 语义。

## 9. 预算

需要同时存在两种预算，但都保持简单：

1. 单 wakeup model decision bound：防止 inline harness tool 在一个进程调用中无限循环；
2. durable physical tool count + deadline：跨 terminal wakeup 生效，防止每次恢复后计数
   重新归零。

预算命中后直接进入 `blocked` 并向用户说明，不自动 resume。默认值应由固定评测确定；
在此之前宁可使用较小值，也不使用当前 24 个物理步骤/1 小时作为简单问答的默认容忍度。

预算只是最后保护，不能替代正确的自然停止语义。

## 10. 分阶段重构

### Phase 0：锁定当前失败

先添加 characterization tests：

- 一次 observation 后模型返回文本，当前实现会继续；
- runtime 中 `inspect_scene` 只返回 generic summary；
- waiting 文本会进入 conversation；
- physical terminal 后新的 wakeup 会重置 model-turn counter。

测试必须使用通用 tool/result，不把具体场景答案写进 Agent Core。

### Phase 1：恢复 Pi 停止语义

- assistant response 无 tool call 时立即结束；
- active task/turn 自动关闭；
- 删除 forced continuation；
- 删除 `complete_task` 全链路；
- trusted environment/control terminal 继续保留。

完成后，“一个物理工具结果 → 一个自然语言回答”必须能够结束。

### Phase 2：统一 observation contract

- `inspect_scene` 走 RobotRuntime + SceneCaptioner 的 canonical path；
- `question` 真正传递给 captioner；
- outcome 返回 semantic `user_summary`；
- 从默认 tool surface 和配置中删除 `look_around`；
- 删除只为两条路径共存产生的分支和测试。

### Phase 3：简化交互和持久状态

- progress 只发布 RuntimeEvent，不写 conversation；
- steer 直接持久化为 conversation user message；
- terminal resume 从 conversation 读取最新用户修正；
- 删除 task amendment/effective-objective 双写路径；
- 保留 pending run、route、step count、deadline 和 resume sequence。

### Phase 4：清理命名和文档

- 评估将 `AgentTask` 改名为 `DurableTurn`；
- 删除旧 schema、旧 prompt vocabulary 和 compatibility tests；
- 更新 Pi 分析文档和增量消融路线；
- 重新统计生产代码净变化。

每个 Phase 都应独立通过 style、lint、Mypy 和完整测试。不要在同一个 Phase 加 verifier、
memory 或新的 provider abstraction。

## 11. 验收测试

### 11.1 自然停止

```text
普通问题 → 一次模型调用 → 文本 → end
```

```text
视觉问题 → inspect_scene → semantic result → 文本 → end
```

```text
工具结果不足 → 模型调用第二个工具 → 允许继续
```

必须证明 active durable turn 不会阻止 assistant text 结束。

### 11.2 Long-horizon

- physical request 先持久化再提交；
- accepted/running 状态不会触发重复提交；
- terminal sequence 重放不会重复 resume；
- 重启恢复 terminal-undeliberated step；
- 重启不会重放旧物理动作；
- durable physical tool count 跨 wakeup 累加；
- deadline 跨进程保持有效。

### 11.3 交互

- inference 中 steer 取消旧 provider request，并从新消息重新决策；
- physical run 中 steer 不产生并发物理动作；
- terminal 后模型能够看到等待期间的用户消息；
- cancel/emergency stop 不经过 AgentRunner；
- progress 可被 UI 观察，但不进入 conversation transcript。

### 11.4 Tool contract

- `inspect_scene` 成功时 summary 不是 generic lifecycle text；
- `question` 到达 scene captioner；
- retryable failure 不会由 harness 自动重试；
- ToolRegistry/System prompt/ContextBuilder 仍保持边界分离。

## 12. 明确不做

本次重构不加入：

- 信息任务/动作任务分类器；
- planner 或 task graph；
- completion verifier；
- evidence ID 选择协议；
- 强制 reobservation gate；
- entity cache；
- context compaction；
- multi-provider registry；
- sub-agent；
- memory/reflection/self-improvement。

不需要分类器的原因是自然停止规则对所有任务都相同：模型认为需要外部事实或动作时调用
工具，否则返回文本。领域差异留在 tool schema 和 tool result 中，不进入 Agent loop。

这些候选能力继续遵循
`cognition-incremental-ablation-roadmap.zh-CN.md`：只有可复现失败和固定指标证明必要时，
才逐项增加。

## 13. 最终目标

重构完成后，核心路径应可以用下面的伪代码完整表达：

```python
while True:
    decision = await model(messages, tools)

    if not decision.tool_call:
        close_durable_turn_if_any()
        return decision.text

    execution = await execute(decision.tool_call)

    if execution.directive == "wait":
        return Waiting(execution.run_id)

    if execution.directive == "finish":
        return execution.text

    messages += execution.tool_messages
```

物理 terminal event 只做：

```python
persist_terminal_event_idempotently()
messages = rebuild_context_with_tool_result()
resume_one_agent_run()
```

Agent Core 不知道这是不是第几个物理步骤，也不因为 ledger 中存在 active record 就拒绝
自然语言。Durable bridge 不替模型选择下一步，只保证已发生的物理动作可追踪、不可重复，
并能在安全边界恢复。

这就是 Hey Robot 的最小 baseline：

> 简单的通用认知循环，外加不可省略的物理可靠性。
