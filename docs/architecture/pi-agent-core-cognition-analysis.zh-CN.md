# Pi Agent Core 对 Hey Robot Cognition 的参考价值分析

> 分析日期：2026-07-24
>
> 代码范围：`/home/liber/embodied_agent/pi/packages/agent` 与
> `src/hey_robot/cognition`
>
> 本文前半部分保留重构前分析作为决策记录；第 11、14 节已按 2026-07-25
> 完成的收缩式重构更新。未来候选机制见
> `cognition-incremental-ablation-roadmap.zh-CN.md`。

## 1. 结论

Pi 的 agent core 对 Hey Robot 有很高的参考价值，但不适合直接替换当前 cognition。

Pi 最值得借鉴的不是某一个类或工具接口，而是它对通用 agent 生命周期的分层：

1. 用极薄的循环处理消息、模型调用、工具调用和下一轮决策；
2. 用状态化 `Agent` 提供 streaming、事件、steering、follow-up 和 abort；
3. 用 harness 承担 session、context transform、save point、hooks 和 compaction；
4. 把领域状态和外部副作用留给宿主系统。

Hey Robot 重构前具备不少 Pi 不负责的机制。其中只有一部分属于物理机器人
long-horizon task 的最小不变量：

- 持久化任务和步骤账本；
- 先持久化 run receipt，再提交物理 Skill；
- 按 SkillEvent sequence 幂等归约状态；
- 服务重启后的 active run reconciliation；
- 自动保存结构化 Skill outcome/evidence；
- 物理动作取消和 emergency stop 控制面。

因此，合理方向不是“把 Hey Robot 改造成 Pi”，而是：

> 形成一个 Pi-shaped agent core，加上 Hey-Robot-shaped durable robot task harness。

Pi 解决的是通用认知循环和交互生命周期；Hey Robot 的 task ledger、执行回执、
幂等终态和恢复机制继续负责物理任务可靠性。动作后强制观察、模型手工选择
evidence ID 和第二次 LLM verifier 已移出 baseline，等待失败数据证明其价值。

## 2. Pi Agent Core 的实际结构

### 2.1 极薄的循环内核

Pi 的主要循环位于：

- `/home/liber/embodied_agent/pi/packages/agent/src/agent-loop.ts:155`

它的基本执行路径是：

```text
pending/steering messages
          │
          ▼
   stream LLM response
          │
          ├── 普通文本 ──────────────┐
          │                          │
          └── tool calls             │
                  │                  │
                  ▼                  │
           validate/execute          │
                  │                  │
                  ▼                  │
             tool results            │
                  │                  │
                  └── next turn ─────┘
```

对应代码边界：

- 模型调用：`agent-loop.ts:192`
- 工具调用检测和执行：`agent-loop.ts:202`
- turn 完成后的运行状态刷新：`agent-loop.ts:226`
- steering queue drain：`agent-loop.ts:259`
- follow-up queue drain：`agent-loop.ts:262`

这个循环没有内置 planner、task graph、goal evaluator、robot state 或 memory policy。它只定义一个可重复的模型—工具反馈循环。这种克制是它保持通用性的根本原因。

### 2.2 AgentMessage 与模型消息解耦

Pi 没有把内部运行时消息限制为模型 provider 支持的消息类型，而是定义了可扩展的 `AgentMessage`：

- `/home/liber/embodied_agent/pi/packages/agent/src/types.ts:296`

在每次模型调用前才执行：

```text
AgentMessage[]
    │
    ├── transformContext
    │     裁剪、压缩或注入外部上下文
    │
    └── convertToLlm
          过滤 UI-only 消息并转成 provider 消息
                 │
                 ▼
             LLM Message[]
```

这个边界很适合机器人系统。机器人 agent 内部需要表示的内容远多于 user、assistant 和 tool result：

- observation update；
- Skill accepted/running/progress/completed；
- task checkpoint；
- evidence；
- safety intervention；
- perception invalidation；
- user steering；
- emergency control；
- recovery marker。

这些内容可以先作为结构化 runtime event/message 存在，只在模型边界投影成模型真正需要看到的上下文。

### 2.3 生命周期事件是一等接口

Pi 暴露完整的生命周期事件：

- `agent_start` / `agent_end`
- `turn_start` / `turn_end`
- `message_start` / `message_update` / `message_end`
- `tool_execution_start` / `tool_execution_update` / `tool_execution_end`

定义位于：

- `/home/liber/embodied_agent/pi/packages/agent/src/types.ts:415`

`Agent` 再把事件归约成 UI 和宿主可观察的状态：

- `isStreaming`
- `streamingMessage`
- `pendingToolCalls`
- `errorMessage`

定义位于：

- `/home/liber/embodied_agent/pi/packages/agent/src/types.ts:327`

这意味着 streaming UI、trace、session persistence 和 tool progress 不需要直接侵入 agent loop。Pi 的“可交互”不只是输出 token delta，而是有一个完整、稳定的运行生命周期协议。

### 2.4 steer、follow-up 和 abort 的语义分离

Pi 区分三类交互：

- `steer`：在当前 assistant turn 和工具执行完成后的安全点注入，影响紧接着的决策；
- `followUp`：agent 原本将结束时，再触发一轮；
- `abort`：通过 `AbortSignal` 请求中止当前 provider 或协作工具执行。

AgentHarness 接口位于：

- `/home/liber/embodied_agent/pi/packages/agent/src/harness/agent-harness.ts:707`

这三者对应不同的用户意图：

```text
“往左一点，不要继续直走”    steer / task amendment
“完成以后给我一个总结”      follow-up
“停止当前思考”              abort inference
“停止机器人”                cancel Skill 或 emergency stop
```

最后一种不能只靠 Pi 的 abort。物理机器人必须保留独立、优先级更高的安全控制面。

### 2.5 Harness 与 turn snapshot

Pi 没有让 agent loop 直接读取不断变化的全局状态，而是在 turn 开始时创建 snapshot：

- `/home/liber/embodied_agent/pi/packages/agent/src/harness/agent-harness.ts:354`

一个 snapshot 包含：

- messages；
- resources；
- tool context；
- system prompt；
- model；
- thinking level；
- active tools；
- stream options。

每个 bounded turn 使用稳定 snapshot；在 save point 后重新读取 session 和运行配置：

- `/home/liber/embodied_agent/pi/packages/agent/src/harness/agent-harness.ts:485`

这使运行时配置可以动态变化，同时不会在一个 provider request 或工具调用中途改变语义。

## 3. Pi 的 Long-Horizon 能力边界

“Long horizon”至少应区分三种不同问题：

| Horizon | Pi | Hey Robot |
|---|---|---|
| 上下文/token horizon | session、context transform、compaction 较强 | 当前主要依赖最近对话和任务 projection |
| 工作流/task horizon | 提供通用循环和 session 基础，不定义领域任务语义 | 有显式 task/step/status/budget/evidence |
| 物理/environment horizon | 不处理 | 有 SkillEvent、RobotObservation、reobservation 和 completion audit |

Pi 的 compaction 会结合 provider usage 和估算 token 判断上下文压力：

- `/home/liber/embodied_agent/pi/packages/agent/src/harness/compaction/compaction.ts:231`

它将旧历史压缩成结构化 checkpoint，同时保留最近上下文：

- `/home/liber/embodied_agent/pi/packages/agent/src/harness/compaction/compaction.ts:446`

这适合保存：

- 用户目标和约束；
- 已完成的认知工作；
- 当前进度；
- 关键决策；
- 下一步；
- 调试所需的关键上下文。

但 summary 不能成为机器人物理状态的权威来源。即使 summary 写着“杯子已经拿起”，系统仍必须由 SkillResult、RobotObservation 和 evidence ledger 判断杯子是否真的被拿起。

### 3.1 Pi 当前并非完整 durable runtime

Pi 的 durable harness 文档明确将完整恢复描述为 semi-durable 目标：

- `/home/liber/embodied_agent/pi/packages/agent/docs/durable-harness.md:19`

文档列出的待持久化状态包括：

- steer/follow-up/next-turn queue；
- queue consumption；
- pending writes；
- operation start/finish/interruption；
- turn start/finish；
- provider request start/finish；
- tool call start/finish。

参见：

- `/home/liber/embodied_agent/pi/packages/agent/docs/durable-harness.md:83`

当前 `AgentHarness` 中的 steering queue、follow-up queue 和 pending session writes 仍主要是进程内状态：

- `/home/liber/embodied_agent/pi/packages/agent/src/harness/agent-harness.ts:177`

因此，Pi 当前强项是通用 agent 生命周期、session 和 context engineering；它本身还不能提供物理任务所需的完整 crash recovery 保证。

## 4. Hey Robot Cognition 的当前结构

> 本节至第 7 节记录的是重构前基线，用于解释后续设计决策；当前实现状态以第 14 节为准。

### 4.1 AgentRunner 是良好的纯决策边界

Hey Robot 的 `AgentRunner` 位于：

- `src/hey_robot/cognition/runtime/agent_runner.py:49`

它负责：

```text
ModelMessage[] + allowed tools
               │
               ▼
          一次模型决策
               │
     ┌─────────┼──────────┐
     ▼         ▼          ▼
   文本    typed proposal  failure
```

它不执行外部 IO，并且：

- 验证消息；
- 验证 allowed tool set；
- 执行有 deadline 的模型调用；
- 每轮只接受一个 tool call；
- 将工具参数 prepare 为 typed proposal；
- 将错误归一化为结构化 failure。

这是一个符合 Pi 精神的纯边界。但它只覆盖“一次模型决策”，还不是完整的 agent runtime。

### 4.2 AutonomousAgentService 承担了过多生命周期职责

`AutonomousAgentService` 位于：

- `src/hey_robot/cognition/autonomous_agent_service.py:88`

它同时负责：

- bus subscription 和 result publish；
- session concurrency；
- conversation persistence；
- prompt/context 构建；
- agent loop；
- tool dispatch；
- sustained task 创建；
- budget enforcement；
- Skill submission；
- SkillEvent consumption；
- completion；
- user-facing result。

当前 Pi 与 Hey Robot 的对应关系为：

| Pi 层 | Hey Robot 当前对应代码 |
|---|---|
| `agentLoop` | `_run_conversation_loop`、`_run_task_step` 和多个 dispatch method |
| `Agent` | 没有独立对象；状态散落在 Service、store 和 session lock 中 |
| `AgentEvent` | conversation result、SkillEvent 和内部返回值，尚未统一 |
| `AgentHarness` | Service、ConversationStore、AgentTaskStore、TaskCoordinator 的组合 |
| `AgentTool.execute()` | prepare proposal、dispatcher、TaskCoordinator 三段式实现 |
| steering/follow-up | 当前没有清晰对应层 |
| context transform | `_conversation_context()` 和 `_skill_event_context()` 手工构造 |

当前结构的问题不是缺少功能，而是通用生命周期、机器人任务语义和 transport adapter 没有形成稳定的中间边界。

## 5. Hey Robot 当前值得保留的设计

### 5.1 先持久化，再提交物理动作

`TaskCoordinator.submit()` 先创建 pending step，再提交 SkillCommand：

- `src/hey_robot/cognition/runtime/task_coordinator.py:21`

```text
persist task step/run_id
          │
          ▼
submit external physical action
          │
          ▼
apply SkillEvent idempotently
```

这是物理 agent 的核心安全语义，可以避免机器人已经执行动作、但 cognition 没有任何 durable receipt。

### 5.2 SkillEvent 幂等归约

`AgentTaskStore.resolve_pending_step()` 使用 `last_event_sequence` 忽略旧事件和重复事件：

- `src/hey_robot/cognition/runtime/agent_task_store.py:443`

Pi 的通用 tool result 没有定义这一层外部执行幂等语义。

### 5.3 active run 启动恢复

服务启动时查询 transport 已知状态并回填 active runs：

- `src/hey_robot/cognition/autonomous_agent_service.py:153`
- `src/hey_robot/cognition/runtime/task_coordinator.py:75`

对于物理工具，这比从 conversation transcript 自动重放 tool call 更安全。未完成物理动作只有在明确声明 retry-safe/idempotent 时才允许自动重试。

### 5.4 动作后重新观察

Hey Robot 使用两层约束阻止模型把动作调用成功错误解释为世界目标完成：

- runtime reobservation gate：`autonomous_agent_service.py:482`
- completion evidence check：`agent_task_store.py:597`

发生世界变化后，完成当前场景相关任务必须引用动作后的观察证据。这是机器人 long-horizon grounding 的关键 invariant。

### 5.5 Evidence-grounded completion

任务完成经历三层检查：

```text
模型提出 complete_task
          │
          ▼
evidence ID 和步骤状态检查
          │
          ▼
独立 completion verifier
          │
          ▼
持久化 task completed
```

独立完成审计位于：

- `src/hey_robot/cognition/runtime/completion_verifier.py:50`

它明确规定动作成功只证明动作本身发生，不能自动证明到达、进入、找到或完整任务成功。这种完成语义不能被通用 agent core 的 stop reason 或 `terminate=true` 替代。

## 6. “可交互”要求下的差距

### 6.1 当前有文本 streaming，但缺少统一 interactive lifecycle

普通、无 active task 的 conversation response 会发布 text delta：

- `src/hey_robot/cognition/autonomous_agent_service.py:184`

但 active task 时，`on_text_delta` 被关闭：

- `src/hey_robot/cognition/autonomous_agent_service.py:258`

当前也没有统一暴露以下状态：

- 当前 assistant partial；
- 当前 turn phase；
- 当前 tool/Skill 状态；
- pending user input；
- agent idle/busy；
- task waiting reason；
- safe steering point。

conversation streaming 和 SkillEvent 是两套独立协议，UI 或 channel adapter 需要自行拼接它们的语义。

### 6.2 session lock 保证顺序，但不等于 steering

每个 session 使用一个 `asyncio.Lock`：

- `src/hey_robot/cognition/autonomous_agent_service.py:182`

它可以防止两个 turn 并发破坏会话状态，但当前 turn 正在模型调用或普通工具执行时，新用户输入只能等待锁。系统缺少：

- 在下一个安全点追加约束；
- 修改下一 bounded step；
- 请求“完成当前动作后暂停”；
- 区分 steer 和普通 follow-up；
- 中断当前模型生成。

Pi 的 steering/follow-up/abort 可以作为这一层的直接参考。

### 6.3 active task 缺少 amendment 语义

当前 active task 支持：

- complete；
- cancel；
- block；
- emergency stop。

但缺少：

- revise objective；
- replace objective；
- add/remove constraint；
- pause/resume；
- steer next action。

例如任务目标是“去厨房”，用户中途说“改去门口”，这条消息可以影响临时模型上下文，但 durable `AgentTask.objective` 仍然是旧目标，completion verifier 也继续审核旧目标。这是交互式 long-horizon task 的语义缺口，不能只靠消息队列解决。

### 6.4 必须区分四种停止语义

未来即使采用 Pi 风格接口，也需要明确区分：

| 操作 | 作用域 |
|---|---|
| abort inference | 停止模型生成 |
| abort local tool | 停止支持协作取消的普通工具 |
| cancel Skill | 请求 Skill Worker 安全取消动作 |
| emergency stop | 绕过 agent deliberation 的优先控制面 |

不能把它们统一为一个 `AbortSignal`。

## 7. Long-Horizon 要求下的风险

### 7.1 Conversation history 与 task ledger 分裂

`ConversationStore` 只持久化最终 user/assistant 文本，并且默认只取最近 16 条：

- `src/hey_robot/cognition/runtime/conversation_store.py:21`

它没有持久化：

- assistant tool calls；
- tool results；
- turn boundaries；
- provider usage；
- interruption；
- context snapshot；
- steering provenance。

任务状态由 `AgentTaskStore` 单独保存，再通过 projection 注入，但 projection 只包含最近 6 个步骤：

- `src/hey_robot/cognition/runtime/agent_task_store.py:638`

这意味着任务事实是 durable 的，但模型可见的任务记忆较薄。长任务可能出现：

- 忘记早期用户约束；
- 看不到早期 evidence ID；
- 重复已完成步骤；
- completion 时不知道应引用哪些早期证据。

Pi 的 context transform 和 compaction 适合补足模型上下文，但只能压缩 conversation/deliberation，不应压缩掉权威 task ledger。

### 7.2 终态落库后存在 deliberation 恢复窗口

terminal SkillEvent 的当前处理顺序是：

```text
apply terminal event 到 task_steps
          │
          ▼
运行下一次 conversation loop
          │
          ▼
发布 conversation result
```

如果进程在 terminal event 已落库之后、下一次 deliberation 之前崩溃，重启时 `active_skill_steps()` 只返回 pending/running step。已经 terminal 的 step 不会再次触发 agent continuation。

可能形成以下状态：

```text
task.status = active
last step = completed/failed
active run = none
pending deliberation trigger = lost
```

任务会等待下一条用户消息才重新进入认知循环。这说明 Hey Robot 已经实现了外部动作恢复，但还缺少完整的 agent operation journal，例如：

- operation started；
- event applied；
- deliberation scheduled；
- deliberation completed；
- result published；
- operation settled/interrupted。

这与 Pi durable harness 文档提出的 operation/turn/tool journal 问题高度一致。

### 7.3 有 task budget，但没有 context budget

`_run_conversation_loop()` 当前有：

- 每个 slice 最大 8 个步骤；
- continuation 上限；
- Skill 数量上限；
- wall-time deadline。

参见：

- `src/hey_robot/cognition/autonomous_agent_service.py:228`

这些可以限制物理任务规模，但没有 provider context-window/token budget。多轮工具结果可能在达到 task budget 前先撑满模型上下文。

### 7.4 Prompt 与实际工具投影存在词汇漂移

`src/hey_robot/templates/agent/SYSTEM.md` 使用 `request_observation` 和 `request_skill` 描述决策协议，但当前 `ToolRegistry` 实际为每个具体 Skill 暴露独立工具，例如 `inspect_scene`、`move_base`，测试也使用具体 Skill 名称。

这可能是历史抽象迁移留下的协议漂移。它不一定立即导致错误，但增加了模型决策的不确定性，也说明 prompt、tool projection 和 agent runtime 缺少统一 owner。

## 8. 可借鉴、需改造与不应照搬的部分

### 8.1 可以直接借鉴

1. **独立 AgentCore/AgentLoop**

   只负责消息、模型、typed tool call、tool result 和继续条件，不拥有机器人任务状态。

2. **清晰 lifecycle 边界**

   Agent、tool execution 和 durable wakeup 有明确 owner；是否增加统一事件协议由可观测性
   failure 决定，不是 baseline 必需品。

3. **steer / abort 的语义区分**

   在此基础上增加机器人专用 task amendment、Skill cancel 和 emergency stop。

4. **单一 context projection**

   在模型边界统一投影 conversation、task 和 outcome；entity、summary 等按实验增加。

5. **turn snapshot 和 save point**

   一次模型调用使用不可变 snapshot；bounded step 完成后刷新 task、observation、tools 和 steering。

6. **tool progress callback**

   将 Skill accepted/running/progress/completed 投影为统一的 tool execution update。

### 8.2 需要机器人化改造后借鉴

1. **Session tree**

   可用于 conversation 和 deliberation 审计，但物理世界不能 branch/rollback。切换会话分支不能暗示机器人状态回滚。

2. **Context compaction**

   可以压缩用户对话和认知历史；task、step、evidence、run receipt 必须保留为结构化权威数据。

3. **AbortSignal**

   适用于模型和支持协作取消的普通工具，不能代替 Skill cancel 或 emergency stop。

4. **Dynamic tools**

   可以动态投影当前可用能力，但物理工具集合变化应在 turn snapshot/save point 生效，并经过 capability/safety validation。

### 8.3 不应照搬

1. **默认并行执行多个工具**

   Pi 默认支持 parallel tool calls；Hey Robot 每次只接受一个物理动作是正确的安全约束。即使未来允许普通非物理工具并行，也应与 physical Skill 明确分组。

2. **根据 transcript 自动重放未完成工具**

   物理动作只能依据 durable run receipt、transport status、idempotency 和 retry-safe metadata 恢复。

3. **使用通用 terminate flag 确认任务完成**

   机器人任务至少需要 durable successful outcome；有权威环境 predicate 时以 predicate
   为准。更强的 evidence check 和 verifier 是候选增强，不是 core 的固定组成。

4. **将 summary 当作世界事实**

   summary 只是认知上下文，不能替代 RobotObservation、SkillResult 和 evidence。

## 9. 推荐的目标边界

以下是架构边界建议，不是本轮实施计划：

```text
Channel / Bus Adapter
        │
        ▼
Interactive Agent Runtime
- prompt / steer
- streaming / inference abort
- direct result callback
- per-session run state
- turn snapshots / save points
        │
        ▼
Generic Agent Core
- model → one typed proposal
- tool result → next turn
- context transform boundary
- no robot/domain state
        │
        ▼
Robot Task Harness
- AgentTaskStore
- TaskCoordinator
- task amendment
- budgets
- evidence ledger
- reobservation gates
- completion verifier
- operation journal/recovery
        │
        ▼
Skill / Robot Control Plane
- persist-before-submit
- progress/result events
- cancel
- emergency stop
- transport reconciliation
```

各层的状态所有权应保持明确：

| 状态 | Owner |
|---|---|
| assistant partial、pending messages、当前 turn | Interactive Agent Runtime |
| model/tool feedback loop | Generic Agent Core |
| objective、task status、step、evidence、budget | Robot Task Harness |
| physical run ownership 和实际执行状态 | Skill/Robot Control Plane |
| conversation/deliberation history | Session/Event Store |
| 当前物理世界事实 | RobotObservation、SkillResult、RobotStatus |

## 10. 四篇论文与 Pi 的共同启示

本节对照以下四份材料：

- `docs/references/Harness Engineering for Self-Improvement.md`
- `docs/references/harness_VLA.md`
- `docs/references/hi_robot.md`
- `docs/references/Hi-VLA.md`

这些工作提供了很多可选机制，但不应将“论文中存在”等同于“Hey Robot 现在必须实现”。它们与 Pi 真正一致的部分，反而是一个很小的闭环：

```text
用户目标 + 当前观察 + 最新反馈
              │
              ▼
      选择一个 bounded action
              │
              ▼
           执行一次
              │
              ▼
      权威结果 / 新观察
              │
              ▼
       继续、完成或停止
```

### 10.1 Hi Robot：交互不等于复杂状态机

Hi Robot 最有价值的启示是：当用户补充、纠正或改变意图时，高层根据当前上下文重新产生一个当前原子命令。它支持 Hey Robot 引入 `steer` 和 objective amendment，但不支持一开始就构建大型的 steering state machine。

对 Hey Robot 而言，最小交互语义可以只有：

1. 当前物理动作不可安全中断时，收下 steer，在下一个安全点生效；
2. 可取消的 Skill 通过 Skill control plane 取消；
3. stop / emergency stop 不经过 LLM，始终使用独立的高优先级控制面；
4. steer 生效后，使用最新 objective 和最新 observation 重新决策。

### 10.2 Hi-VLA：先做好 termination 和 observation

Hi-VLA 对当前最有实践价值的结论不是“增加更长的 memory”，而是：

- termination 判断是 long-horizon 成败的关键部分；
- observation representation 直接影响高层是否能做对下一步；
- 低层执行器必须可被高层用稳定、有界的命令驱动；
- 最近一步 memory 可以与 full memory 相当，甚至更好；
- 当前 episode summary 没有显示出稳定收益；
- 跨 episode 的 affordance/failure memory 才显示出更明确的价值。

因此，Hey Robot 不应默认把整段 conversation、所有 step 和 summary 都塞回 prompt。应优先保证“当前目标 + 最新可信 observation + 最近 action outcome”的质量。这也意味着 summary 只能是上下文压缩，不是物理事实源。

### 10.3 Harness VLA：小型 primitive vocabulary 和单步闭环

Harness VLA 支持以下约束：

- 保持固定、小型、结构化的 primitive vocabulary；
- 高层每次只提交一个 structured call；
- 每个动作执行后重新观察，不预先开环生成长 action list；
- VLA 应是 contact-rich primitive 之一，而不是 cognition 的另一个中心；
- Task-Specific Memory 和 Global Memory 是闭环稳定之后的增强项。

这与 Hey Robot 当前“每轮最多一个物理工具”一致。应保留这个约束，不因 Pi 支持 parallel tool calls 就放宽物理工具并发。

### 10.4 Harness Engineering：简单和通用是设计约束

Harness Engineering 明确支持 deliberately simple and generic 的 harness，但其 self-improvement、sub-agent、evolution 和 meta-harness 是远期探索方向。它们不应现在进入 Hey Robot 生产 core。

当前真正可借鉴的是实验方法：每个新机制都要对应一个可复现 failure 和一项可观测改善，而不是把所有可能的 harness feature 一次堆入系统。

## 11. 以“最小可用闭环”重新分类当前 Cognition

当前 `src/hey_robot/cognition` 约三千行，复杂度主要集中在 `autonomous_agent_service.py` 和 `agent_task_store.py`。简化不应按文件数量或行数硬删，而应按“是否维持交互与物理 long-horizon 的最小不变量”分类。

### 11.1 Keep：现在就是 core

| 能力 | 保留原因 |
|---|---|
| `ModelClientLike` | 保持 agent core 与 provider transport 解耦 |
| `AgentRunner` | 已是较纯的单次 model/tool 决策边界 |
| `ToolRegistry` 与 typed tool schema | 给模型一个小型、明确的 action vocabulary |
| 单物理工具调用约束 | 避免物理动作竞争和不可推理的并发副作用 |
| `TaskCoordinator` 的 persist-before-submit | 是 crash-safe 物理执行的核心不变量 |
| SkillEvent sequence 幂等归约 | 对抗重复、乱序事件 |
| active run reconciliation | 是进程重启后恢复物理任务的基础 |
| cancel / emergency stop | 机器人安全控制面，不可并入普通 prompt |
| 最小 task/step durable receipt | 记录 objective、call、outcome 和 pending run，不依赖 LLM memory |

### 11.2 Simplify：保留问题，缩小机制

| 当前机制 | 建议的最小形态 |
|---|---|
| completion | `environment_done` 权威完成；显式 `complete_task` 只确定性要求至少一个成功步骤 |
| evidence | Store 自动保存 ToolOutcome/evidence；模型不再手工传 evidence ID |
| duplicate observation gate | 优先下沉为 observation Skill 的 freshness/identity，不做通用 core 规则 |
| reobservation | 由模型读取 Skill outcome 后选择 observation；core 不强制特定领域顺序 |
| entity context | baseline 删除独立订阅和缓存；需要事实时显式调用 observation Skill |
| 多层 budget | 先收敛为 `max_steps + deadline`，其他 budget 由具体 failure 驱动增加 |
| conversation + task projection | 建立一个权威 event/step history，对 UI 和 model 做不同 projection，不再维护两套历史语义 |

### 11.3 Move out of core：可用但不应定义 Agent

- scene captioning 和特定 perception representation 应作为 observation provider/Skill；
- 具体 bus topic、channel payload 和 transport retry 应属于 Service adapter；
- battery policy、特定机型 safety threshold 应属于 robot control/safety policy；
- VLA 应作为可替换的 Skill backend，不进入通用 agent loop；
- trace writer 应由 lifecycle event subscriber 实现，不成为决策路径的必选依赖。

### 11.4 Defer：有证据再加

- long-term memory、reflection 和 memory curator；
- task-specific/global memory；
- self-improving harness 和自动 prompt/code evolution；
- sub-agent、skill discovery 和动态自组织；
- 复杂 planner/task graph；
- 默认并发物理工具。

Hi-VLA 显示的跨 episode affordance memory 值得保留为未来方向，但前提是已经积累足够的跨 episode 数据，并能将成功率改善归因给 memory，而不是系统其他变化。

### 11.5 已验证的删除项

Patch 5 对仓库内调用者完成审计后，已经删除：

- 只在 runtime package 导出、从未进入运行路径的 `RunTraceWriter`；
- 无生产调用者的 `ToolDispatcher` 和旧 `tools/robot.py` 转发模块；
- `proposal()` compatibility alias，所有调用统一为 `prepare()`；
- 无消费者的 `skill_result_timeout_sec`、`min_battery_percentage`、`enable_auto_reobserve_once`、`enable_no_progress_review`；
- `hard_max_continuations` 和持久化 `continuation_count`。
- 无调用者的 `EntityResolver.resolve()`、`entity_catalog` 和 `entity_aliases`；
- 无消费者的 Agent lifecycle callback；streaming 继续使用已有直接 callback；
- 最后一个 `start_skill_step()` compatibility API。
- `TaskCompletionVerifier` 及其第二次模型调用；
- `complete_task.evidence_ids` 手工协议和 deterministic reobservation gate；
- cognition 内部的 `RobotObservation` 订阅、`EntityResolver` 与 entity prompt 投影；
- resume 时硬编码的 `inspect_scene`；恢复只从最近 durable outcome 继续；
- 仅有两个兼容包装的 coordinator reconciliation API；
- 没有独立运行语义的 `ConversationTurn.follow_up`（gateway 统一映射为 steer）。

旧 SQLite 数据库中已经存在的 `continuation_count` 列不做破坏性表重建；新代码不再
选择、写入或暴露它。durable step 仍保留 evidence IDs，作为执行结果记录和未来消融
数据，但它不再扩大模型工具协议。

## 12. Provider 选择：自己维护边界，不自己重写 SDK

Hey Robot 适合继续维护自己的 **canonical model protocol**，但不适合自己维护一个庞大的 provider framework。推荐分工是：

```text
Hey Robot cognition
  只依赖 ModelClientLike / canonical message & tool types
                         │
                         ▼
                 OpenAI adapter
  负责类型转换、stream event 归一和错误语义
                         │
                         ▼
              官方 OpenAI Python SDK
  负责 HTTP、认证、重试、SSE 与 API 细节
```

这里的“自己维护 provider”应只指维护一个薄 adapter，而不是重写 transport SDK。这样做有三个原因：

1. cognition 需要的是稳定语义：message、tool call、tool result、usage、stream delta 和 abort；
2. OpenAI API 的传输细节应交给官方 SDK，避免重复维护协议变化；
3. robot task、steering、durable recovery 和 safety 是 Hey Robot 的领域语义，不应让 OpenAI Agents SDK 或任何 provider SDK 接管。

因此建议是：继续使用官方 OpenAI Python SDK 作为 transport，保留 `ModelClientLike`，并将 adapter 限制在尽可能小的表面。如果未来真正需要第二个 provider，再用相同协议增加第二个 adapter；不要为了假想的多 provider 需求预先构建 registry、capability matrix 和 fallback router。

## 13. 建议的最小模块与渐进路线

### 13.1 先收敛到四个概念

不必立即拆成十几个类。目标语义只需要四个概念：

```text
Agent
- prompt / steer / resume / abort
- model-tool feedback loop

Tool
- name / schema / execute
- physical tool 内部继续使用 TaskCoordinator

Session
- messages
- active objective / status
- ordered calls / outcomes
- pending run
- deadline

Service
- bus / channel adapter
- prompt / steer / stop 路由
- per-session Agent
- event publish
```

这是概念上的所有权边界，不意味着必须一次创建四组新 abstraction 或迁移全部表结构。

### 13.2 按失败驱动的增量顺序

1. **最小通用 loop**：prompt、one tool、outcome、next turn 和 streaming。
2. **物理持久性**：task/step receipt、pending run、reconciliation、cancel 和 emergency stop。
3. **真实交互**：steer、objective amendment、pause/resume，并明确安全生效点。
4. **由失败补强闭环**：只在评测暴露问题后增加 success detector、reobservation、scene representation 或 completion fallback。
5. **跨 episode 学习**：有足够数据后才引入 affordance/failure memory。
6. **远期探索**：最后才评估 self-improvement、sub-agent 和 skill discovery。

### 13.3 每个新功能的准入门槛

任何进入 cognition core 的新机制，至少应回答：

1. 它解决哪一种可复现 failure？
2. 哪一项评测指标会因此改善？
3. 能否用 prompt、ToolOutcome 或 Skill contract 更简单地解决？
4. 删除这个机制会破坏哪些真实任务或测试？

如果没有具体答案，默认不进入 core。这个门槛比“某篇论文使用了它”更适合 Hey Robot 当前阶段。

## 14. 结合当前代码的具体重构方案

本节将前面的架构判断收敛为可分步提交的工程方案。这不是“重写 cognition”，而是以当前已通过测试的两个边界为锚点：

- 保留 `runtime/agent_runner.py` 作为无 IO 的单次模型决策器；
- 保留 `runtime/task_coordinator.py` 作为物理 Skill 的 durable execution 边界。

重构的主要对象是 `autonomous_agent_service.py`：它当前从 `_run_conversation_loop()` 到 `_handle_skill_event()` 同时拥有 Agent loop、Tool execution、task policy、completion、bus 和 resume 语义。

### 14.1 当前方法的具体去向

| 当前代码 | 目标 owner | 处理 |
|---|---|---|
| `AutonomousAgentService._on_turn()` | Service | 保留 bus decode/publish，其余委托给 per-session `Agent` |
| `_conversation_context()` | `AgentContextBuilder` | 与 `_skill_event_context()` 合并为一个 context projection |
| `_run_conversation_loop()` | `Agent` | 迁移为通用 model–tool feedback loop |
| `_run_task_step()` | `Agent` | 使用 `AgentRunner.run()` 完成一次决策 |
| `_dispatch_skill_call()` | `AgentToolExecutor` | 创建 task、检查 budget、调用 `TaskCoordinator.submit()` |
| `_dispatch_harness_tool()` | `AgentToolExecutor` | 执行 non-physical tool，返回统一 directive |
| `_dispatch_complete_task()` / `_complete_task()` | `AgentToolExecutor` | 收敛为成功步骤检查和 task terminal write |
| `_dispatch_control_task()` / `_control_task()` | `AgentToolExecutor` + Service control path | Agent 自主 block/cancel 留在 executor；用户 stop 走独立路径 |
| `_reobservation_gate()` | 删除 | 是否观察由模型和 Skill contract 决定 |
| `_duplicate_observation_gate()` | 删除 | frame-ID 启发式不属于通用 Agent core |
| `_consume_skill_events()` | Service | 保留 transport subscription |
| `_handle_skill_event()` | Service + `Agent.resume()` | Service 归约 event，Agent 负责恢复决策 |
| `_on_robot_observation()` | 删除 | cognition 通过 observation Skill 获取所需事实 |
| `_tool_outcome_context()` 等 prompt helper | `AgentContextBuilder` | 改为从结构化 state 生成 model projection |

`AgentToolExecutor` 是唯一执行边界；无生产调用者的 `ToolDispatcher` 已删除，避免 registry、dispatcher、executor 三层路由。

### 14.2 目标文件结构

第一次重构只新增三个运行时文件，不拆出更多细粒度 manager：

```text
src/hey_robot/cognition/
├── autonomous_agent_service.py       # 只负责 bus/channel 和组装
├── runtime/
│   ├── agent.py                    # per-session Agent lifecycle/loop
│   ├── agent_context.py            # state -> ModelMessage[]
│   ├── agent_runner.py             # 保留：单次、无 IO 决策
│   ├── agent_task_store.py         # 保留：durable task/step/run
│   ├── conversation_store.py       # 暂时保留：UI transcript
│   └── task_coordinator.py          # 保留：physical execution boundary
└── tools/
    ├── executor.py                 # 执行 typed prepared call
    ├── registry.py                 # 保留：schema + prepare
    └── ...
```

不建议立即把 `ConversationStore` 和 `AgentTaskStore` 做大规模数据库合并。先用 `AgentContextBuilder` 明确两者语义：

- `ConversationStore` 是用户和 assistant 的 UI transcript，不是世界事实；
- `AgentTaskStore` 是 objective、physical call、outcome、evidence 和 pending run 的权威记录；
- observation Skill 返回的 `RobotObservation` / `SkillResult` 仍是当前世界事实源。

先统一读模型，再决定是否统一 SQLite 表，可以避免将架构重构与数据迁移风险绑在一起。

### 14.3 `Agent` 的具体协议

`Agent` 是 per-session 对象，而不是全局 singleton。Service 只保留：

```python
self._agents: dict[str, Agent]
```

最小公开接口：

```python
class Agent:
    async def prompt(self, command: AgentCommand) -> AgentRunResult: ...
    async def resume(self, trigger: ResumeTrigger) -> AgentRunResult: ...
    async def steer(self, command: AgentCommand) -> AgentRunResult: ...
    async def abort_inference(self) -> None: ...
```

建议的核心类型：

```python
@dataclass(frozen=True)
class AgentCommand:
    session_key: str
    interaction_id: str
    envelope: Envelope
    text: str

@dataclass(frozen=True)
class ResumeTrigger:
    task_id: str
    source: Literal["skill_terminal", "startup_recovery", "user_resume"]
    step_id: str | None = None

@dataclass(frozen=True)
class AgentRunResult:
    status: Literal[
        "responded", "waiting", "completed", "blocked", "cancelled", "failed"
    ]
    text: str
    operation_id: str | None = None
```

`Agent._drive()` 只保留一个简单 loop：

```python
async def _drive(self, trigger) -> AgentRunResult:
    while True:
        state = self._sessions.view(self.session_key)
        if state.deadline_expired:
            return self._block_for_deadline(state)

        messages = self._context.build(state, trigger)
        decision = await self._runner.run(...)

        if decision.status == "failed":
            return AgentRunResult("failed", ...)
        if decision.status == "returned":
            if state.active_task is None:
                return AgentRunResult("responded", decision.final_text or "")
            trigger = self._continue_active_task(state, decision)
            continue

        execution = await self._executor.execute(state, decision)
        if execution.directive == "continue":
            trigger = execution
            continue
        return execution.result
```

这个 loop 不引入 planner、task graph、memory curator 或 provider-specific object。`AgentRunner` 仍然只返回 text / one prepared call / typed failure。

实现审计发现 lifecycle callback 没有任何消费者，因此没有为了对齐 Pi 而保留空抽象。文本 streaming 继续通过 `TextDeltaCallback` 直接交给 Service；durable Skill 状态继续通过 `SkillEvent` 表达。未来只有出现至少一个真实 metrics/trace 消费者时，才重新引入统一 lifecycle event。

每个 Agent 内部只允许一个 `_run_task: asyncio.Task` 存在。一个小型 lock 只保护“替换当前 inference / 排入 steer / 设置 waiting”这些状态转移，不在整个模型请求或物理 Skill 周期内持有 lock。这样 steer 和 emergency control 不会被当前长请求阻塞。

### 14.4 `AgentToolExecutor` 的具体协议

Tool executor 只接收已被 `ToolRegistry.prepare()` 校验的 typed call：

```python
@dataclass(frozen=True)
class ToolExecution:
    directive: Literal["continue", "wait", "finish"]
    outcome: ToolOutcome
    result: AgentRunResult | None = None

class AgentToolExecutor:
    async def execute(
        self,
        state: AgentSessionView,
        decision: AgentTurnResult,
    ) -> ToolExecution: ...
```

各 call 的语义保持明确：

| call | executor 行为 | directive |
|---|---|---|
| `SkillCallProposal` | 必要时创建 task，然后 `TaskCoordinator.submit()` | pending/running 返回 `wait`；同步终态返回 `continue` |
| `HarnessToolCall` | 执行 handler 并生成结构化 outcome | `continue` |
| `CompleteTaskProposal` | 至少一个 durable successful step；环境 predicate 可直接完成 | 成功 `finish`，拒绝 `continue` |
| `ControlTaskProposal` | block/cancel 并通过 SkillClient 取消 active run | `finish` |

物理 Skill 仍严格遵循：

```text
ToolExecutor
   │
   ├─ AgentTaskStore.add_pending_step()   # 持久化原 proposal 和 pending receipt
   │
   └─ SkillClient.submit()                # 之后才发生物理提交
```

`HarnessToolCall` 当前没有 durable receipt。因此在它被纳入通用 recovery 前，extra tool 应被限定为 read-only 或业务幂等；有不可重复副作用的 tool 必须走与 Skill 相同的 operation journal，不能仅依赖 transcript。

`TaskCoordinator.submit()` 直接将完整 `SkillCallProposal` 传给 `add_pending_step()`，避免 model-visible objective 和 durable receipt 之间产生两套语义。迁移期的 `start_skill_step(tool_name, arguments)` 已删除。

### 14.5 Context 不再有两条分叉路径

当前 `_conversation_context()` 和 `_skill_event_context()` 分别构造 prompt，容易漂移。改为：

```python
class AgentContextBuilder:
    def build(
        self,
        state: AgentSessionView,
        trigger: AgentCommand | ResumeTrigger | ToolExecution,
    ) -> tuple[ModelMessage, ...]: ...
```

`AgentSessionView` 是只读 projection，第一阶段可以同时从两个现有 store 读取：

```python
@dataclass(frozen=True)
class AgentSessionView:
    session_key: str
    transcript: tuple[ModelMessage, ...]
    active_task: AgentTask | None
    recent_steps: tuple[AgentTaskStep, ...]
```

发送给模型的默认上下文收敛为：

```text
system policy
+ active objective / latest user guidance
+ last action outcome
+ small recent transcript
```

工具 schema 不属于 prompt projection。参考 Pi Agent Core，model request 将
`systemPrompt`、`messages` 和 `tools` 作为三个并列输入：

```text
AgentContextBuilder ──→ system policy + runtime messages
ToolRegistry        ──→ function definitions
AgentRunner         ──→ model.chat(messages=..., tools=...)
```

因此 `SYSTEM.md` 只描述通用行为边界，不列举工具名、参数、具体 observation 行为或
领域示例；`ToolRegistry` 不再提供 `instructions` 文本，`AgentContextBuilder` 也不依赖
Registry。每个 Skill/Harness Tool 的 description 和 parameters 是能力语义的唯一来源。

在 Skill terminal resume 时，应使用 step 已保存的 `tool_call_id`、proposal 和 outcome 重建标准 assistant-tool pair，而不再将“Skill 已返回终态事件”伪装成新 user message：

```text
assistant(tool_call_id, name, arguments)
tool(tool_call_id, structured ToolOutcome)
```

这样 user message 只表示真实用户输入，tool result 始终表示权威执行结果。

架构测试禁止 `SYSTEM.md` 和 ContextBuilder 引用具体工具名，防止未来再次形成
“prompt 工具说明 + function schema”两条事实源。

### 14.6 物理等待与恢复的准确流程

物理 call 被接受后，Agent 必须结束当前 model loop 并返回 `waiting`，不持有模型请求或 session lock 等待机器人：

```text
Agent proposes Skill
        ↓
persist pending step
        ↓
submit Skill
        ↓
AgentRunResult(waiting)
        ↓
Service 可继续接收 steer/stop
        ↓
terminal SkillEvent
        ↓
TaskCoordinator.apply(event)
        ↓
Agent.resume(step)
```

为修复已发现的“terminal event 已落库，但下一轮 deliberation 未开始”崩溃窗口，建议在 `sustained_tasks` 增加最小 durable wakeup 状态：

```text
resume_required INTEGER NOT NULL DEFAULT 0
resume_after_sequence INTEGER NOT NULL DEFAULT 0
```

状态转移必须和 step 更新使用同一 SQLite transaction：

1. terminal SkillEvent 落库时，同时设置 `resume_required=1` 和对应 sequence；
2. Service 调用 `Agent.resume()`；
3. Agent 产生下一个 physical step 时，在写入 pending receipt 的同一 transaction 清除 flag；
4. Agent 完成或终止 task 时，在 task terminal update 中清除 flag；
5. 进程在模型返回后、下一 step 落库前崩溃，flag 仍为 1，重启后可安全重新 deliberation；
6. 进程在 pending step 落库后崩溃，flag 已清除，重启后只 reconcile run，不重复提交物理动作。

Service startup 应依次执行：

```text
reconcile_active_run_events()
        ↓
load active tasks where resume_required = 1
        ↓
per-session Agent.resume(startup_recovery)
```

这比仅对 startup reconcile 返回的 terminal event 调用 `_handle_skill_event()` 更完整，因为它也覆盖上一次进程已落库的 terminal step。

Coordinator 应返回比 `AgentTaskStep | None` 更明确的归约结果：

```python
@dataclass(frozen=True)
class AppliedSkillEvent:
    step: AgentTaskStep
    task_status: TaskStatus
    should_resume: bool
    final_text: str | None = None
```

这可以修正当前 `environment_done` 的通知窗口：`TaskCoordinator.apply()` 会先 `complete_from_environment()`，而 `_handle_skill_event()` 重新读取 task 后因其已不是 active 而直接返回。新结果应表达“任务已由环境完成，不需要 resume Agent，但 Service 必须发布 final result”。

### 14.7 steer、pause、cancel 和 emergency stop

当前实现用 typed protocol 稳定区分 prompt、steer 和 stop，不依赖 LLM 猜测 emergency stop：

```python
@dataclass(frozen=True)
class ConversationTurn:
    ...
    kind: Literal["prompt", "steer"] = "prompt"

@dataclass(frozen=True)
class AgentControl:
    envelope: Envelope
    session_key: str
    interaction_id: str
    action: Literal["pause", "resume", "cancel", "emergency_stop"]
    reason: str = ""
```

`AgentControl` 使用独立 topic 和 Service handler，不经过 `AgentRunner`：

| 输入 | 当前状态 | 行为 |
|---|---|---|
| steer | 正在 model inference | 持久化 guidance，取消当前 inference task，用新 context 重启 |
| steer | physical Skill pending/running | 持久化 guidance，不默认中断动作，在 terminal safe point 生效 |
| pause | physical Skill pending/running | 对支持取消的 run 发 cancel，task 进入 paused |
| cancel | 任意 active task | cancel active runs，task 进入 cancelled |
| emergency stop | 任意状态 | 直接 `SkillClient.emergency_stop()`，再收敛 task state |
| resume | paused task | 确认没有 active run，再从最近 durable outcome `Agent.resume()` |

为了保留用户修正，`AgentTaskStore` 可增加小型 `task_amendments` 表：

```text
task_id, sequence, text, created_at
```

root objective 保持不变，context 和 completion 使用“root objective + ordered amendments”。不需要引入 plan patch 或复杂 steering state machine。

Patch 4 还需要将 `TaskStatus` 增加 `paused`，并将“一个 session 最多一个未结束 task”的唯一索引从仅限 `active` 改为覆盖 `active/paused`。Store 需要区分：

```text
current_task(session)     -> active 或 paused
runnable_task(session)    -> 仅 active
pause_task(task_id)       -> active -> paused
resume_task(task_id)      -> paused -> active
```

否则 paused task 会被 `active_task()` 当作不存在，同一 session 可能错误创建第二个任务。

### 14.8 budget 的具体简化

重构前同时有：

- `_MAX_STEPS_PER_SLICE = 8`；
- `hard_max_continuations`；
- `hard_max_skills`；
- `hard_max_wall_time_sec`。

在改成“每个物理 Skill 后自然 yield，terminal event 后 resume”之后，slice 和 continuation 不再是物理 long-horizon 的必要概念。当前任务级状态只保留：

```text
max_steps       # durable physical step 上限
deadline_at     # wall-clock 截止时间
```

所有 inline model/tool feedback 共用一个进程内 `MAX_MODEL_TURNS_PER_WAKEUP=8` 防御值。它不是 task 持久状态；达到上限时 active task 转为可恢复的 `paused`，而不是不可恢复地终结。

已完成的迁移：

- `hard_max_skills` 映射到 `max_steps`；
- `hard_max_wall_time_sec` 继续用于创建 `deadline_at`；
- 删除 `hard_max_continuations`、`continuation_count` 领域字段与读写逻辑；
- 旧数据库中的额外列原位保留但不再消费，避免 SQLite 破坏性重建；
- 删除四个已解析但未消费的配置项。

### 14.9 completion 的当前最小语义

当前 baseline 只有两条完成路径：

```text
Skill environment_done ──→ 权威完成
模型 complete_task(recap) ──→ 至少一个 durable completed step ──→ 完成
```

Store 继续自动保存每一步 outcome 和 evidence，但模型不负责账本索引。该规则刻意很小，
不声称已经解决开放世界的 false completion。若评测出现可复现误完成，再按新路线文档
分别加入 post-action observation、evidence selection 或 verifier，并逐项做消融。

### 14.10 分阶段 patch 计划

#### Patch 0：锁定当前行为和已知缺口

不移动生产逻辑，先增加特征测试：

- terminal step 已落库但没有 active run 时，startup 必须 resume；
- replayed terminal event 不会重复启动 Agent；
- pending physical step 绝不会被当作任务完成；
- `environment_done` 直接完成后仍向用户发布 final result；
- prompt 不得引用 registry 中不存在的 tool name；
- 记录当前 82 项相关测试的 baseline。

#### Patch 1：行为等价抽取 ContextBuilder 和 ToolExecutor

- 新增 `runtime/agent_context.py`；
- 新增 `tools/executor.py`；
- 将 service 中的 context/dispatch/completion/control helper 迁入；
- 保持数据库 schema、bus payload、prompt 效果和完成规则不变；
- 将目前通过 `object.__new__(AutonomousAgentService)` 的单元测试迁到 executor/context 直接测试。

验收：Service 不再出现 `_dispatch_*`、completion verifier 调用或 observation gate 细节。

#### Patch 2：抽取 per-session Agent

- 新增 `runtime/agent.py`；
- 迁移 `_run_conversation_loop()` / `_run_task_step()`；
- Service 管理 `dict[session_key, Agent]`；
- Service 仅负责输入路由和输出发布。

验收：`Agent` 不 import bus、`create_bus_client`、Skill transport topic 或 OpenAI 具体 SDK；Service 不直接调用 `AgentRunner.run()`。

#### Patch 3：将 physical wait/resume 变成 durable wakeup

- 增加 `resume_required` / `resume_after_sequence` migration；
- terminal event 和 wakeup flag 同 transaction 更新；
- startup 查询全部 resumable active task；
- 下一 pending step 或 task terminal 与 flag clear 同 transaction；
- 增加 event/resume 竞争和 crash-window 测试。

验收：在 terminal SkillEvent 之后、下一轮模型请求之前杀掉进程，重启后 task 继续，且旧物理 action 不被重放。

#### Patch 4：加入真正交互语义

- protocol 增加 typed steer 和 `AgentControl`；
- 增加 task amendments 持久化；
- inference 期 steer 取消并重启当前 Agent run；
- physical run 期 steer 排队到安全点；
- pause/cancel/emergency stop 走独立 control handler；
- resume 从最近 durable outcome 继续，不规定下一 Skill。

验收：运动中改变目标不会并发发出第二个物理 Skill；emergency stop 在模型超时时仍可用。

实现中 physical-running steer 只持久化 amendment 并返回 waiting；`TaskCoordinator` 同时
拒绝同一 task 的第二个 active run。仍在取消中的旧 run 会使 task 保持 paused；恢复后
是否先观察由 Agent 根据最近结果和工具契约决定，Service 不知道具体 Skill 名。

#### Patch 5：在数据支持下删减 policy

该 patch 已在 Patch 0–4 稳定并通过完整回归后实施：

- 用 `max_steps + deadline` 取代 continuation/slice budget；
- 删除无人消费的 `requires_reobservation` 专用 contract；
- 删除 duplicate observation gate，由 observation Skill 的结构化 outcome 表达变化；
- `environment_done` 保持权威；删除 completion verifier 和手工 evidence ID；
- 删除已证明无消费者的配置和 trace export；
- 删除 `proposal()` 等已完成迁移的 compatibility alias。

实现验收包括 compatibility surface 不再出现、单次唤醒达到 turn bound 后任务仍可
resume，以及 Patch 0–4 的交互和 crash recovery 回归。线上 task success、false
completion、user intervention latency 和平均 model calls/step 用于决定下一项增量。

### 14.11 建议新增的关键测试

| 测试 | 证明的不变量 |
|---|---|
| `test_agent_returns_waiting_after_physical_submit` | Agent 不占用 loop 等待机器人 |
| `test_agent_resumes_from_terminal_tool_pair` | terminal outcome 使用正确 tool role 续接 |
| `test_startup_resumes_terminal_undeliberated_step` | 修复当前 crash window |
| `test_startup_never_resubmits_pending_physical_step` | durable receipt 防止动作重放 |
| `test_replayed_terminal_event_does_not_double_resume` | event sequence 与 wakeup 幂等 |
| `test_environment_done_publishes_final_without_agent_resume` | 环境权威完成不丢失用户通知 |
| `test_steer_during_inference_rebuilds_context` | 交互可修正当前决策 |
| `test_steer_during_skill_waits_for_safe_point` | 不产生并发物理动作 |
| `test_emergency_stop_bypasses_agent_runner` | stop 不依赖 LLM |
| `test_user_resume_continues_without_hard_coded_observation` | core 不依赖具体 observation Skill 名 |
| `test_environment_done_publishes_final_without_agent_resume` | 权威 predicate 优先 |

上述关键语义现由 `test_steer_during_inference_rebuilds_context`、
`test_steer_during_skill_waits_for_safe_point`、
`test_user_resume_continues_without_hard_coded_observation`、
`test_coordinator_rejects_second_concurrent_skill_run` 和
`test_startup_resumes_terminal_undeliberated_step` 等回归测试覆盖。

### 14.12 明确不在这次重构中做的事

- 不替换 `ModelClientLike`，不引入 provider framework；
- 不让 OpenAI Agents SDK 接管 loop/session/tool execution；
- 不将 Pi 的 parallel tools 带入物理 Skill；
- 不重新定义 Skill/VLA 执行层；
- 不在架构迁移同时加 memory、reflection、sub-agent 或 self-improvement；
- 不一次合并两个 SQLite 数据库；
- 不在 baseline 中加入 completion verifier、手工 evidence selection 或强制 reobservation；
  它们进入独立消融路线。

重构完成后，主路径应只剩：

```text
ConversationTurn / ResumeTrigger
              ↓
        per-session Agent
              ↓
          AgentRunner
              ↓
      one PreparedToolCall
              ↓
       AgentToolExecutor
              ↓
  ToolOutcome / waiting receipt
              ↓
       continue or yield
```

Service 不再知道 completion evidence 怎样审计、为什么要 reobserve，也不直接驱动 model loop；Agent 不知道 bus topic 和 OpenAI SDK；`AgentRunner` 不知道 task 和 robot；`TaskCoordinator` 不知道 prompt 和 conversation。

### 14.13 同期完成的死代码与兼容层清理

全仓生产入口审计后，额外删除了以下不属于 baseline 的表面：

- 零生产消费者的 `motion` 与 `notifications` 包；
- 从未接入 Agent/Executor 的 `foundation.catalog` tool policy/resolver 框架；
- 仅导出、从未注入的 `NullEventPublisher` 和 `EpisodeStore` 抽象；
- Web 的旧页面重定向、音频旧字段 alias 和未调用的 RoboCasa action helper；
- `AgentTaskStore` 的历史 SQLite `ALTER TABLE` 分支，当前字段直接由唯一 schema 声明。

这次清理没有删除真实入口：RoboCasa manifest loader 被 evaluation harness 消费；
episode/event/health/human-follow/classic runtime 均有生产调用者。旧版 task SQLite 文件若需
沿用，应在部署升级前显式迁移；baseline runtime 不再内置历史 schema 兼容逻辑。

## 15. 总结

Pi 的优秀之处是将通用 agent 归约为一个简单、稳定、可组合的运行协议：

```text
message → model → tool → result → next turn
```

然后把 streaming、steering、follow-up、abort、persistence、context transform 和 compaction 放到清晰的外围层。

Hey Robot 当前的核心优势则是机器人原生的 long-horizon 可靠性：

```text
durable task → bounded Skill → authoritative result
     ↑                                │
     └──── observation/evidence ──────┘
```

未来如果演进 cognition，最值得追求的不是减少机器人任务语义，而是让通用 agent lifecycle 与机器人 task harness 解耦：

> 用 Pi 风格的简单 core 提升通用性和交互性，用 Hey Robot 现有的 durable task/evidence/control 机制保证物理 long-horizon 的正确性。

四篇论文的价值应体现在帮助排序，而不是帮助堆叠功能：先完成可交互、可停止、单步闭环、可恢复的最小系统；然后让真实 failure 和评测数据决定下一个功能。
