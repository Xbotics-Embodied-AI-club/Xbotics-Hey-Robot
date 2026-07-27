# Hey Robot 最小 Embodied Agent Harness 开发指南

> 分析日期：2026-07-26
> 文档定位：后续系统设计、开发、验证和功能准入的长期指南
> 设计依据：四篇 Harness/VLA 材料、pi-agent-core、RPent 及 Hey Robot 当前代码
> 目标约束：当前不增加新功能；先保持简单、通用、最小，并验证现有模块
> 核心目标：交互能力、long-horizon task、配置驱动的 Embodied Agent Harness、边界清晰
> 的快慢双系统
> 对外定位：Embodied Agent Harness · Fast–Slow Dual System · Distributed Model Services

本文不是对当前已交付能力的声明。代码、配置和测试仍是运行事实源；本指南用于约束后续
取舍，并应随着验证结论更新。

## 1. 结论先行
重新结合四篇论文/博客、`pi-agent-core` 和 RPent 后，结论与“继续补齐 Harness VLA
功能”不同：

**Hey Robot 现在最重要的工作不是增加 memory、verifier、primitive、planner 或
self-improvement，而是冻结一条最小主链，证明它在交互、长程任务、失败恢复和物理动作
安全上成立。**

六份材料最有价值的地方，是帮助判断哪些能力属于 Harness 的不可缺少内核，哪些只是
实验系统为了提升 benchmark 成绩增加的策略。对当前 Hey Robot：

1. `pi-agent-core` 最值得参考的是小而清楚的 agent loop、steer/follow-up 语义，以及只在
   模型边界变换上下文；不应照搬它仍在增长的完整 `AgentHarness`。
2. RPent 最值得参考的是 `Planner + Toolkit + 每次动作后的新观测` 这条直接实验链；
   不应复制它约 2,000 行 LIBERO primitives、大型任务提示词、多 planner adapter 和
   benchmark-specific memory。
3. 四篇材料应当成为未来实验的“候选假设库”，而不是当前产品架构的需求清单。
4. Hey Robot 现有分层已经足够表达最小 Harness。此时再引入第二套 agent runtime、
   plan graph、通用 memory、通用 verifier 或更多事件抽象，只会增加尚未验证的状态空间。
5. VLA/VLN、RoboCasa、Human Follow、Voice、Feishu、仿真和真机驱动都可以保留，但应放在
   核心之外逐层验收，不能反向定义核心。InternNav 仿真接入和 RoboCasa365 LeRobot policy
   链路已经提供了部分集成证据；这不等同于 XLeRobot 真机或开放世界长程能力已验证。
6. 系统应继续由一份 deployment config 选择模块和实现，但配置只能驱动组装、能力暴露与
   有界参数，不能成为运行状态、隐藏控制流或另一种编程语言。

建议当前路线只有三个动词：

```text
冻结 Golden Path  →  逐层验证  →  删除或隔离重复复杂度
```

## 2. “最小系统”要证明什么

### 2.1 交互能力

最小交互不是“支持尽可能多的 Channel”，而是用一个 Channel 证明：

- 用户输入能够进入唯一 Agent；
- Agent 执行期间能够接收纠正，而不是必须取消整个任务；
- 当前动作结束后，纠正能影响下一轮决策；
- 用户能看到必要的进度、结果和失败原因；
- 重启后会话和任务身份仍可关联。

Web、CLI、Voice、Feishu 都是入口适配器。只需一条入口通过上述验收，就能验证交互内核；
其余入口不应进入第一阶段的必要依赖。

### 2.2 Long-horizon task 能力

这里的 long-horizon 不等于引入 plan tree、自动分解 DSL 或长期记忆。最小定义是：

- 一个用户目标能跨越多次 Agent turn 和多次 Skill 执行；
- 系统持久记录已完成、执行中、失败和待继续的事实；
- 进程重启后可继续推理，但不会盲目重放不确定的物理动作；
- 中途纠正、Skill 失败和超时不会破坏任务身份；
- 任务只有在有明确依据时结束。

换言之，long-horizon 首先是**任务连续性和恢复语义**，其次才是更聪明的规划。

### 2.3 快慢双系统

当前目标只需要一条稳定边界：

```text
用户 / 环境事件
        ↓
慢系统：Agent 决策、选择 Skill、解释结果
        ↓
唯一物理提案边界
        ↓
快系统：Skill / RobotRuntime 有界执行
        ↓
结构化结果与新观测
        └────────────→ 慢系统继续决策
```

慢系统不应直接操纵机器人驱动；快系统不应自行扩展开放目标。两者之间的 contract 比
再增加一种 planner 或 policy 更重要。

### 2.4 配置驱动与边界清晰

“配置驱动”在这里有严格含义：

- 同一套代码通过 deployment config 选择 Channel、Robot、Model Service、Skill surface、
  bus 和部署 profile；
- composition root 读取已校验的 typed config，创建具体实现并注入依赖；
- 模块内部只依赖小型 protocol/contract，不读取其他模块的配置片段来绕过组装；
- 配置决定“启用什么、连接谁、预算多少”，代码决定状态机、安全规则和业务不变量；
- 运行中的 task、conversation、run 和 robot state 进入各自 store，绝不回写配置。

模块边界清晰则意味着：

```text
config → composition root → interface/protocol → module implementation
                                  ↓
                         explicit result / event
```

不允许出现：

- Channel 直接调用 RobotRuntime；
- Agent 直接构造硬件动作协议；
- Skill 绕过 `LocalRobotClient` 访问 driver；
- adapter 直接修改 Agent task store；
- 某个模块通过全局 singleton 或读取任意 YAML 路径取得隐藏依赖；
- 为不同 profile 复制一套业务主链。

当前 `DeploymentConfig`、`SkillSurfaceConfig`、`create_bus_client()`、
`build_local_runtime_components()` 和 `AutonomousAgentService` 已经构成配置驱动组装的基础。
下一步应验证并收紧这条边界，而不是再建立一套配置框架。

## 3. 统一评价标准

本文不再问“这个功能先进吗”，而用六个问题评价每项参考：

1. 没有它，交互或 long-horizon 主链是否不能成立？
2. 它是否跨机器人、模型和任务通用？
3. 当前能否用确定性测试或小型端到端场景验证？
4. 它会新增多少持久状态、并发状态和失败状态？
5. 能否在不破坏未来扩展的前提下延后？
6. 它能否作为配置选择的边界实现接入，而不让核心依赖具体 backend？

只有同时满足“必要、通用、可验证”的能力，才应进入当前核心。

## 4. pi-agent-core：参考小循环，不复制大 Harness

参考代码：`/home/liber/embodied_agent/pi/packages/agent`

### 4.1 小核心为什么有价值

pi 的基础循环可以概括为：

```text
pending messages
  → model response
  → tool calls / tool results
  → steering messages
  → follow-up messages
  → end
```

关键位置包括：

- `src/agent-loop.ts` 的 `runLoop()`：模型和工具的循环主体；
- `src/agent.ts` 的 `PendingMessageQueue`：区分执行中纠正与执行后追加；
- `Agent.steer()` / `Agent.followUp()`：两种交互时序；
- `createContextSnapshot()`：每轮使用一致的运行快照；
- `processEvents()`：用事件观察生命周期，而不是让事件成为第二套控制流；
- `transformContext`：只在提交给模型前变换上下文，不篡改 canonical history。

这对 Hey Robot 的启示不是改成 TypeScript 或复刻 API，而是保住四条原则：

1. Agent loop 应能用一页流程解释；
2. canonical conversation 与模型输入投影分开；
3. steering 是队列语义，不是另起一个 Agent；
4. event 用于观测和集成，核心状态仍由唯一运行时收敛。

### 4.2 完整 AgentHarness 是复杂度警示

pi 的 `src/harness/agent-harness.ts` 已超过基础 loop 的规模，加入 model registry、
session persistence、hooks、recovery、工具策略等职责。其文档也明确把 lifecycle
hardening、model registry、semi-durable recovery 等列为仍在推进的工作。

`docs/durable-harness.md` 进一步指出，包含 stream、tool closure 和 runtime object 的
“完全 durable”执行并不现实；更实际的是 semi-durable recovery。未完成的工具调用如果
不具备幂等性，也不能安全自动重试。

这支持 Hey Robot 当前较保守的恢复方向：

- 对话和任务事实可以恢复；
- 未知结果的物理执行标为 `execution_lost`；
- 重启后交给 Agent 重新判断；
- 不承诺从任意 token 或任意机械动作中点继续。

因此，pi 提供的是一个**复杂度上限警告**：Hey Robot 不需要为了叫作 Harness，就把
所有 session、model、hook、recovery 能力集中到一个更大的抽象中。

### 4.3 取舍

| pi 能力 | 当前决定 | 原因 |
| --- | --- | --- |
| 简单 model/tool loop | 保留原则 | 是慢系统最小内核 |
| steer / follow-up 区分 | 验证现有等价语义 | 直接支撑交互 |
| 模型边界 context transform | 保留原则 | 避免多份真相 |
| 生命周期事件 | 只保留现有必要事件 | 不再增加 event framework |
| full durable continuation | 不追求 | 物理工具无法普遍安全重放 |
| 完整 AgentHarness | 不迁移 | 职责和状态空间过大 |
| hooks、model registry 扩张 | 延后 | 不影响当前两项核心能力 |

## 5. RPent：参考直接实验闭环，不复制领域复杂度

参考代码：`/home/liber/embodied_agent/RPent`

### 5.1 值得保留的最小形状

RPent 的通用边界并不复杂：

- `rpent/planner/base.py`：小型 `Planner` protocol；
- `rpent/tools/toolkit.py`：工具注册与 `execute_tool()`；
- `robots/libero/toolkit.py`：执行 primitive 后立即导出新状态、图像和视频；
- planner loop：限制 turn 数、接受 steering、裁剪旧图像上下文、保存 transcript。

最值得参考的是“每次物理动作后都给 planner 新鲜反馈”。对具身系统而言，陈旧观测会让
再好的 planner 也基于错误世界状态工作。

RPent 还把每次实验的 transcript、recipe、state、image 和 video 落盘。这种 artifact
优先的实验方法适合 Hey Robot 后续验证，但不需要先变成新的运行时服务。

### 5.2 RPent 不是最小通用 Harness 模板

RPent 的简单是“单次 benchmark run”的简单，不等同于长期在线服务的简单。每次 run
可启动环境和 VLA daemon，执行任务后保存结果并结束；它无需同时解决长期会话、服务
重启、多个入口、物理资源互斥和未知执行结果恢复。

同时，它把大量复杂度下沉到了 LIBERO 专用层：

- `robots/libero/tools.py` 约 2,000 行，包含 pick、move、状态导出、回投影等大量原语；
- `robots/libero/prompts/system.py` 包含 single-attempt 规则、seed/task tactics、
  fixture 行为和具体 offset；
- guides 与 `MEMORY.md` 提供 benchmark-specific 经验；
- API、Claude Code、Codex 等多种 planner adapter 服务于实验对比。

这些能力可能提高 LIBERO 成绩，但不是 Hey Robot 核心的通用性证据。把它们复制进主链，
只会把 benchmark 偶然性变成产品架构。

### 5.3 RPent memory 的真实含义

RPent 文档把 memory 定义为经过审阅、每次 run 开始时读取的只读知识库，并且项目本身
没有 agent 自助上传通道。因此它证明的是：

> 经审核的离线经验可能有用。

它没有证明：

> 在线 Agent 应立即拥有自动写入、召回和自我修改的通用 memory service。

后一种能力会引入 provenance、污染、版本、淘汰、权限和回归评估等新问题，不属于当前
最小核心。

### 5.4 取舍

| RPent 能力 | 当前决定 | 原因 |
| --- | --- | --- |
| Planner / Toolkit 小 contract | 作为边界校验参考 | 与慢/快系统接口同向 |
| 动作后新观测 | 必须验证 | 是闭环控制基本条件 |
| transcript / state / media artifact | 复用现有能力并验收 | 有助于诊断 |
| turn/image budget | 保留原则 | 控制上下文和运行边界 |
| LIBERO 大型 primitive 集 | 不复制 | 领域和 benchmark 耦合 |
| 巨型 system prompt / task tactics | 不进入核心 | 不通用、难回归 |
| 多 planner adapter | 不增加 | 当前无需做 planner 横评 |
| 自动 memory 系统 | 不增加 | RPent 本身也不是在线自改进 |

## 6. 四篇材料在“保持最小”约束下的重新解释

### 6.1 Harness Engineering for Self-Improvement

当前最有用的不是 self-improvement，而是三个前置条件：

- workflow 可重复；
- artifact 可观察；
- evaluator 独立于被评对象。

如果现有模块尚未完成基线验证，让 Agent 自动改 prompt、工具或 Harness 只会让故障来源
继续增加。现在应吸收“先把系统变成可测对象”，延后候选生成、自动修改、sub-agent 和
自演化闭环。

### 6.2 Harness VLA

它提供的重要原则是：冻结的 VLA 应被当作有界、可观察、可退出的低层 option，而不是
整个系统的中心。高层可以在新观测后决定下一步。

但论文中的 memory-guided retry、re-stage 和解析 primitive 库属于性能增强。Hey Robot
当前只需证明：

- VLA option 有明确预算；
- 终止原因和成功语义不会混淆（当前仍需修复并回归验证 `no_action`、`max_steps`）；
- SkillResult 已保存 option 后的观测引用，但这些观测是否完整投影到下一轮 Agent context
  仍需验证；
- VLA 失败不会破坏 Agent task。

不应为了对齐论文而立即增加通用 verifier、重试策略或 primitive catalog。

### 6.3 Hi Robot

最相关的是两点：

- 开放语言目标由高层转换为低层可执行目标；
- 用户能在任务中途纠正系统。

Hey Robot 已有实现这些语义所需的 Agent、task、Skill 和 Channel 分层。当前应先用
deterministic Skill 和 Mock Robot 验证纠正是否真正影响后续动作，再讨论高频 replanning、
新模型训练或 synthetic instruction 数据。

### 6.4 Hi-VLA

Hi-VLA 对 planner、VLA、termination、observation、memory 的系统比较，很适合形成未来
实验变量；它不意味着这些变量都要成为生产 abstraction。

正确使用方式是：

1. 保持 production path 不变；
2. 一次只替换一个实验变量；
3. 用同一任务集比较成功率、误完成、恢复和延迟；
4. 有稳定收益后再决定是否进入核心。

## 7. Hey Robot 当前代码：核心已经够用

当前主链已有以下职责：

| 层 | 当前组件 | 在最小系统中的职责 |
| --- | --- | --- |
| 入口 | `Channel` / `ChannelManager` | 输入输出适配 |
| Agent 调度 | `AgentRunner` | 唯一 Agent 运行入口 |
| 慢系统 | `Agent` | 模型循环与 Skill 选择 |
| 对话事实 | `ConversationStore` | canonical conversation |
| 长程任务 | `AgentTaskStore` | 跨 turn 的 task/step 事实 |
| 协调 | `TaskCoordinator` | Skill 事件推动任务继续 |
| 动作边界 | cognition tool executor | 把工具调用变成受控 Skill |
| 快系统 | `Skill` / `SkillWorker` | 有界执行和执行状态 |
| 机器人边界 | `LocalRobotClient` | 唯一本地机器人调用面 |
| 运行时 | `RobotRuntime` | driver、安全和资源执行 |
| 验证替身 | Mock driver / deterministic skills | 无硬件验证语义 |
| 配置模型 | `DeploymentConfig` / typed specs | 唯一部署选择和参数入口 |
| 组装边界 | app composition roots / factories | 把配置解析与业务执行隔离 |

这些组件已经足够表达：

```text
一个用户 → 一个 Agent → 一个任务事实源
         → 一次至多一个物理动作提案 → 一个 RobotRuntime
         → 一个结果 → 同一个 Agent 继续
```

当前没有证据表明需要第二个 Agent、第二种 task engine 或第二条物理执行路径。

需要把以下能力标记为 `partial`，而不是已验证：

- `AutonomousAgentService` 虽然向 Agent 传递了文本增量回调，但 Agent 驱动层当前没有继续
  发布该回调，因此流式回复尚未在完整 Agent 路径打通；
- `agent_runtime.enabled` 是配置模型字段，但当前服务启停实际由
  `agents.<id>.enabled` 决定；
- VLA 的 `no_action` / `max_steps` 可能仍被任务完成逻辑视为成功步骤。

## 8. 核心与可选能力的边界

### 8.1 当前核心

核心只包含回答以下问题所需的组件：

- 用户消息如何进入唯一 Agent？
- Agent 如何调用一个受控 Skill？
- Skill 的状态和结果如何成为持久事实？
- 新观测如何返回 Agent？
- 用户如何在执行中纠正？
- 进程崩溃后如何安全恢复？

用户回复与持续任务状态必须通过一个原子的结构化 contract 返回。普通问答明确表达
`none`，等待确认、完成和取消明确表达相应状态；普通模型文本既不能隐式创建任务，也
不能隐式结束任务。SYSTEM prompt 只声明“必须结构化回复”这一运行不变量，具体工具名、
字段和值域由 function schema 自己拥有，避免策略层依赖某个工具实现。

### 8.2 应隔离验证的可选能力

以下模块可以继续存在，但不能成为 Golden Path 的启动前提：

- Voice、Feishu、CLI 等额外 Channel；
- Human Follow；
- MuJoCo、RoboCasa 和其他 simulator；
- 真实机器人 driver；
- VLA、VLN、captioner 和模型 sidecar；
- NATS 或其他分布式部署设施；
- dashboard、media UI 和实验工具。

“可选”不是否定价值，而是要求它们通过 adapter 接入；关闭后，核心交互和长程任务测试
仍应运行。

### 8.3 配置的职责边界

| 配置应该负责 | 配置不应该负责 |
| --- | --- |
| 启用/关闭模块 | 保存任务进度或机器人状态 |
| 选择符合相同 contract 的实现 | 描述跨模块运行工作流 |
| 声明 Agent 可见的 Skill surface | 动态生成新的核心抽象 |
| 设置路径、endpoint、timeout、budget | 用条件表达式实现业务状态机 |
| 在启动时完成引用和兼容性校验 | 让模块自行查找未声明依赖 |

配置层自身也要保持最小：

1. 一项概念只保留一个规范字段，旧字段直接拒绝而不是永久兼容；
2. 所有引用在启动前 fail fast，例如 Agent、Robot、Model Service 和 Skill implementation；
3. profile 只组合已有模块，不 fork 业务语义；
4. secret 通过环境注入，但环境变量不能变成第二套配置模型；
5. `settings: dict` 只用于 backend-specific 叶子参数，稳定的跨实现字段应逐步 typed；
6. 文档中的“已配置”“已注册”“Agent 可见”“已验证”必须明确区分。

### 8.4 模块 contract 的最低要求

每个可替换模块都应有：

- 单一职责和明确 owner；
- 小型输入/输出类型，而不是共享任意字典；
- 明确 timeout、cancel、error 和 recovery 语义；
- 不泄漏 backend-specific 类型到上层；
- 一个 contract test，至少包含正常、失败、超时和关闭四条路径；
- 一个由配置替换实现的测试，证明 Mock 与真实/仿真实现使用同一上层主链。

边界是否清晰，不以目录数量判断，而以“替换某个实现时需要改动哪些上层代码”判断。
理想结果是只改配置；如果必须修改 composition root，也不应修改 Agent、task 或 Skill
状态机。

### 8.5 当前不应新增

- 第二套 Agent runtime 或 planner runtime；
- plan graph、task tree 或工作流 DSL；
- 通用长期 memory service；
- 通用 completion verifier framework；
- sub-agent orchestration；
- 在线 self-improvement；
- 自动 skill discovery；
- remote Skill 协议；
- 另一套 event bus 或观察抽象；
- 仅为某一 benchmark 设计的大型 primitive 层。

## 9. 两个实现项目与 Hey Robot 的正确关系

| 维度 | pi-agent-core | RPent | Hey Robot 应选择 |
| --- | --- | --- | --- |
| 主要场景 | 通用 model/tool agent | 单次机器人 benchmark | 长期在线具身 Agent |
| 最小循环 | 很清楚 | Planner/Toolkit 很直接 | 保持唯一 Agent/Skill 主链 |
| 交互纠正 | steer/follow-up | input queue | 验证现有纠正链 |
| 持久性 | 正在走 semi-durable | run artifact 为主 | 任务事实持久、执行保守恢复 |
| 物理动作 | 普通 tool，可并行 | LIBERO primitive | 单物理提案、RobotRuntime 串行约束 |
| 观测 | message context | 每步新状态/图像 | 结果必须携带新鲜观测引用 |
| 领域复杂度 | harness 逐渐增大 | primitives/prompt 很大 | 保持在 adapter/实验 profile 外围 |
| 当前借鉴方式 | 学小核心和边界 | 学直接实验和 artifact | 不迁移任一完整框架 |

特别需要注意：pi 对普通软件工具的并行执行不适合直接复制到物理 Skill。机器人动作会
共享空间、底盘、机械臂和安全状态，Hey Robot 保持单一物理提案边界更稳妥。

## 10. 当前最应该做的验证阶梯

### 10.1 先冻结一个 Golden Path

建议唯一基线配置（目标形状，当前尚未作为正式 deterministic profile 完整验收）：

```text
一个 Web Channel
  + in-process AgentRunner
  + 一个 Agent
  + SQLite / 当前 FileRunStore
  + deterministic skills
  + LocalRobotClient
  + Mock RobotRuntime
```

该基线最终必须是一份正常的 deployment profile，而不是测试代码中手工拼出的另一套系统。
当前 deterministic model/Skill 主要通过测试注入；在正式 profile 建立前，不能把 Golden
Path 写成已交付能力。测试仍应经过与生产一致的 typed config、registry、factory 和
composition root。这样才能真正验证“配置驱动”，而不仅是验证若干孤立类。

Golden Path 只需要一个三步任务：

1. 用户提交需要三个 Skill 才能完成的目标；
2. 第二步执行期间用户发送纠正；
3. 纠正影响第三步；
4. 在另一轮测试中于第二步制造进程中断；
5. 重启后保留任务事实，把不确定执行标记为 lost，且不自动重放；
6. Agent 基于事实继续或向用户确认。

它同时覆盖交互、long-horizon、快慢系统边界和恢复，是当前最高价值的系统测试。

### 10.2 分层验收

| 层级 | 验收对象 | 通过标准 |
| --- | --- | --- |
| L0 | 静态架构与配置 | 单 Agent、单事实源、单物理主链；profile 引用合法，无 legacy surface |
| L1 | 模块单测 | store、queue、Skill 状态机、RobotRuntime 各自确定性通过 |
| L2 | 同进程 Golden Path | 三步任务、steer、失败回流完整 |
| L3 | crash/recovery | 无物理动作盲重放，lost 状态可解释并可继续 |
| L4 | 一个 simulator adapter | 只改配置/叶子组装即可替换 Mock，artifact 足够复现 |
| L5 | 一个 real driver | 只替换配置实现，安全和失败语义与 Mock contract 一致 |
| L6 | VLA/VLN/RoboCasa 等模型与评测能力 | 已有部分集成验证；逐项启用，失败不污染核心结论 |

在 L0–L3 没有稳定通过以前，不应以 benchmark 新功能替代基础验证。

### 10.3 每个模块都要有一张验证记录

每个现有模块至少记录：

- 它解决的唯一问题；
- 是否属于 Golden Path；
- 由哪个配置字段启用、默认是否关闭；
- 输入、输出和持久状态；
- 它依赖和实现的 contract；
- 超时、取消、崩溃时的语义；
- 最小单测和集成测试；
- 关闭后哪些能力受影响；
- 与其他模块是否有重复职责；
- 当前结论：verified / partial / experimental / unused。

这比继续写一份更大的架构设计更能暴露真实复杂度。

## 11. 简化策略：先隔离，再删除

仓库已经复杂，直接大规模删除也会制造风险。建议按以下顺序收敛：

1. 固定一个 composition root 和 Golden Path 配置；
2. 让可选模块默认不启动、不被核心 import；
3. 用测试证明关闭它们后 L0–L3 仍通过；
4. 标记重复、无入口或无测试模块；
5. 连续一段验证周期无使用证据后再删除。

最终应始终能回答四个“唯一”：

- 唯一 Agent 在哪里？
- 唯一 canonical task state 在哪里？
- 唯一 Skill dispatch 在哪里？
- 唯一物理动作路径在哪里？

如果一个新设计让答案变成两个，就需要非常强的证据。

## 12. VLA 与 RPent 应如何保留为未来实验

Hey Robot 当前 VLA option 已有 bounded steps，termination policy 会把 `no_action`、
`max_steps` 的 `subgoal_succeeded` 表示为 unknown；但 option 结果当前仍可能以
`success=True` 进入任务完成判定。这部分应先修正并做 contract 回归测试，确认：

- option 正常结束不等于物理子目标成功；
- unknown 不会被上层转换为 task success；
- 执行后的 observation 引用不仅被保存，而且能按约定进入下一轮 Agent context；
- crash 不会触发隐式动作重试。

如果发现语义缺陷，修复现有 contract 属于验证和正确性工作，不是扩功能。但无需因此
立即加入一个通用 verifier 系统。

RPent 可以作为 L6 的实验参照：

- 在独立 profile 中对比 VLA-only 与 planner+primitive；
- 复用它的 per-step artifact 思路；
- 一次只引入少量、任务无关的 primitive；
- 不把 RPent 变成运行时依赖；
- 不把 LIBERO prompt 和 memory 合并进 production prompt。

## 13. 新功能准入门槛

未来每个功能进入核心前，至少应回答：

1. 它解决了哪个已复现的失败？
2. 该失败是否阻断交互或 long-horizon？
3. 能否通过修正现有 contract 解决？
4. 能否只存在于 adapter 或实验 profile？
5. 新增了哪些持久状态和并发状态？
6. crash、cancel、timeout 和 retry 语义是什么？
7. 是否产生第二份 canonical truth？
8. 是否有 deterministic test？
9. 是否在 held-out 任务上有稳定收益？
10. 如果删除它，系统会失去什么核心能力？
11. 它通过哪个现有 contract 和配置字段接入？
12. 是否迫使上层理解 backend-specific 类型或分支？

无法回答这些问题的功能，默认延后，而不是默认进入。

还应设置复杂度预算：

- 新增一个核心 abstraction，原则上应删除或合并一个旧 abstraction；
- 新增一种状态，必须定义恢复和观测方式；
- 新增一种后台服务，必须证明 in-process Golden Path 不足；
- 新增一个模型，必须能在 Mock/deterministic 基线上单独评价增益。
- 新增一种 backend，优先只增加 leaf implementation、typed config 和 contract test；
- 新增配置字段，必须有校验、默认值、文档和无效组合测试。

## 14. 六份材料的最终采纳表

| 来源 | 现在采纳 | 暂不采纳 |
| --- | --- | --- |
| Harness Engineering | 可重复 workflow、artifact、独立 evaluator | 在线自改、sub-agent、自动候选合入 |
| Harness VLA | bounded option、动作后观测、失败回到高层 | 通用 memory、retry/re-stage 框架、大原语库 |
| Hi Robot | 开放目标与中途纠正的接口目标 | 新训练流程、高频重规划要求 |
| Hi-VLA | 把变量变成隔离实验和消融 | 把所有变量产品化 |
| pi-agent-core | 小 agent loop、steer/follow-up、context 边界 | 完整 AgentHarness 迁移、full durability |
| RPent | 小 Planner/Toolkit contract、fresh observation、artifact | LIBERO 专用复杂度、多 adapter、巨型 prompt/memory |

## 15. 最终建议

Hey Robot 不缺一张更宏大的路线图，缺的是对当前系统的可信证据。接下来应停止横向扩展，
用一份正式配置驱动 Mock + deterministic Skill 完成 L0–L3；在该 profile 和完整三步
场景验收前，Golden Path 应视为待验证目标，而不是当前已交付能力。

只有三类工作应进入当前主线：

1. 修复阻断 Golden Path 的现有 bug；
2. 删除或隔离重复职责和非必要启动依赖；
3. 增加验证现有 contract 的测试、artifact 和文档。

四篇材料和两个项目仍然很有参考价值，但应放在不同位置：

- pi-agent-core 是“慢系统最小到什么程度”的参照；
- RPent 是“机器人实验怎样形成直接闭环”的参照；
- 四篇材料是未来性能实验的假设库；
- Hey Robot 自己的核心，则应保持一个 Agent、一套任务事实、一条 Skill 主链和一个
  RobotRuntime 边界。

架构上再加两条长期不变量：

- 一份 typed deployment config 决定系统如何组装，但不承载运行逻辑；
- 每个可选模块只能通过稳定 contract 接入，替换 backend 不改变上层主链。

这条路线不会阻碍未来增加 VLA、memory、verifier 或 self-improvement。相反，只有先建立
简单且经过验证的基线，未来每项增加才有可能证明自己确实带来价值，而不是仅仅带来更多
代码。
