# Hey Robot：面向交互式长程机器人任务的 Embodied Agent Harness

<p align="center">
  <strong>Hey Robot: An Embodied Agent Harness for Interactive Long-Horizon Robot Tasks</strong>
</p>

> **系统论文草稿 · 2026-07-25**
>
> 本文以仓库当前代码为事实来源，描述已经实现的系统机制、仍处于实验阶段的能力，以及
> 后续评估协议。除非同时给出可复现 artifact、代码 commit、运行配置和统计结果，本文
> 不把接口接入、自动化测试或单次演示表述为已验证的真机任务性能。

## 摘要

机器人 Agent 不只是生成动作的模型，而是一个持续的闭环：它接收环境状态与用户反馈，
选择动作，通过物理世界获得结果，再据此修正后续行为。对于交互式长程任务，这个闭环
可能跨越多次模型决策、多个物理技能、用户中途干预和服务重启。通用 Tool Calling 循环
能够提出下一步，却不天然拥有物理任务的持久状态、动作唯一性、终止条件、资源约束和
故障恢复语义。

本文提出 **Hey Robot**，一个面向真实机器人、开放源代码的 **Embodied Agent
Harness**。系统采用快慢双系统：慢系统维护目标、交互、任务连续性和语义决策；快系统
在明确预算内执行感知、导航、操作或经典控制 Skill，并返回结构化结果与证据。二者不要求
常驻于不同进程，而是通过持久任务状态、唯一 Skill run 和终态事件形成可恢复的闭环。

Hey Robot 实现了单会话单未终止任务、persist-before-submit、Task step 与 Skill run
关联、单调事件归并、终态事件驱动续跑、启动 reconciliation、执行中 steer、确定性暂停/
取消/急停，以及 deployment 级 Skill allowlist。默认部署在一个 asyncio 进程中组合
Agent、Skill Worker、Robot Runtime 和 Gateway；计算密集的 VLA/VLN ModelService
可以通过 gRPC 独立部署。当前代码已覆盖 Mock、MuJoCo、XLeRobot driver、多通道交互和
持久任务主链；默认 XLeRobot profile 只开放场景检查与底盘移动，VLA/VLN 和复杂长程
真机任务仍属于实验能力。

本文当前贡献是系统抽象、开源实现与预注册式评估设计，而不是尚未获得的成功率结论。
评估将把模型或底层策略能力与 Harness 收益分开，分别测量任务连续性、用户纠偏、终止
准确性、崩溃恢复、重复物理动作风险和端到端任务进展。

## 1. 引言

语言模型、视觉语言模型和视觉语言动作模型正在缩短自然语言、视觉观测与机器人控制之间
的距离 [1–5]。层级系统进一步让高层模型承担目标分解和语义推理，让低层策略执行局部
动作，从而处理复杂指令和长程组合任务 [6–9]。

然而，模型能力并不等于可部署的 Agent 能力。真实系统还必须回答：

- 当前目标、已完成步骤和未决物理动作存储在哪里？
- 用户在机器人运动时发来的修正应立即打断，还是在下一个安全边界生效？
- 服务重启后，状态不明的动作应恢复、失败还是重放？
- 什么机制保证模型只能使用当前机器人与 deployment 明确允许的能力？
- 如何区分一次模型 decision、一次 Skill run 和整个任务生命周期？
- 如何在模型失效或误解指令时仍然执行暂停、取消和急停？
- 如何判断低层能力已经终止，并把可靠反馈返回给高层？

这些问题属于 **Harness**：围绕模型组织工作流、工具、上下文、持久状态、权限、反馈、
评估和可观测性的系统层 [10]。对机器人而言，Harness 还必须面对观测过期、动作不可随意
回滚、资源互斥和物理安全等约束。

Hey Robot 面向 XLeRobot、MuJoCo 和 RoboCasa 构建这一系统层。它不把 LLM 视为完整
机器人控制器，也不要求 VLA/VLN 一次覆盖整个任务。慢系统跨时间维持目标与交互，快系统
把一个局部目标落实为有界 Skill；Skill 的结果、终止原因、观测和证据再次进入慢系统，
构成完整的 Agent–environment 闭环。

### 1.1 研究问题

本文研究以下问题：

1. 如何使机器人任务跨越模型唤醒、物理动作和进程重启而保持一致？
2. 如何让用户在执行中追问、修正、暂停或终止任务，同时不直接扰动低层控制？
3. 如何约束模型提出的物理能力，并避免恢复过程重复执行状态不明的动作？
4. 如何让经典控制、VLA、VLN、仿真和真机共享稳定的 Agent-facing 能力接口？
5. 终止、观测、记忆与执行反馈如何进入长程闭环，而不是被隐藏在模型调用内部？
6. 如何把 Harness 机制收益与 planner、VLA/VLN 和机器人本体能力分开评估？

### 1.2 贡献

本文和当前开源实现提供：

1. **面向具身闭环的 Harness 抽象。** 将 Agent 表示为用户、环境、持久状态、慢速语义
   决策和快速有界执行共同构成的闭环，而不是单一模型或聊天角色。
2. **持久长程任务协议。** 使用 SQLite 保存目标、路由、step、Skill run、终态和恢复
   位置；每个 session 至多存在一个 active 或 paused 任务。
3. **事件驱动执行语义。** 物理 Skill 提交后结束当前模型控制流；终态事件更新任务
   step，并用结构化结果唤醒同一 session 的 Agent。
4. **物理动作一致性机制。** 在提交前持久化 pending step，以 run ID 和单调 sequence
   幂等归并事件；启动时查询状态，但不盲目重放所有权已丢失的物理命令。
5. **可交互且确定性的控制边界。** 执行中新输入被保留到下一个 Skill 边界；pause、
   resume、cancel、block 和 emergency stop 不依赖 LLM 解释。
6. **可替换的具身能力层。** Agent、Skill、Foundation Model、Robot Runtime 和 Driver
   具有明确所有权，物理动作只经过一条受安全门控的主执行链。
7. **可证伪的评估协议。** 分离软件机制、仿真、真机 bring-up 和 Foundation Model
   闭环，避免把接口正确性外推为复杂任务成功率。

## 2. 问题形式化

### 2.1 交互式具身闭环

令 \(x_t\) 表示不可完全观测的环境与机器人状态，\(o_t\) 表示系统获得的观测，
\(h_t\) 表示用户在时刻 \(t\) 提供的指令、追问或控制事件，\(m_t\) 表示 Harness
持久状态。Hey Robot 的一次高层决策可写为：

$$
z_k \sim \pi_{\mathrm{slow}}
\left(g, h_{\le k}, o_{\le k}, y_{<k}, m_k, \mathcal{C}\right),
$$

其中 \(g\) 是当前目标，\(\mathcal{C}\) 是 deployment 允许的能力集合，\(z_k\) 是一个
有界 Skill 或非物理 Harness Tool。对于物理 Skill，快系统在预算 \(B_k\) 内执行：

$$
(a_{k,1:n}, y_k) =
\pi_{\mathrm{fast}}(z_k, o_k; B_k),
\qquad
x_{t+1} \sim P(x_{t+1}\mid x_t,a_t).
$$

\(y_k\) 不是单一布尔值，而是包含 success、failure mode、termination reason、progress、
新观测和 evidence 的结构化结果。它在经典强化学习意义上承担环境反馈的作用，但 Hey
Robot 当前不是在线 RL 算法，也不要求运行时存在标量 reward。任务成功谓词或 benchmark
reward 可以在环境适配器中定义。

终态结果被归并到持久状态：

$$
m_{k+1}=U(m_k,z_k,y_k,h_{\le k+1}),
$$

随后慢系统重新决策。用户因此不是闭环外的启动按钮，而是可以持续改变目标和控制状态的
参与者。

### 2.2 三种时间尺度

系统显式分离：

```text
model decision  <<  one bounded Skill run  <<  durable user task
```

- **Decision**：一次模型返回最终文本或至多一个 typed Tool proposal；
- **Run**：一个具有唯一 `run_id`、预算、资源和终态的 Skill 执行；
- **Task**：跨越多个 run、用户轮次和进程生命周期的持久目标。

“快”与“慢”描述职责和时间范围，而不是固定频率、两个必然独立的进程或硬实时保证。
VLA/VLN 内部还可以拥有自己的多频率控制结构。

### 2.3 持久任务状态

一个持续任务表示为：

$$
\mathcal{T}=(g,u,r,q,S,\rho,d),
$$

其中 \(u\) 是 session，\(r\) 是机器人，\(q\) 是任务状态，
\(S=(s_1,\ldots,s_n)\) 是物理 step，\(\rho\) 是会话与回复路由，\(d\) 是 deadline。

$$
q\in\{\text{active},\text{paused},\text{completed},
\text{blocked},\text{cancelled},\text{failed}\}.
$$

`active` 与 `paused` 是未终止状态；`completed`、`blocked`、`cancelled` 和 `failed`
是终态。特别地，当前实现只有 `paused` 可以通过 `resume` 恢复；`blocked` 表示该任务
已停止，需要人工检查后创建新的后续任务，而不是直接续跑旧任务。

每个 step 持有唯一 `run_id`、Tool call ID、参数、状态、最后事件 sequence 和
`ToolOutcome`。默认任务预算为最多 24 个物理 step 和 3600 秒，profile 可以显式覆盖。

### 2.4 设计目标与非目标

设计目标包括：

- **Continuity**：任务事实不能只存在于模型上下文或 coroutine；
- **Steerability**：执行中新输入可在明确的 Skill 边界生效；
- **Bounded autonomy**：每个 Skill 有 schema、资源、超时、取消和终止语义；
- **No blind replay**：未知物理动作不能因重试语义被自动重复；
- **Capability least privilege**：Agent 只看到 deployment 显式开放的 Skill；
- **Replaceability**：模型、Skill、Driver、通道和传输可以沿稳定端口替换；
- **Observability**：task、step、run、事件、状态和媒体可以被检查。

Hey Robot 当前不提供硬实时控制、工业功能安全认证、通用 SLAM、开放世界自主导航、
任意物体操作、多机器人协调或跨存储分布式事务。

## 3. 相关工作与系统定位

### 3.1 语言模型与机器人技能组合

SayCan 将语言模型语义概率与技能可供性结合 [1]；Inner Monologue 使用环境反馈改进
语言模型规划 [4]；Code as Policies 让语言模型在受控 API 上生成策略 [5]。这些工作
说明了高层推理和结构化能力接口的价值。Hey Robot 不提出新的 planner，而是实现 planner
周围的任务生命周期、能力权限和物理执行协议。

### 3.2 具身多模态模型

PaLM-E 和 RT-2 展示了视觉语言知识向机器人决策与动作的迁移 [2,3]。OpenVLA 与
LeRobot 提供开放模型和训练部署生态 [11,12]。Hey Robot 把这些模型视为可替换的
ModelService 或 Skill implementation，不让模型服务拥有用户任务状态、资源锁或急停
权限。

### 3.3 层级、交互与终止

Hi Robot 使用高层 VLM 解释开放指令和 situated feedback，再由低层 VLA 执行原子动作
[6]。Hi-VLA 的系统研究表明，层级收益并不只来自“增加 planner”；终止、观测表示、
记忆和控制接口都会显著影响表现 [7]。InternVLA-N1 在导航模型内部以不同频率组合
grounding 与低层控制 [9]。

Hey Robot 在 Harness 层实现快慢分工，并将 termination reason、fresh observation 和
结构化 outcome 暴露为系统接口。它与模型内部层级正交：同一个 Harness Skill 可以包装
经典控制，也可以包装内部具有快慢结构的 VLA/VLN。

### 3.4 Harness 与长期运行

Harness Engineering 将 harness 概括为模型外部的工作流、工具、上下文、持久状态、
评估、权限和反馈系统，并强调用简单稳定的接口隐藏复杂运行逻辑 [10]。Harness VLA
进一步把冻结 VLA 作为可重试的局部 primitive，由外层 Agent 负责 grounding、staging、
反馈诊断和跨步骤记忆 [8]。

Hey Robot 接受这一“模型不拥有完整生命周期”的观点，但研究对象不同：本文关注生产式
机器人运行时的交互、持久化、幂等事件、取消和故障恢复。当前系统不执行自我改进，也没有
声称能够复现 Harness VLA 的操作性能。

### 3.5 Robot middleware 与行为编排

ROS 提供机器人通信、设备和进程生态 [13]；行为树提供显式可解释的任务结构 [14]。
Hey Robot 不替代 ROS，也不是行为树执行器。它位于通道/模型与机器人能力之间，管理
模型驱动任务跨越会话、异步执行和恢复的生命周期。

## 4. 系统设计

### 4.1 Embodied Agent Harness

![Hey Robot Embodied Agent Harness architecture](../images/architecture.png)

架构图中的 “skill intent” 表示慢系统选择的概念性子目标，不表示代码中已退役的
`SkillIntent` 消息控制链。当前生产主路径使用 typed Tool proposal 和 `SkillCommand`。

| 维度 | 慢系统 | 快系统 |
| --- | --- | --- |
| 时间范围 | 跨用户轮次、Skill 与服务重启 | 一个有界 Skill 和局部控制循环 |
| 职责 | 目标、推理、任务连续性、steer、恢复 | 感知、执行、资源、安全、局部反馈 |
| 实现 | `Agent`、`AgentRunner`、`AgentTaskStore`、`TaskCoordinator` | `SkillWorker`、option runner、`RobotRuntime`、Driver |
| 核心状态 | session、task、step、resume position | run、progress、resource、robot state |
| 时间保证 | 非实时模型调用 | 有预算和超时，但不是硬实时 |

### 4.2 当前运行拓扑

```text
Human
  Web / Voice / Feishu / CLI
              |
              v
GatewayService
  identity / episode / route / deterministic control
              |
              | ConversationTurn / AgentControl
              v
AutonomousAgentService
              |
              v
Agent -> AgentRunner -> AgentToolExecutor
              |
              v
TaskCoordinator -> SkillClient
                      |
                      v
             SkillWorker -> SkillRunner -> Skill handler
                                           |          |
                                           |          +-- gRPC --> ModelService
                                           v
                                  LocalRobotClient
                                           |
                                           v
                                  RobotRuntime -> Driver
```

默认 `DeploymentRunner` 在一个 asyncio 进程中组合 RobotService、SkillWorker、
AutonomousAgentService、GatewayService 和可选 HumanFollowService。该选择减少当前
单机器人部署的协调复杂度。NATS 用于服务与通道的消息传输和事件投影；测试或简化仿真
可以使用 in-memory bus。GPU 模型服务可通过 gRPC 独立部署，因此系统支持混合式分布，
但不应被描述为所有组件默认分布式运行。

旧 `skill.intent`、`robot.action` 和基于 topic 的第二条动作入口已经退出当前执行主链；
少量同名 DTO 与 Topic 常量仅为源码兼容或投影保留。

### 4.3 代码所有权与稳定端口

| 层 | 拥有 | 不拥有 |
| --- | --- | --- |
| Gateway | identity、channel route、episode、确定性控制路由 | 物理执行与任务 step |
| Cognition | 对话上下文、目标、Tool 选择、持续任务 | RobotAction 与 Driver |
| Skill | schema、资源、预算、取消、progress、结果 | 用户 session 与硬件实现 |
| Foundation Model | VLA/VLN 推理及取消 | Agent task、资源锁与机器人安全 |
| Robot Runtime | lifecycle、observation、health、safety、control plane | 高层目标规划 |
| Driver | 真机或仿真的具体 I/O | Agent/Skill 生命周期 |

物理请求的唯一主链为：

```text
Agent -> SkillClient -> SkillWorker -> Skill handler
      -> LocalRobotClient -> RobotRuntime -> RobotDriver
```

这条所有权规则使上层模型无法绕过 Skill schema、资源管理和 Robot Runtime safety。

## 5. 慢系统：持续任务与交互

### 5.1 单次决策边界

`AgentRunner` 是无外部 I/O 的 decision boundary。它接收规范化 `ModelMessage` 和当前
Tool allowlist，返回最终文本或一个 typed proposal。未知 Tool、多 Tool 并发提议、
参数错误、空响应和超时均被结构化拒绝。

`Agent` 在一次 wake-up 内最多执行 8 次模型 decision。非物理 Harness Tool 可以把结果
在同一次 wake-up 送回模型；物理 Tool 提交成功后返回 `waiting`，结束当前模型控制流，
等待 Skill terminal event。若有持续任务且单次 wake-up 达到模型轮次上限，任务进入
可恢复的 `paused`；这与任务预算耗尽产生的终态 `blocked` 不同。

### 5.2 Persist-before-submit

对物理 proposal，`AgentToolExecutor` 先取得或创建持续任务，随后由
`TaskCoordinator`：

1. 为 step 分配唯一 `run_id`；
2. 在 SQLite 写入 pending step、Tool call ID 和参数；
3. 构造 `SkillCommand`；
4. 调用 `SkillClient.submit()`；
5. 将提交异常归并到同一个 step，而不是创建无关联重试。

因此，系统不会先发出物理动作、事后才尝试补记任务。该机制缩小了动作已执行但系统没有
step 记录的故障窗口，但并不构成数据库与硬件之间的原子事务。

### 5.3 事件驱动续跑

`SkillRunner` 为一个 run 产生单调 sequence 的生命周期：

```text
accepted -> running -> progress* -> completed | failed | cancelled
```

`FileRunStore` 持久化 command 和事件；`AgentTaskStore` 用 `run_id` 关联 step，拒绝
sequence 小于或等于已应用值的重复/乱序事件。终态事件写入 outcome 和恢复位置。

`AutonomousAgentService` 消费 `SkillClient.events()`。当终态已成功归并且任务仍为
active 时，它恢复 session、route 和 step，构造原 assistant Tool call 与新 Tool
result 上下文，再唤醒同一 session 的 Agent。模型随后选择下一个 Skill 或用最终文本
结束任务。若环境明确返回 `termination_reason=environment_done`，协调器可以直接完成
任务，避免要求 LLM 再次确认环境终态。

### 5.4 启动 reconciliation

服务启动时，系统检查 active task 的非终态 run，并调用 `SkillClient.status(run_id)`：

- 已有持久 terminal event：归并结果并恢复 Agent；
- worker 仍拥有 run：返回最新 active event；
- 有持久 submission 但没有 worker 所有者：产生 `execution_lost` failure；
- 没有足够证据判断动作状态：不自动重放原命令。

这是一种偏向物理安全的最终一致性策略。对数字 API，“至少一次”重试通常可接受；对
`move_base`、夹爪闭合或机械臂动作，重复执行可能改变环境甚至造成危险。

### 5.5 执行中交互

Gateway 把 Web、CLI、Voice 和 Feishu 输入统一为 `ConversationTurn`，并通过 identity、
episode、account、chat 与 sender metadata 保留回复路由。Gateway 的 append-only
`JsonlEpisodeStore` 保存通道 episode；Cognition 的 SQLite `ConversationStore` 保存
模型会话上下文；`AgentTaskStore` 单独保存物理任务事实。

当 session 有 active Skill run 时，新用户文本会先写入 conversation，系统告知用户该
修改将在当前操作结束后处理。terminal event 到达后，最新 transcript 和 Skill outcome
共同进入恢复上下文。当前实现不把自然语言实时注入正在运行的 VLA/VLN 或经典控制循环，
因此交互延迟受 Skill 粒度和终止时间影响。

### 5.6 确定性控制

- `pause`：请求取消 active run，并把任务设为可恢复的 `paused`；
- `resume`：仅在任务为 `paused` 且没有 active run 时重新唤醒 Agent；
- `cancel`：取消 active run，并把任务终止为 `cancelled`；
- `block`：停止推进并形成终态 `blocked`，不能直接 resume；
- `emergency_stop`：调用 Robot Runtime stop primitive，同时取消机器人相关 run，
  并将关联任务取消。

Gateway 对明确的暂停、恢复、取消和急停短语提供确定性路由。软件急停路径不能代替物理
断电、硬件限位或经过认证的安全系统。

## 6. 快系统：有界 Skill 执行

### 6.1 显式能力面

`skills.modules` 加载 registry，`skills.tools` 是投影给 Agent 的 deployment allowlist。
一个 Skill 描述：

```text
name / description / JSON Schema / handler
resources / timeout_sec
supported_robots / required_actions / required_models
```

注册不等于授权；未进入 `skills.tools` 的 Skill 不会成为 Agent Tool。默认 XLeRobot
real/sim profile 当前只开放 `inspect_scene`、`move_base` 和 `turn_base`。因此默认
配置可以验证交互、感知和底盘移动闭环，不能据此声称具备通用抓取、放置或家务能力。

### 6.2 Skill Worker 与资源语义

`SkillWorker` 和 `SkillRunner` 提供二次参数校验、run ID 幂等提交、deadline/timeout
收敛、cooperative cancellation、progress/terminal event、run store 和事件投影。
`ResourceManager` 以 `(robot_id, resource)` 为键串行化同名资源，并支持同一 run 的
可重入持有。Skill contract 还可以按 `arm`、`gripper`、`camera` 等实例化资源名称；
声明 `robot` 或 `robot.actuation` 表示占用全局执行资源。

投影队列有界；压力过大时可以丢弃陈旧 progress，但不会把 durable run store 中的执行
事实改写为失败或成功。事件投影是可观察性数据面，不是执行事实源。

### 6.3 终止与新鲜观测

VLA option 实现有界 observe–infer–act 循环。每一步使用当前 observation 调用模型，
经 Robot Runtime 执行动作，再请求 frame ID 更新的 fresh observation。终止策略依次
处理 environment done、model done、no action 和 max steps；action failure、模型失败
或 observation stale 形成显式 failure mode。

VLN option 同样在 `max_steps` 内执行 observe–plan–act，支持 secondary observation，
将 heading、pixel goal 或离散 action code 转换为 `move_base`、`turn_base` 和
`stop_motion` primitive。只有环境、模型或 stop primitive 给出终止证据时才报告到达；
预算耗尽会失败为 `budget_exhausted`。

这种设计呼应层级 VLA 研究中“termination 与 observation interface 和 planner 同样
重要”的发现 [7]，但当前仓库尚未提供足以比较不同终止策略的实验结果。

### 6.4 Foundation Model 边界

VLA/VLN 通过统一 gRPC ModelService contract 提供：

```text
GetHealth / ExecuteSkill / CancelSkill
```

模型进程负责推理，不直接访问 AgentTaskStore，也不决定用户任务生命周期。VLA 输出的
embodiment-native action 和 VLN 输出的移动 primitive 最终仍经过 Robot Runtime。
实验配置、adapter 或 checkpoint 路径的存在只证明集成入口存在，不证明特定模型已经在
目标环境稳定闭环。

### 6.5 Robot Runtime 与安全门

`LocalRobotClient` 将 Skill 请求转到 `RobotRuntime`。Runtime 统一拥有 driver lifecycle、
capabilities、health、observation、media、control plane 和 emergency stop。

动作进入 Driver 前至少经过两类确定性检查：

1. **Skill admission**：必需参数、robot readiness、资源可用性、当前状态和 battery；
2. **Runtime safety**：robot online、estop/protective-stop/collision 等 flag、critical
   battery、动作维数以及 driver capability 中声明的数值边界。

`stop_motion` 绕开普通 readiness 限制，并由 control plane 的停止路径执行。当前这些
检查属于软件安全防线，不提供碰撞规避、形式验证或功能安全认证。

## 7. 持久化、可观测性与不变量

### 7.1 状态事实源

| 状态 | 实现与职责 |
| --- | --- |
| 通道 episode | append-only `JsonlEpisodeStore` |
| 模型会话 | SQLite `ConversationStore` |
| 持续任务与 step | SQLite `AgentTaskStore` |
| Skill command/event/artifact | `FileRunStore` |
| Runtime event | 有界 JSONL `RuntimeEventStore` |
| 图像与大对象 | `LocalMediaStore` |

物理状态的最终事实仍来自 Robot Runtime 的 observation、status 和 SkillResult，而不是
聊天文本。多个本地存储之间没有跨介质事务；一致性依赖唯一 ID、persist-before-submit、
单调事件和 reconciliation。

### 7.2 当前代码不变量

架构边界和测试维护以下不变量：

1. 每个 session 至多一个 active 或 paused task；
2. 每个 task 至多一个 active Skill run；
3. 一个 deployment 至多一个 enabled autonomous Agent；
4. `AgentRunner` 不执行外部 I/O；
5. Cognition 不构造 RobotAction，也不直接访问 Driver；
6. Agent 只通过 `SkillClient` 提交和观察物理 Skill；
7. Skill 参数在 proposal 和执行边界分别校验；
8. 物理 run 提交前已有持久 pending step；
9. 重复或乱序 terminal event 不会重复推进任务；
10. 所有权丢失的物理动作不会自动重放；
11. 急停路径不依赖 LLM。

### 7.3 运维界面

Web `/tasks` 与 task API 暴露任务列表、step、run 状态和时间线。Gateway 还把 Robot
status、Skill lifecycle 和 Runtime event 投影给通道 UI。日志可输出人类可读格式或
JSONL。上述界面用于诊断与审计，不是安全认证控制台。

## 8. 当前实现与能力边界

### 8.1 论文主张与代码映射

为使系统论点可由开源 artifact 检查，主要机制对应如下：

| 论文机制 | 主要实现 |
| --- | --- |
| stateful Agent 与每次唤醒预算 | `src/hey_robot/cognition/runtime/agent.py` |
| 无 I/O、单 proposal decision | `src/hey_robot/cognition/runtime/agent_runner.py` |
| 物理/非物理 Tool 分流与任务预算 | `src/hey_robot/cognition/tools/executor.py` |
| persist-before-submit、事件归并、reconciliation | `src/hey_robot/cognition/runtime/task_coordinator.py` |
| task/step 状态机与单 active task/run | `src/hey_robot/cognition/runtime/agent_task_store.py` |
| session Agent、终态消费与启动恢复 | `src/hey_robot/cognition/autonomous_agent_service.py` |
| Skill 生命周期、资源和持久 run | `src/hey_robot/skills/worker.py`、`runner.py`、`resources.py` |
| VLA/VLN 有界循环与终止 | `src/hey_robot/skills/vla/`、`src/hey_robot/skills/vln/` |
| observation、safety 与 driver boundary | `src/hey_robot/robot_runtime/` |
| 默认单进程组合与可选模型 sidecar | `src/hey_robot/app/runner.py`、`runtime_components.py` |
| deployment 能力面 | `configs/*.yaml` 中的 `skills.tools` 与 `model_services` |

这张表描述代码所有权，不替代实验 artifact。论文提交版本还应固定 commit hash，并将每项
实验绑定到具体配置快照。

### 8.2 能力状态

| 能力 | 状态 | 可验证边界 |
| --- | --- | --- |
| typed Agent Tool loop | 已实现 | 每次 decision 至多一个 proposal；错误结构化拒绝 |
| 持续任务 | 已实现 | SQLite task/step、事件续跑、暂停/恢复、启动 reconciliation |
| 多通道交互 | 已实现 | Web、CLI、Voice、Feishu；启用情况依 profile 与凭据 |
| Skill Harness | 已实现 | local worker、schema、resource、timeout、cancel、run store |
| Mock / MuJoCo | 已实现 | 用于 Harness 与有限机器人能力验证 |
| XLeRobot native driver | 已接入 | 仍需逐机标定、诊断和物理安全验证 |
| VLA / VLN option | 实验中 | contract、循环和 profile 已有；默认 XLeRobot 不开放 |
| RoboCasa365 agent profile | 实验中 | `inspect_scene` + `manipulate`，依赖独立环境和模型 artifact |
| 开放世界长程家务 | 待评估 | 当前版本不作能力承诺 |

当前默认主链是 local Skill execution；NATS 承担 Gateway/Agent 等服务通信与只读事件
投影。ModelService 可以独立部署，但本地 SQLite、JSONL 与 FileRunStore 不构成分布式
数据库。项目更准确的定位是“具有可分离服务边界的混合式 Harness”，而不是默认全分布式
机器人平台。

## 9. 评估方法

本节是预注册式实验协议，不包含结果。最终论文必须在填入实验环境、commit、模型版本、
样本量、随机种子、置信区间和失败案例后，才能把本节改写为经验结论。

### 9.1 评估原则

参考 Hi Robot、Hi-VLA 和 Harness VLA 的实验组织方式 [6–8]，评估分离四个问题：

1. **机制是否正确？** 持久化、幂等、控制和恢复是否满足状态机不变量；
2. **交互是否有效？** 用户反馈是否被接收，并在预期安全点改变后续决策；
3. **局部能力是否可靠？** Skill 的执行、终止、观测和 failure mode 是否可信；
4. **组合是否产生收益？** 完整 Harness 是否提高长程任务进展，而非只增加开销。

模型、prompt、Skill surface、观测和机器人初始状态应在对应消融中保持不变。否则不能把
性能变化归因于 Harness。

### 9.2 任务分层

| 层级 | 任务 | 目的 |
| --- | --- | --- |
| L0 软件在环 | protocol 拒绝、事件乱序、崩溃、恢复、控制竞争 | 验证系统机制 |
| L1 Mock/MuJoCo | 观察—移动—再观察、失败注入、执行中修正 | 验证闭环与交互 |
| L2 XLeRobot bring-up | 场景检查、转向、短距离移动、暂停与急停 | 验证真机基础链 |
| L3 Foundation 闭环 | VLA 操作、VLN 导航、termination ablation | 验证局部模型能力 |
| L4 长程组合 | 多子目标、用户纠偏、恢复和任务验证 | 验证端到端 Harness |

L2–L4 必须分别报告，不能用仿真结果替代真机结果，也不能把单 Skill 成功率写成长程任务
成功率。

### 9.3 任务类别

- **短程技能任务**：一个 Skill 可完成，用于估计低层能力上限；
- **长程组合任务**：需要多次感知、移动或操作；
- **语义推理任务**：目标含约束、指代或需要重新 grounding；
- **持续交互任务**：执行中追问、目标修正、暂停、恢复和取消；
- **恢复任务**：timeout、model failure、transport failure、worker ownership loss、
  Agent/Gateway 重启；
- **终止任务**：模型完成、环境完成、无动作、预算耗尽和 stale observation。

### 9.4 对照与消融

| 条件 | 隔离的机制 |
| --- | --- |
| H0 完整 Harness | 当前持久任务、事件续跑、Skill/Runtime 边界 |
| H1 无持久 task/step | 只使用短期对话历史 |
| H2 同步 Tool loop | 模型请求阻塞等待完整物理 Skill |
| H3 无 reconciliation | 重启后只允许人工恢复 |
| H4 无执行中 steer | active run 期间拒绝新用户文本 |
| H5 无结构化 outcome | 仅向 planner 返回自然语言成功/失败 |
| H6 flat policy | 在适用任务中让单个 VLA/VLN 覆盖更长 horizon |
| H7 termination ablation | 固定 horizon、仅 model done、仅 environment done |
| H8 observation ablation | stale/latest frame、结构化 observation、图像历史 |

绕过 readiness、资源互斥或 safety gate 的 H5 类安全消融只允许在纯软件或隔离仿真中
执行，不能在真机上制造不安全对照。

### 9.5 指标

**Harness 与交互指标**

- task state continuity 与合法状态转移率；
- persist-before-submit 覆盖率；
- terminal event 到下一 decision 的恢复率和延迟；
- crash recovery resolution rate；
- duplicate physical action rate；
- duplicate/stale event rejection rate；
- correction incorporation rate 与 correction latency；
- pause/resume/cancel/emergency-stop success rate；
- unauthorized/infeasible proposal rejection rate；
- 人工介入次数与任务终止原因分布。

**具身任务指标**

- Task Success、Subgoal Success 和 Task Progress；
- Instruction/Steering Accuracy；
- Skill success、timeout、cancel 和 failure-mode 分布；
- termination precision/recall 与 premature/late termination；
- semantic Skill 数、底层 action 数和总任务时长；
- planner、model、queue、Skill、Robot Runtime 各阶段延迟；
- 真机安全事件、操作员接管和硬件异常。

结果应报告均值与分位数或置信区间，并发布失败轨迹。软件在环、仿真和真机必须分表。

### 9.6 待验证假设

- **R1**：持久 task/step 状态降低长程任务中的上下文与执行状态漂移；
- **R2**：terminal event 续跑比同步等待更能承受长时 Skill 与服务重启；
- **R3**：persist-before-submit、run identity 和幂等归并降低重复物理动作风险；
- **R4**：安全点 steer 与确定性控制提高用户纠偏和中断成功率；
- **R5**：显式 Skill surface、admission 和 Runtime safety 降低不可执行请求进入 Driver
  的比例；
- **R6**：层级 Harness 仅在 planner、termination、observation 和低层 policy 同时可靠
  时提高长程任务进展；
- **R7**：结构化 outcome 与明确 failure mode 比自由文本反馈更有利于诊断和恢复。

## 10. 讨论

### 10.1 Harness 与模型能力必须分开

更强的 LLM/VLM 可以改善语义理解，更强的 VLA/VLN 可以提高局部成功率，但它们不能
替代任务持久化、资源互斥、取消、启动恢复与急停。反过来，可靠 Harness 也不能把未训练
或未对齐 embodiment 的策略变成有效技能。评估必须同时报告模型条件与系统条件。

### 10.2 为什么不把快慢等同于异步

快慢系统是认知职责和执行 horizon 的抽象；事件驱动是 Hey Robot 的实现选择。默认
Agent 与 Skill Worker 可以在同一事件循环中运行，而一个 VLA option 内部又可以连续
执行多个高频动作。把二者等同会混淆概念层与部署层。

### 10.3 为什么使用持久 artifact

模型 context 适合语义工作记忆，不适合作为物理事实数据库。Task、step、run、event、
observation 和 artifact 需要可寻址、可检查和可恢复。持久状态也让失败分析可以引用证据，
而不是只依赖模型生成的解释 [10]。

### 10.4 人仍然在闭环中

Hey Robot 的目标不是把人从系统中移除，而是让人从逐动作遥控上移到目标、纠偏、审批和
异常处理。当前 safe-boundary steer 是保守设计：它牺牲一部分即时性，换取不在未知时刻
修改正在执行的物理控制目标。

### 10.5 开源生产化要求

生产级开源不仅要求代码可运行，还要求能力主张可复现、配置边界清晰、失败可诊断、升级
可迁移。当前项目仍需持续完善 CI/release artifact、安全披露、认证与 TLS 基线、存储
migration、benchmark manifest、模型 checkpoint 管理和真机安全手册。

## 11. 局限性

- 一个 deployment 当前只允许一个 enabled autonomous Agent，尚无多机器人协调；
- 默认 XLeRobot Skill surface 只覆盖感知和底盘基础移动；
- 复杂真机长程任务尚无可报告的统计结果；
- steer 只在 Skill 终止后的安全边界生效；
- 本地 SQLite、JSONL 和 FileRunStore 没有跨介质事务；
- NATS core pub/sub 不默认提供完整 durable replay 语义；
- event projection 与 Agent/Gateway 总线仍需更系统的断连恢复实验；
- VLA/VLN 的权重、观测映射、动作空间和 embodiment 需要独立验证；
- termination、目标关联和世界状态表示仍然有限；
- 软件 safety gate 不等于碰撞规避、硬实时控制或功能安全认证；
- 当前恢复以显式状态和重新决策为主，尚未使用学习型恢复策略；
- 本文仍是系统设计与评估协议草稿，不能代替已完成的对照实验。

## 12. 结论

Hey Robot 将机器人 Agent 从“一个会调用工具的模型”扩展为用户、环境、持久状态、
语义决策和物理执行共同构成的闭环。系统以 Embodied Agent Harness 组织快慢双系统：
慢系统维持目标、交互与任务连续性，快系统执行有界 Skill，并以结构化 outcome、观测和
终止原因反馈给慢系统。

当前开源实现已经建立 persist-before-submit、Skill lifecycle、事件驱动续跑、启动
reconciliation、能力 allowlist 和确定性控制等主机制，并将 Mock、MuJoCo、XLeRobot、
RoboCasa 及可选 VLA/VLN 接入统一边界。但默认能力仍以基础感知与移动 bring-up 为主。
下一阶段的重点不是扩大系统口号，而是执行分层对照实验：分别证明 Harness 连续性、交互
可控性、局部 Skill 可靠性，以及它们组合后对长程任务进展的实际贡献。

## References

1. Ahn, M. et al. “Do As I Can, Not As I Say: Grounding Language in Robotic Affordances.” CoRL, 2022.
2. Driess, D. et al. “PaLM-E: An Embodied Multimodal Language Model.” ICML, 2023.
3. Brohan, A. et al. “RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control.” 2023.
4. Huang, W. et al. “Inner Monologue: Embodied Reasoning through Planning with Language Models.” CoRL, 2022.
5. Liang, J. et al. “Code as Policies: Language Model Programs for Embodied Control.” ICRA, 2023.
6. Shi, L. X. et al. “Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models.” arXiv:2502.19417, 2025.
7. Hu, J. et al. “What Matters in Orchestrating Robot Policies: A Systematic Study of Hierarchical VLA Agents.” arXiv:2606.10267, 2026.
8. Zhang, Y. et al. “Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents.” arXiv:2607.08448, 2026.
9. Wei, M. et al. “Ground Slow, Move Fast: A Dual-System Foundation Model for Generalizable Vision-Language Navigation.” ICLR, 2026.
10. Weng, L. “Harness Engineering for Self-Improvement.” Lil’Log, 2026.
11. Kim, M. J. et al. “OpenVLA: An Open-Source Vision-Language-Action Model.” arXiv:2406.09246, 2024.
12. Cadene, R. et al. “LeRobot: An Open-Source Library for End-to-End Robot Learning.” ICLR, 2026.
13. Quigley, M. et al. “ROS: an Open-Source Robot Operating System.” ICRA Workshop, 2009.
14. Colledanchise, M. and Ögren, P. *Behavior Trees in Robotics and AI*. CRC Press, 2018.
