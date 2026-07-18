# Hey Robot：面向真实机器人的原生 Embodied Agent Harness 与异步快慢双系统

## 摘要

大语言模型智能体（LLM agents）正在被用于把自然语言指令连接到机器人技能。问题在于，普通的“LLM + 工具调用”循环并不天然适合真实机器人：观测会过期，动作可行性依赖硬件状态，任务进度需要跨轮次持久化，失败技能需要结构化恢复，用户还会在机器人执行过程中追问、修正、打断或重新安排优先级，而不是等待一个单轮工具调用结束。

本文提出 **Hey Robot**，一个不依赖通用 LLM Agent 框架、面向真实机器人原生构建的 Embodied Agent Harness。系统采用异步快慢双系统：上层慢系统由自主实现的 LLM Agent Runtime 承担长程认知、任务状态、记忆、规划与恢复；下层快系统由 Skill OS、VLA/VLN 等可选模型服务和 Robot Runtime 承担短时域具身决策与执行。这里的“快慢”描述决策层级和时间范围，不表示下层已经达到硬实时控制。

Hey Robot 将通道入口、Agent 组件、任务状态、Skill OS、Foundation Model 服务、感知、恢复和机器人驱动划分为清晰边界，并通过类型化异步消息连接。当前代码已经实现主动感知新鲜度检查、技能契约门控、持久任务检查点、执行反馈、类型化恢复和忙碌状态交互路径。默认 real/sim 配置仍处于硬件 bring-up 阶段，启用的是感知、底盘、跟随、安全、机械臂和夹爪等经典技能；VLA/VLN 通过独立实验配置接入，其中 VLA 尚未加载真实模型，仅能验证服务接口和调度链路。因此，本文区分已实现的 Harness 机制、实验中的 Foundation Model 接入，以及尚待完成的真机任务评估，不把路线图表述为已验证结果。

## 1. 引言

LLM-based agents 让自然语言任务转换为工具调用变得非常容易。在数字环境中，这种模式通常有效，因为状态显式、动作可回滚、失败重试成本低。

真实机器人不同。机器人可能在 LLM 推理期间已经移动；相机图像可能在规划时已经过期；夹爪显示关闭并不等于真正抓住物体；用户可能在任务中途插话、修正、询问状态或打断；一个技能完成也不等于整个任务完成。

因此，把 ReAct-style “thought -> action -> observation” 循环直接套到机器人上，会产生系统性问题。本文面向半开放、中程、桌面和办公室任务。以下是计划评估的目标任务，而不是当前版本已经完成真机验证的能力：

- “把前面桌子上的水瓶拿给我。”
- “清理桌面，把垃圾扔到垃圾桶。”
- “把杯子放到桌上，把木块放到架子上。”

这些任务不是完全开放世界任务。它们有明确约束：工作空间较小，任务相关物体通常为 1-5 个，基础任务的技能预算不超过 10，场景可以有轻微变化但不是高动态环境，成功需要通过观测、机器人状态或操作员标签验证。涉及抓取、放置和递交的真机任务以可用且经过验证的 VLA 或其他操作 backend 为前提。

Hey Robot 的核心观点是：LLM 循环只是具身系统中的一个认知组件。真正让机器人可靠完成中程任务的，是围绕它的系统机制：任务状态、主动感知、技能契约、执行反馈、恢复机制和交互连续性。对产品化机器人而言，用户体验不只来自最终成功率，还来自机器人能否清楚表达当前状态、接住用户修正、在忙碌时快速响应，并在失败时给出可理解的恢复路径。

### 贡献

本文贡献包括：

1. 总结中程具身任务中 LLM 驱动机器人智能体的主要失败模式。
2. 自主实现不依赖 LangChain、LangGraph、AutoGen 等通用 LLM Agent 框架的 Agent Runtime，并以异步消息连接上层慢系统与下层短时域执行。
3. 实现 Agent/Cognition、Skill OS、Foundation Model 和 Robot Runtime 四层架构。四层描述代码与部署边界，快慢系统描述决策层级，两者是正交视角。
4. 设计技能契约与 backend resolution 机制，在物理执行前检查必需参数、资源、硬件就绪状态、backend availability 和安全状态。
5. 引入任务内交互机制，支持状态询问、纠偏、确认、打断和继续执行。
6. 提出分阶段的软件在环、仿真与真实 XLeRobot 实验协议；本文草稿不将接口联调或计划实验当作真机成功率证据。

## 2. 与 Agentic Robot OS 的关系

LimX COSA 等系统提出了一个重要方向：具身机器人需要连接高层认知和运动控制的系统层，而不是只部署单个模型 [9]。Hey Robot 接受这一问题设定，但不声称实现了完整人形机器人 Agentic OS。

Hey Robot 的范围更窄：上层慢系统由 Agent 层负责任务理解、对话连续性、主动感知和恢复；下层快系统由 Skill OS、可选的 VLA/VLN ModelService 和 Robot Runtime 完成短时域决策与执行。系统当前面向 XLeRobot 的单机部署，不包括完整全身控制、通用移动操作、硬实时运动生成或开放世界自主。

Hey Robot 不主张解决完整人形小脑基础模型。它是面向小型机器人平台的四层 Harness，重点是：

- LLM 推理
- 场景证据
- 技能契约
- 任务状态
- 执行反馈
- 类型化恢复
- 多轮交互和忙碌状态处理
- Skill OS
- 可替换的 Foundation Model execution boundary

因此，本文研究的不是完整人形 Agentic OS，而是：如何在小型真实机器人上，用原生 Harness 约束语言模型对机器人能力的调用，并为 VLA/VLN 等模型留下可替换的服务边界。当前实现能够验证 Harness、经典技能和 ModelService 接口链路；“由短时域 Foundation skills 组合完成复杂真机任务”仍是后续需要实验验证的假设。

需要区分两种“双系统”：InternVLA-N1 等工作在导航模型内部组合慢速 grounding/planning 与快速控制 [10, 11]；Hey Robot 的快慢划分位于系统层，上层 Agent 是慢系统，下层 VLA/VLN、Skill OS 和 Robot Runtime 整体属于相对更短时域的快系统。下层模型内部仍然可以继续包含自己的快慢层级。

## 3. 失败模式

本文将中程具身任务中的主要失败归纳为六类。

### F1：感知缺失或过期

Agent 使用旧观测做规划。例如上一帧图像中水瓶在桌面中央，但实际已经被移动。更强的 VLM 不能单独解决这个问题，因为问题不在“看不懂”，而在“看的不是当前状态”。

系统层面的解决方案是主动感知新鲜度门控：在视觉规划前检查观测时间、图像数量和新鲜度，必要时触发 `inspect_scene`。

### F2：目标定位错误

系统看到了场景，但选错物体或目标位置。例如用户要水瓶，Agent 却锁定杯子。这类问题需要结构化场景证据、目标确认和动作后验证。

### F3：物理不可行

LLM 选择了语义上合理的技能，但当前机器人状态不允许执行：

- 电量过低
- 机械臂不可用
- 夹爪资源被占用
- 相机不可用
- 必需参数缺失
- 机器人处于紧急状态

标准工具 schema 只能检查参数类型，不能检查物理就绪状态。因此需要技能契约门控。

### F4：操作执行失败

计划是合理的，但执行失败。例如夹爪没夹住、物体滑落、机械臂姿态不合适、底盘未对准，或 VLA 策略执行失败。系统必须区分子目标失败和任务失败，并决定重试、重新观察、重规划、停止或询问操作员。

### F5：任务状态漂移

多步骤任务中，Agent 可能忘记已完成哪些子目标、哪个物体被移动到哪里、哪个技能失败过、恢复是否已经尝试过。这不能只靠更长上下文窗口稳定解决，需要持久任务检查点和任务局部记忆。

### F6：交互状态断裂

用户在机器人执行中途可能说“不是这个”“先停一下”“你现在在干什么”“继续刚才那个”。如果系统只把这些输入当成新的独立聊天轮次，就会丢失当前任务、活跃技能、指代对象和机器人忙碌状态，导致错误打断、重复执行或无法恢复。因此需要显式的对话状态、指代记忆和忙碌轮次处理机制。

## 4. 系统架构

Hey Robot 是一个原生实现、面向服务的 Embodied Agent Harness。它没有把 planning、tool loop 或 task state 委托给通用 LLM Agent 框架；OpenAI-compatible SDK 只作为模型 provider client。Agent 循环、工具协议、任务状态和执行反馈由 `AgentRuntime`、`RobotAgentCore` 及任务运行时自主实现。

默认 `hey-robot run` 会在同一进程中启动多个独立 asyncio 服务，各服务通过 NATS client 交换消息；VLA/VLN 等模型可作为独立 gRPC `ModelService` 部署。因此，“面向服务”描述协议和责任边界，不表示默认部署必须是多主机微服务。

```text
                         slow system
User Channel -> Gateway -> Agent/Cognition
                           | AgentRuntime
                           | TaskRunManager / MemoryBroker / SceneRuntime
                           | request_skill / request_perception
                           v
                     NATS: skill.intent
                           |
---------------------------+--------------------------------
                         fast system
                     Skill OS
                     | SkillContractRuntime
                     | resource/readiness/backend gates
                     +------------------+
                     |                  |
                 classic skill     gRPC ModelService
                                      VLA / VLN
                     |                  |
                     +--------+---------+
                              v
                        Robot Runtime
                        driver / hardware
                              |
             robot.status / robot.observation /
                  skill.event / skill.result
```

图中的上下边界表示决策层级。NATS、gRPC、设备总线和局部控制循环各自具有不同延迟特征；当前系统不提供硬实时调度保证。

### 4.1 协议边界

核心消息类型包括 `UserTurn`、`AgentReply`、`RobotObservation`、`RobotStatus`、`SkillIntent`、`SkillEvent` 和 `SkillResult`。主要 topic 使用 `user.turn`、`skill.intent`、`robot.status`、`robot.observation`、`skill.event` 和 `skill.result` 等点分名称。`RobotAction` 属于 Robot/driver 侧执行边界，不作为 Agent 层提交动作的接口。消息共享 `Envelope`，其中包含 trace、episode、channel、agent、robot 和部署元数据，用于维持跨通道、跨服务的任务连续性。

### 4.2 Gateway

`GatewayService` 将 CLI、Web、Voice、Feishu 等输入统一成 `UserTurn`。它负责路由用户消息、分配 episode、追加对话历史，并把 `AgentReply` 送回原通道。Teleop 可以作为未来通道扩展，但不是当前论文主线。

### 4.3 Agent 运行时

`AutonomousAgentService` 是 Agent 侧服务壳，订阅用户轮次、机器人状态、机器人观测、技能事件和技能结果。复杂职责拆分给：

- `RobotAgentLoop`：轮次生命周期状态机，负责 restore → build → run → save 流程。
- `TaskRunManager`：持久任务状态、检查点和恢复上下文。
- `MemoryBroker`：统一记忆路由，根据 task state（active / recovering / completed）选择性组合 task memory、scene evidence 和 LTM。
- `SceneRuntime`：场景记忆、场景证据查询和主动感知门控。
- `RobotAgentCore`：LLM / 工具执行与 skill-level 决策。
- Gateway 的确定性安全路由：处理急停、取消、状态查询和确认；其他自然语言交给 presentation router。
- `RobotTurnPolicy` / `AgentTurnPolicy`：构建工具权限、主动感知要求和 scene freshness；它们不是用户意图分类器。
- `BusyTurnHandler`：机器人忙时绕过完整 LLM 推理，直接处理状态查询和打断，或将纠偏、追问、重试和复位排入安全边界。
- `AgentNotificationRuntime`：任务进度和恢复通知。

### 4.4 轮次生命周期

每个用户轮次经过固定状态机：

```text
restore -> build -> run -> save
```

意义在于：真实机器人任务不是一次性聊天。系统必须恢复任务状态，构建最新上下文，运行认知核心，然后保存执行结果。

### 4.4.1 交互连续性

Hey Robot 将对话状态与任务状态绑定，而不是只保存聊天历史。机器人忙碌时，规则分类器将输入区分为状态查询、打断、急停、纠偏、追问、重试或复位。状态查询直接从缓存的 robot snapshot 生成回复；打断和急停发布 interrupt skill intent；其余更新进入 pending-turn 队列，在后续安全边界合并。待确认回复另由 confirmation interpreter 处理。当前分类以关键词和任务状态为主，并不等价于通用自然语言意图理解。

### 4.5 主动感知

在视觉相关任务中，`SceneRuntime` 和 turn policy 会检查最近观测是否缺失、没有图像或超过 freshness 阈值。如果需要刷新，Agent 通过 `request_perception` 请求新的感知证据；具体 backend 可以调用 `inspect_scene`。新证据随后进入当前轮次上下文。当前实现主要验证时间与图像可用性，不声称已经解决开放词汇目标关联。

### 4.6 技能契约

`SkillContractRuntime` 根据 skill catalog、当前 implementation 和机器人状态检查技能请求：

- 技能是否存在、是否启用以及能否解析到 implementation。
- 必需参数是否齐全。
- 必需资源是否可用（arm、gripper、camera、base）。
- 机器人状态是否允许执行（电量、急停、硬件就绪、readiness gate）。
- 是否与正在执行的技能发生资源冲突。
- ModelService 是否可用（如 VLA foundation backend）。

这相当于把 LLM 的动作提议放进确定性的安全和可行性门控。生产目标是只向 Agent 暴露稳定的 semantic skills；当前 XLeRobot 主配置采用 `bringup` mode，仍包含关节、夹爪等实现级技能，因此不能把现有目录直接表述为已经收敛的纯语义接口。

### 4.7 执行反馈和恢复

技能完成不等于任务完成。`SkillResultHandler` 收到结果后调用 execution feedback evaluator，再由任务运行时记录反馈并结合 `RecoveryManager` 生成恢复决策。规则或视觉 evaluator 用于判断执行证据，`TaskRunManager` 负责状态协调与持久化，而不是独自完成所有评估。

恢复动作可以包括重新观察、重试、重规划、停止和询问操作员。

## 5. 能力状态与目标任务

为避免把工程接入、软件在环结果和真机能力混为一谈，本文使用三种状态：

| 状态 | 含义 |
| --- | --- |
| 已实现 | 主执行路径存在，并有自动化测试或可重复的 bring-up 路径。 |
| 实验中 | 已有配置、adapter 或服务边界，但依赖额外模型、子模块、环境或真机验证。 |
| 计划评估 | 作为论文实验目标提出，尚不能据此报告成功率或泛化结论。 |

Hey Robot 的计划评估任务具有如下约束：

| 维度 | 目标 |
| --- | --- |
| 场景 | 半开放桌面或办公室场景 |
| 动态性 | 静态或缓慢变化 |
| 物体数量 | 1-5 个任务相关物体 |
| 技能长度 | <= 10 次技能调用 |
| 机器人 | 单台 XLeRobot 或对应仿真环境 |
| 交互 | 自然语言指令，可选修正或打断 |
| 交互体验 | 支持状态询问、目标纠偏、确认、取消和任务中断 |
| 成功检查 | 观测、状态或操作员验证 |

当前已实现状态覆盖 Harness 主链路、经典技能、交互与恢复机制；VLA/VLN 属于实验中状态；取水、桌面清理和多物体重排属于计划评估任务。全屋导航、任意开放世界操作、密集杂物清理、可变形物体操作、全身人形机器人行为和硬实时移动操作不在当前主张范围内。

## 6. 系统实现

系统使用 Python 3.12 实现，并通过 LeRobot 生态连接部分机械臂、相机和策略组件 [12]。代码按 `cognition/`、`skill_os/`、`foundation/`、`robot_runtime/`、`channels/`、`gateway/`、`episode/`、`events/` 和 `notifications/` 等边界组织。当前完整测试集超过 1,200 项；精确数量随迭代变化，不作为论文贡献本身。

### 6.1 部署模型

`DeploymentRunner` 根据 YAML 配置构建默认本地部署，在一个进程中启动 robot service、skill controller、task supervisor、agent service、gateway，以及可选的 human-follow service。各服务拥有独立 asyncio 生命周期和消息客户端。模型服务按需在独立进程中启动。配置通过 `{robot}.{env}.{os}.yaml` 约定区分真实机器人、仿真和开发测试部署。

每个 deployment 绑定一个 default agent 和 default robot。CLI、Web、Voice、Feishu 可以接入同一 Agent Runtime，但各通道是否启用取决于部署配置和外部凭据。

### 6.2 轮次生命周期与交互连续性

每个用户轮次经过固定状态机：

```text
restore → build → run → save
```

`restore` 阶段从 `TaskRunManager` 恢复任务状态、执行反馈和恢复上下文。`build` 阶段由 `MemoryBroker` 根据 task status 选择性组合记忆：active 任务注入完整 context（task state + recent scene evidence + relevant LTM），recovering 任务只注入 recovery state + last failure，completed 任务只注入 generic LTM。`run` 阶段由 `RobotAgentCore` 执行 LLM 推理和 skill 调用。`save` 阶段持久化检查点。

交互连续性由持久化 Goal、Gateway 的确定性安全路由和 confirmation interpreter 共同实现。机器人忙碌时，状态查询和安全打断可以绕过完整 LLM 推理；新目标不会注入正在执行的物理循环，而是要求在安全边界取消后以新合同创建。该路径在结构上减少了模型调用开销，但本文尚未获得足以支持固定延迟上界的基准数据。未解决 recovery 前，`block_actuation=True` 贯穿 Agent pipeline，阻止提交新的 actuation skill。

### 6.3 Skill OS 与当前部署表面

real/sim 主配置使用 `skills.mode: bringup`，默认启用 11 个非 VLA skill。它们适合硬件联调和系统验证，但同时包含语义能力与实现级 primitive，尚不是面向最终 Agent 的最小生产 skill surface。VLA 入口 `vla_manipulation` 已注册，但真机主配置没有对应 ModelService，也不把它加入 `skills.enabled`。实验配置 `xlerobot.sim.vla_vln.yaml` 单独声明 VLA/VLN 能力。

| Skill | 当前状态 | 说明 |
| --- | --- | --- |
| `inspect_scene` | enabled | 场景观察和描述 |
| `look_around` | enabled | 转动/扫描视野并观察 |
| `detect_marker` | enabled | 检测可见 marker |
| `move_base` | enabled | 底盘前进/后退 |
| `turn_base` | enabled | 底盘左转/右转 |
| `human_follow` | enabled | 视觉跟随人 |
| `stop_motion` | enabled | 停止运动 |
| `reset_posture` | enabled | 复位到安全姿态 |
| `set_arm_pose` | enabled | 机械臂命名姿态 |
| `move_arm_joints` | enabled | 机械臂关节控制 |
| `set_gripper` | enabled | 夹爪开合 |
| `vla_manipulation` | 实验中 | 主配置 disabled；实验配置尚未提供真实 `model_path`，仅用于接口联调 |

因此当前主配置可以验证感知、底盘、跟随、安全、机械臂和夹爪能力，以及 Harness 到 Robot Runtime 的完整消息链路。它不能直接证明开放词汇抓取、放置、交付或中程任务成功。后续生产 profile 应隐藏关节和夹爪 primitive，只暴露经过验证的 semantic skills。

`SkillContractRuntime` 在每个 skill 执行前检查：skill 存在性、必需参数、resource lock（arm/gripper/camera/base）、电量阈值、急停状态、readiness gate 和 ModelService availability。这相当于把 LLM 的动作提议放进确定性的安全和可行性门控。

### 6.4 ModelServices

VLA/VLN 等模型驱动能力通过独立 `ModelService` 暴露，使用 gRPC transport。当前 wire contract 是 `GetHealth`、`ExecuteSkill` 和 `CancelSkill`。Skill OS 负责契约、资源与结果处理，Robot Driver 专注硬件边界；ModelService 可以通过 deployment profile 启用、关闭或替换。

当前系统是 **foundation-ready，而不是 foundation-first 的已完成系统**：classic skills 是主配置的实际执行路径，VLA/VLN 是实验接入。实验配置中的 VLA 尚未使用真实 checkpoint；VLN 依赖 InternNav、模型权重和独立仿真环境。只有在这些依赖安装、路由一致并完成闭环评估后，才能报告 Foundation Model 的任务结果。

### 6.5 感知

相机观察由 `RobotService`、Robot Runtime 和感知组件共同管理。系统发布结构化 `robot.observation`，并通过 `robot.camera.frame.<robot_id>` 提供按机器人分区的 raw frame stream；感知技能、human follow 和 Foundation adapter 可以作为 consumer 使用观测。当前实现提供共享边界和 freshness 管理，但仍需在目标硬件上测量多 consumer 条件下的吞吐、延迟和相机资源竞争。

### 6.6 执行反馈与类型化恢复

技能完成不等于任务完成。`SkillResult` 到达后，service handler 调用 execution feedback evaluator，并将评估结果交给任务运行时和 `RecoveryManager`。当前恢复策略包括：

- `reobserve`：重新收集视觉证据再继续
- `reposition`：调整视角再 inspect
- `retry_with_adjustment`：带参数调整重试
- `ask_operator`：请求用户补充信息或授权
- `safe_abort`：停止任务，需要人工介入
- `degraded_continue`：非关键资源降级时继续

Recovery state 进入 `TaskSessionView`，对 UI 可见。未解决 recovery 前，`block_actuation=True` 阻止 Agent 提交新的 actuation skill。恢复策略目前是显式规则和 playbook，不是从真机失败数据自动学习得到的策略。

### 6.7 任务驾驶舱

运维视图 `TaskSessionView` 通过 `/tasks/{episode_id}` 页面展示聚合的任务状态、timeline、scene evidence 和 recovery；`/cockpit/{episode_id}` 继续作为兼容数据 API，`/cockpit` 页面入口重定向到 `/tasks`。当前代码中没有独立的 `request_quick_action` Agent 工具；用户、语音和 Web 入口仍通过 Gateway、Agent、`request_skill`、`SkillGateway` 和 SkillController 这条主链路提交机器人能力请求。

## 7. 实验设计

本节是待执行的实验协议，不是结果报告。在完成实验前，不应把“预期结果”写入摘要、贡献或结论作为事实。

### 7.1 Task Suite

实验分三阶段进行：

1. **软件在环阶段**：验证协议、任务检查点、忙碌交互、契约拒绝、故障注入和恢复状态机；该阶段只报告系统机制的正确性，不计入机器人任务成功率。
2. **经典技能真机阶段**：验证观察、底盘移动、转向、跟随、停止、复位、命名姿态、关节和夹爪等当前已启用能力，以及状态询问和安全打断。
3. **Foundation Model 闭环阶段**：仅在真实 checkpoint、模型服务、观测输入、动作执行和结果验证全部连通后，评估 VLA 操作与 VLN 导航。取水瓶、垃圾清理、多物体重排等任务属于这一阶段。

基础任务默认要求不超过 10 次 Agent 可见的 semantic skill 调用。复杂组合任务可以扩展到 10–15 次，但必须同时报告 task plan、subtask trace、semantic skill trace 及底层 primitive 数量，避免通过隐藏长时程 opaque policy 获得不可比较的结果。

### 7.2 Conditions

| 条件 | 说明 |
| --- | --- |
| C0 完整 Harness | 所有系统机制开启。 |
| C1 无主动感知 | 禁用新鲜度门控。 |
| C2 无技能契约 | 在软件在环实验中绕过契约门控。 |
| C3 无恢复机制 | 禁用恢复剧本。 |
| C4 无任务检查点 | 禁用持久任务检查点。 |
| C5 LLM 工具循环 | 普通工具调用基线。 |
| C6 无任务内交互机制 | 禁用对话行为分类、忙碌快路径和指代状态，只保留普通多轮历史。 |
| C7 classic-only | 禁用 VLA/VLN ModelService，只使用当前经典技能，衡量模型 backend 的增量贡献。 |

### 7.3 Metrics

主指标包括任务成功率、子目标成功率、技能成功率、平均 semantic skill 调用数、底层 primitive 数、平均任务时长、恢复成功率、组合成功率（CSR）和 Foundation skill 成功率（FSkSR）。软件在环、仿真和真机结果必须分表报告。

诊断指标包括感知刷新次数、契约拒绝次数、人工介入次数、交互响应延迟、纠偏成功率、打断成功率、失败类别分布、composition depth 和 opaque long-horizon policy calls。延迟应报告分位数和测量边界，不能用代码路径推断固定上界。

### 7.4 待验证假设

实验将检验以下假设，而不是预先假定其成立：

- H1：完整 Harness 在多技能任务上比普通 LLM 工具循环具有更高成功率和更低的不可行调用率。
- H2：主动感知降低由观测缺失或过期导致的失败。
- H3：技能契约降低参数、资源、readiness 和 backend 不满足导致的执行失败。
- H4：检查点与恢复机制提高故障注入后的恢复成功率。
- H5：忙碌交互路径降低状态查询和安全打断的响应延迟与误处理率。
- H6：接入经过验证的 VLA/VLN 后，系统可在不暴露长时程 opaque task policy 的前提下提高对应操作或导航子任务成功率。

## 8. 讨论

### 8.1 为什么系统结构重要

机器人可靠性不是单纯的模型能力问题。更强的 LLM / VLM 能提高推理和定位能力，但不能替代系统机制：

- 感知新鲜度
- 硬件就绪状态
- 任务状态
- 恢复策略
- 技能资源管理

这些机制必须作为系统结构存在，而不是临时写进 prompt。

### 8.2 与 COSA-like Agentic OS 的关系

Hey Robot 和 COSA-like Agentic OS 的共同点是：都强调连接认知、技能、记忆、感知、backend control 和执行的系统层。

区别在于，COSA 面向完整人形机器人及全身控制，Hey Robot 当前面向单台 XLeRobot，重点是 Harness、任务状态、Skill OS、主动感知、执行反馈和恢复机制。Foundation Model 在 Hey Robot 中是可替换的下层 backend，不是当前所有部署的必需条件。

因此，更准确的描述是：Hey Robot 是一个面向受约束中程 XLeRobot 任务、可接入 Foundation Model 的原生 Embodied Agent Harness。它与 Agentic OS 的分层思想一致，但不应直接等同于完整 Robot OS。

### 8.3 为什么交互体验是系统能力

对长期部署的机器人产品而言，交互体验不是前端附属功能。用户在真实环境中不会一次性给出完整、无歧义、永不变化的任务描述。系统必须把任务内语言关联到当前任务、活跃技能、最近场景证据和机器人状态。Hey Robot 因此把多轮对话看作系统问题，而不是只通过更长的聊天历史解决。当前实现对状态、打断、纠偏等常见表达采用规则分类；更完整的指代解析仍属于未来工作。

### 8.4 局限性

当前系统局限包括：

- 单机器人部署。
- 主配置采用 `bringup` mode，技能目录同时暴露 semantic skill 和实现级 primitive，尚未形成最小生产表面。
- VLA 实验配置尚未加载真实模型，当前只能用于接口联调，尚无可报告的真机 VLA 成功率。
- VLN 依赖 InternNav 子模块、模型权重和独立环境，尚未成为默认可复现实验。
- 现有 VLA semantic routing、真实观测注入和闭环执行仍需在真机 profile 中完成验证。
- 不解决密集杂物。
- 不解决任意物体抓取。
- 不提供 SLAM、全局地图或通用障碍物规划。
- “快系统”不是硬实时控制系统，NATS、gRPC 和 Python asyncio 路径没有硬实时保证。
- 恢复剧本仍是手工设计。
- 语义记忆还不是完整世界模型。
- 交互意图主要依靠关键词和任务状态，不追求开放域对话理解。
- 默认持久化使用本地 JSON/JSONL；默认 NATS core pub/sub 不提供完整的消息持久与重放语义。
- Web、NATS 和 gRPC 的认证、授权及 TLS 不是默认开启的生产安全基线。
- 真实机器人结果依赖机械臂、夹爪、相机、模型服务和标定的稳定性，目前不能从自动化单元测试外推真机成功率。

### 8.5 未来工作

近期工作包括收敛 production semantic skill surface、修正并验证 VLA/VLN 路由、让模型消费真实观测、建立可重复的仿真与真机基准，以及报告端到端延迟和失败分布。后续方向包括更丰富的物体与位置记忆、任务相关指代解析、从执行日志中学习恢复策略、物体级姿态估计、自动技能契约挖掘和更强的学习型操作策略。多机器人协作与完整开放世界记忆不是当前阶段的优先目标。

## 9. 结论

Hey Robot 实现了一个不依赖通用 LLM Agent 框架、面向真实机器人的原生 Embodied Agent Harness。系统以异步消息连接上层慢速认知与下层短时域执行，并用四层架构划分 Agent/Cognition、Skill OS、Foundation Model 和 Robot Runtime 的责任。当前代码已实现主动感知、任务检查点、技能契约、执行反馈、恢复和忙碌交互等系统机制。

当前主配置仍以经典 bring-up 技能为执行主体，VLA/VLN 接入处于实验阶段，取水瓶、桌面清理和多物体重排尚是计划评估任务。因此本文现阶段能够成立的结论是：Hey Robot 已提供可调度、可观察、可恢复的系统骨架；Foundation Model 对复杂真机任务的增益，以及整个 Harness 相对普通 LLM 工具循环的效果，必须由后续受控实验给出。

## References

1. Ahn, M. et al. "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances." CoRL, 2022.
2. Brohan, A. et al. "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control." arXiv:2307.15818, 2023.
3. Driess, D. et al. "PaLM-E: An Embodied Multimodal Language Model." ICML, 2023.
4. Huang, W. et al. "Inner Monologue: Embodied Reasoning through Planning with Language Models." CoRL, 2022.
5. Liang, J. et al. "Code as Policies: Language Model Programs for Embodied Control." ICRA, 2023.
6. Quigley, M. et al. "ROS: an open-source Robot Operating System." ICRA Workshop, 2009.
7. Yao, S. et al. "ReAct: Synergizing Reasoning and Acting with Language Models." ICLR, 2023.
8. Colledanchise, M. and Ogren, P. "Behavior Trees in Robotics and AI." CRC Press, 2018.
9. LimX Dynamics. "LimX COSA, the First-of-Its-Kind Agentic OS for Humanoid Robots." Official product note, 2026. <https://www.limxdynamics.com/en/news/BK000055>
10. Wei, M. et al. "Ground Slow, Move Fast: A Dual-System Foundation Model for Generalizable Vision-Language Navigation." ICLR, 2026. <https://openreview.net/forum?id=GK4rznYwhn>
11. InternNav Team. "InternVLA-N1: An Open Dual-System Navigation Foundation Model with Learned Latent Plans." Technical Report, 2025. <https://arxiv.org/abs/2512.08186>
12. Cadene, R. et al. "LeRobot: An Open-Source Library for End-to-End Robot Learning." ICLR, 2026. <https://arxiv.org/abs/2602.22818>
