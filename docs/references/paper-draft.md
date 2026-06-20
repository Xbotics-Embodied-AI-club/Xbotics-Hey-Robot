# Hey Robot：面向半开放中程具身任务的交互式 Agent 运行时

## 摘要

大语言模型智能体（LLM agents）正在被用于把自然语言指令连接到机器人技能。问题在于，普通的“LLM + 工具调用”循环并不天然适合真实机器人：观测会过期，动作可行性依赖硬件状态，任务进度需要跨轮次持久化，失败技能需要结构化恢复，用户还会在机器人执行过程中追问、修正、打断或重新安排优先级，而不是等待一个单轮工具调用结束。

本文提出 **Hey Robot**，一个面向半开放室内操作任务的 foundation-first 交互式具身 Agent 运行时（interactive embodied Agent runtime）。系统目标不是开放世界人形机器人自主，而是在 XLeRobot 这类小型机器人平台上，通过一套稳定的 `semantic skill` 表面和 foundation backend，完成中程任务，例如取水瓶、清理桌面垃圾、把物体移动到目标位置，并在任务过程中支持自然的多轮交互、状态询问、实时纠偏和安全打断。

Hey Robot 将通道入口、对话运行时、任务运行时、认知工具执行、semantic Skill OS、foundation backend、感知、恢复和机器人驱动划分为清晰的运行时边界，并通过类型化消息协议连接。核心机制包括主动感知新鲜度检查、确定性技能契约门控、持久任务检查点、执行反馈、类型化恢复剧本、backend resolution，以及面向机器人忙碌状态的多轮交互处理。系统不把复杂任务交给单个 opaque VLA policy，而是通过有限数量的 reliable short-horizon foundation skills 进行层级组合。本文计划在 mock XLeRobot 和真实 XLeRobot 上评估系统，并与普通 LLM 工具调用循环、classic backend 和多个消融条件对比。

## 1. 引言

LLM-based agents 让自然语言任务转换为工具调用变得非常容易。在数字环境中，这种模式通常有效，因为状态显式、动作可回滚、失败重试成本低。

真实机器人不同。机器人可能在 LLM 推理期间已经移动；相机图像可能在规划时已经过期；夹爪显示关闭并不等于真正抓住物体；用户可能在任务中途插话、修正、询问状态或打断；一个技能完成也不等于整个任务完成。

因此，把 ReAct-style “thought -> action -> observation” 循环直接套到机器人上，会产生系统性问题。本文聚焦一个更现实的任务范围：半开放、中程、桌面和办公室操作任务。例如：

- “把前面桌子上的水瓶拿给我。”
- “清理桌面，把垃圾扔到垃圾桶。”
- “把杯子放到桌上，把木块放到架子上。”

这些任务不是完全开放世界任务。它们有明确约束：工作空间较小，任务相关物体通常为 1-5 个，技能预算不超过 10，场景可以有轻微变化但不是高动态环境，成功需要通过观测、机器人状态或操作员标签验证。

Hey Robot 的核心观点是：LLM 循环只是具身系统中的一个认知组件。真正让机器人可靠完成中程任务的，是围绕它的运行时机制：任务状态、主动感知、技能契约、执行反馈、恢复机制和交互连续性。对产品化机器人而言，用户体验不只来自最终成功率，还来自机器人能否清楚表达当前状态、接住用户修正、在忙碌时快速响应，并在失败时给出可理解的恢复路径。

### 贡献

本文贡献包括：

1. 总结中程具身任务中 LLM 驱动机器人智能体的主要失败模式。
2. 实现一个面向服务的三层 Agentic runtime，将认知、任务状态、感知、semantic Skill OS、foundation backend 和机器人执行分离。
3. 设计 semantic skill contract 与 backend resolution 机制，在物理执行前检查必需参数、资源、硬件就绪状态、backend availability 和安全状态。
4. 引入交互运行时视角，支持任务中的状态询问、纠偏、确认、打断和继续执行。
5. 提出 mock + 真实 XLeRobot 实验协议，用于评估 10 次 semantic skill 调用以内的半开放中程任务、多轮交互能力，以及有限 reliable foundation skills 的组合能力。

## 2. 与 Agentic Robot OS 的关系

LimX COSA 等系统提出了一个重要方向：具身机器人需要 Agentic OS，而不是单个大模型。这样的系统需要管理认知、技能、记忆、感知和运动。

Hey Robot 与这个方向一致，但范围更窄。COSA 面向完整人形机器人自主，包括全身控制、移动操作、类小脑基础模型、实时运动生成和全尺寸人形机器人部署。Hey Robot 当前目标是一个 COSA-aligned XLeRobot runtime：上层强调任务理解、对话连续性、主动感知和恢复；中层强调 semantic Skill OS、技能契约、调度和组合；下层通过 VLA capability service 以及未来 foundation services 连接真实机器人执行。

Hey Robot 不主张解决完整人形小脑基础模型。它更像是小型机器人平台上的 COSA-aligned 三层运行时，重点是：

- LLM 推理
- 场景证据
- 技能契约
- 任务状态
- 执行反馈
- 类型化恢复
- 多轮交互和忙碌状态处理
- semantic Skill OS
- foundation backend execution boundary

因此，本文不是“实现一个完整人形 COSA”，而是研究：如何在小型真实机器人上，通过具身 Agent 运行时和 foundation backend，让语言模型可靠调用并组合一套稳定的 semantic skills，完成半开放中程任务。复杂任务的泛化来自 task/subtask/semantic skill 的层级组合，而不是让单个 VLA 或其他 foundation policy 执行整段 user task。

## 3. 失败模式

本文将中程具身任务中的主要失败归纳为六类。

### F1：感知缺失或过期

Agent 使用旧观测做规划。例如上一帧图像中水瓶在桌面中央，但实际已经被移动。更强的 VLM 不能单独解决这个问题，因为问题不在“看不懂”，而在“看的不是当前状态”。

运行时层面的解决方案是主动感知新鲜度门控：在视觉规划前检查观测时间、图像数量和新鲜度，必要时触发 `inspect_scene`。

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

计划是合理的，但执行失败。例如夹爪没夹住、物体滑落、机械臂姿态不合适、底盘未对准，或 VLA 策略执行失败。运行时必须区分子目标失败和任务失败，并决定重试、重新观察、重规划、停止或询问操作员。

### F5：任务状态漂移

多步骤任务中，Agent 可能忘记已完成哪些子目标、哪个物体被移动到哪里、哪个技能失败过、恢复是否已经尝试过。这不能只靠更长上下文窗口稳定解决，需要持久任务检查点和任务局部记忆。

### F6：交互状态断裂

用户在机器人执行中途可能说“不是这个”“先停一下”“你现在在干什么”“继续刚才那个”。如果系统只把这些输入当成新的独立聊天轮次，就会丢失当前任务、活跃技能、指代对象和机器人忙碌状态，导致错误打断、重复执行或无法恢复。因此需要显式的对话状态、指代记忆和忙碌轮次处理机制。

## 4. 系统架构

Hey Robot 是一个面向服务的运行时。组件之间通过类型化消息协议通信：

```text
User Channel
  -> GatewayService
  -> UserTurn
  -> RobotAgentService
      -> RobotAgentRuntimeContainer
      -> AgentTurnSessions
      -> RobotAgentLoop
      -> TaskRunManager
      -> SceneRuntime
      -> MemoryBroker
      -> RobotAgentCore
          -> AgentRuntime
          -> request_capability / request_perception
          -> SkillGateway
          -> SkillIntent
  -> SkillControllerService
      -> SkillContractRuntime
      -> CapabilityRuntime
  -> RobotService / RobotRuntime / CapabilityService
  -> RobotObservation / RobotStatus / SkillResult
```

### 4.1 协议边界

核心消息类型包括 `UserTurn`、`AgentReply`、`RobotObservation`、`RobotStatus`、`SkillIntent`、`SkillEvent` 和 `SkillResult`。`RobotAction` 属于 Robot/driver 侧内部执行边界，不作为 Agent 层提交动作的接口。每条消息带有共享的 `Envelope`，包含 trace、episode、channel、agent、robot 和部署元数据，以保持跨通道、跨服务的任务连续性。

### 4.2 Gateway

`GatewayService` 将 CLI、Web、Voice、Feishu 等输入统一成 `UserTurn`。它负责路由用户消息、分配 episode、追加对话历史，并把 `AgentReply` 送回原通道。Teleop 可以作为未来通道扩展，但不是当前论文主线。

### 4.3 Agent 运行时

`RobotAgentService` 是 Agent 侧服务壳，订阅用户轮次、机器人状态、机器人观测、技能事件和技能结果。复杂职责拆分给：

- `RobotAgentLoop`：轮次生命周期状态机，负责 restore → build → run → save 流程。
- `TaskRunManager`：持久任务状态、检查点、执行反馈和恢复上下文。
- `MemoryBroker`：统一记忆路由，根据 task state（active / recovering / completed）选择性组合 task memory、scene evidence 和 LTM。
- `SceneRuntime`：场景记忆、场景证据查询和主动感知门控。
- `RobotAgentCore`：LLM / 工具执行与 skill-level 决策。
- `AgentTurnPolicy` / `RobotTurnPolicy`：轮次类型分类（新任务 / 追问 / 纠偏 / 打断 / 确认）、忙碌门控和 `block_actuation` 策略。
- `BusyTurnHandler`：机器人忙时处理用户输入的低延迟路径。
- `AgentNotificationRuntime`：任务进度和恢复通知。

### 4.4 轮次生命周期

每个用户轮次经过固定状态机：

```text
restore -> build -> run -> save
```

意义在于：真实机器人任务不是一次性聊天。系统必须恢复任务状态，构建最新上下文，运行认知核心，然后保存执行结果。

### 4.4.1 交互连续性

Hey Robot 将对话状态与任务状态绑定，而不是只保存聊天历史。一个用户输入会先被解释为新任务、追问状态、纠偏、打断、确认、取消或普通对话。机器人忙碌时，状态询问和安全打断走低延迟路径；纠偏和任务改派会进入任务检查点或待处理队列。这样系统能够在中程任务内处理自然的多轮交互，而不是把每句话都当作新的独立任务。

### 4.5 主动感知

在视觉相关任务中，`SceneRuntime` 会检查最近观测是否缺失、没有图像、已经过期或与当前任务不相关。如果需要刷新，Agent 会触发 `inspect_scene`，然后把新的场景证据注入当前轮次上下文。

### 4.6 技能契约

`SkillContractRuntime` 根据 semantic skill catalog 和单一 implementation 检查每个技能请求：

- 技能是否存在且对 Agent 可见（semantic skill）。
- 必需参数是否齐全。
- 必需资源是否可用（arm、gripper、camera、base）。
- 机器人状态是否允许执行（电量、急停、硬件就绪、readiness gate）。
- 是否与正在执行的技能发生资源冲突。
- capability service 是否可用（如 VLA foundation backend）。

这相当于把 LLM 的动作提议放进确定性的安全和可行性门控。

### 4.7 执行反馈和恢复

技能完成不等于任务完成。系统在 `SkillResult` 到达后会更新任务状态、评估执行反馈、判断子目标是否成功、判断任务是否需要恢复，并在必要时触发类型化恢复剧本。

恢复动作可以包括重新观察、重试、重规划、停止和询问操作员。

## 5. 目标任务类别

Hey Robot 当前目标任务具有如下约束：

| 维度 | 目标 |
| --- | --- |
| 场景 | 半开放桌面或办公室场景 |
| 动态性 | 静态或缓慢变化 |
| 物体数量 | 1-5 个任务相关物体 |
| 技能长度 | <= 10 次技能调用 |
| 机器人 | 单台 XLeRobot 或 mock XLeRobot |
| 交互 | 自然语言指令，可选修正或打断 |
| 交互体验 | 支持状态询问、目标纠偏、确认、取消和任务中断 |
| 成功检查 | 观测、状态或操作员验证 |

不在当前主张范围内的任务包括全屋导航、任意开放世界操作、密集杂物清理、可变形物体操作、全身人形机器人行为和实时移动操作。

## 6. 系统实现

系统使用 Python 3.12 实现，约 1100+ 测试用例，按可部署服务边界组织为以下模块：`agents/`、`skills/`、`robots/`、`capability/`、`memory/`、`perception/`、`tasks/`、`channels/`、`monitoring/`。

### 6.1 部署模型

`DeploymentRunner` 根据 YAML 配置构建本地部署，在一个进程中启动所有启用的服务：robot service、skill controller、task supervisor、agent service、gateway，以及可选的 human-follow service。配置按 `{robot}.{env}.{os}.yaml` 约定命名，通过 `environment: real|sim|mock` 区分部署环境。

每个 deployment 绑定一个 default agent 和一个 default robot，同时通过 CLI、Web、Voice、Feishu 多个 channel 接入同一套 Agent Runtime。

### 6.2 轮次生命周期与交互连续性

每个用户轮次经过固定状态机：

```text
restore → build → run → save
```

`restore` 阶段从 `TaskRunManager` 恢复任务状态、执行反馈和恢复上下文。`build` 阶段由 `MemoryBroker` 根据 task status 选择性组合记忆：active 任务注入完整 context（task state + recent scene evidence + relevant LTM），recovering 任务只注入 recovery state + last failure，completed 任务只注入 generic LTM。`run` 阶段由 `RobotAgentCore` 执行 LLM 推理和 skill 调用。`save` 阶段持久化检查点。

交互连续性通过 `AgentTurnPolicy` 实现：用户输入被分类为新任务、追问状态、纠偏、打断、确认或普通对话。机器人忙碌时，状态询问和安全打断走 `BusyTurnHandler` 低延迟路径（响应延迟 <= 2s）；纠偏和任务改派进入任务检查点或待处理队列。未解决 recovery 前，`block_actuation=True` 贯穿整个 pipeline，阻止 Agent 提交新的 actuation skill。

### 6.3 Semantic Skill OS

当前 XLeRobot 代码中的 skill surface 已先收敛到真实可验证能力。real/sim 配置默认启用 11 个非 VLA skill；VLA 入口 `vla_manipulation` 已注册，并可通过独立 capability service 部署，但默认不加入 `skills.enabled`。

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
| `vla_manipulation` | registered, disabled | VLA 自然语言机械臂操作入口 |

因此当前系统可以验证感知、底盘、跟随、安全、机械臂和夹爪原子能力；暂不把 VLA 抓取、放置、交付等自然语言操作任务暴露给 Agent。后续 VLA 稳定后，`vla_manipulation` 可以加入 deployment 的 `skills.enabled`。

`SkillContractRuntime` 在每个 skill 执行前检查：skill 存在性、必需参数、resource lock（arm/gripper/camera/base）、电量阈值、急停状态、readiness gate 和 capability service availability。这相当于把 LLM 的动作提议放进确定性的安全和可行性门控。

### 6.4 Capability Services

VLA 等长时间运行、模型驱动的能力通过独立的 `capability_service` 暴露，使用 gRPC transport。当前 `arm_vla` capability service 封装 LeRobot policy server，通过 `ExecuteCapability` RPC 调用。机器人驱动专注硬件执行边界，capability services 通过 deployment profile 启用、关闭或替换。

`classic backend` 在本文中只作为硬件 bring-up、fallback 和 ablation baseline（实验条件 C7）。最终系统呈现以 foundation backend 为主。

### 6.5 感知

相机观察由 `RobotService / RobotRuntime / PerceptionService` 共同管理。系统发布结构化 `robot_observation`，并通过 `robot.camera.frame` 提供 raw frame stream；RobotRuntime、perception skill、human follow 和 VLA adapter 都可以作为 consumer 读取，各自控制读取频率和 freshness，避免相机所有权冲突。

### 6.6 执行反馈与类型化恢复

技能完成不等于任务完成。`SkillResult` 到达后，`TaskRunManager` 评估执行反馈、判断子目标成功或失败、决定是否需要恢复。当前恢复类型包括：

- `reobserve`：重新收集视觉证据再继续
- `reposition`：调整视角再 inspect
- `retry_with_adjustment`：带参数调整重试
- `ask_operator`：请求用户补充信息或授权
- `safe_abort`：停止任务，需要人工介入
- `degraded_continue`：非关键资源降级时继续

Recovery state 进入 `TaskSessionView`，对 UI 可见。未解决 recovery 前，`block_actuation=True` 阻止 Agent 提交新的 actuation skill。

### 6.7 任务驾驶舱

产品化视图 `TaskSessionView` 通过 `/cockpit` API 提供聚合的任务状态、timeline、scene evidence 和 recovery 信息。当前代码中没有独立的 `request_quick_action` Agent 工具；用户、语音和 Web 入口仍通过 Gateway、Agent、`request_capability`、`SkillGateway` 和 SkillController 这条主链路提交机器人能力请求。

## 7. 实验设计

### 7.1 Task Suite

实验包含基础 mock 任务、复杂组合 mock 任务和更窄的真实机器人任务。Mock 任务覆盖感知、单物体操作、取物 / 放置、双物体重排、清理、故障注入恢复，以及层级复杂任务。真实机器人任务更窄，主要验证取水瓶、简单垃圾清理和受控交互。

基础成功任务默认要求 semantic skill 调用数 <= 10。复杂组合任务可以扩展到 10-15 次 semantic skill 调用，但必须提供 task plan、subtask trace 和 semantic skill trace。

### 7.2 Conditions

| 条件 | 说明 |
| --- | --- |
| C0 完整运行时 | 所有运行时机制开启。 |
| C1 无主动感知 | 禁用新鲜度门控。 |
| C2 无技能契约 | 在 mock 中绕过契约门控。 |
| C3 无恢复机制 | 禁用恢复剧本。 |
| C4 无任务检查点 | 禁用持久任务检查点。 |
| C5 LLM 工具循环 | 普通工具调用基线。 |
| C6 无交互运行时 | 禁用对话行为分类、忙碌快路径和指代状态，只保留普通多轮历史。 |

### 7.3 Metrics

主指标包括任务成功率、子目标成功率、技能成功率、平均 semantic skill 调用数、平均任务时长、恢复成功率、组合成功率（CSR）和 foundation skill 成功率（FSkSR）。

诊断指标包括感知刷新次数、契约拒绝次数、人工介入次数、交互响应延迟、纠偏成功率、打断成功率、失败类别分布、composition depth 和 opaque foundation task calls。后者目标必须为 0。

### 7.4 Expected Results

预期结果：

- 完整运行时在超过 4 个技能调用的任务上优于 LLM 工具循环。
- 复杂任务的成功率来自较高 CSR 和 FSkSR，而不是 opaque foundation task policy。
- 主动感知主要降低感知过期导致的失败。
- 技能契约主要降低物理不可行失败。
- 恢复机制对夹爪失败、相机失败、物体被移动等任务最重要。
- 任务检查点在被打断任务和多物体任务中贡献最大。
- 交互运行时在忙碌状态追问、目标纠偏和安全打断场景中降低误处理率，并提升用户感知到的可控性。

## 8. 讨论

### 8.1 为什么运行时结构重要

机器人可靠性不是单纯的模型能力问题。更强的 LLM / VLM 能提高推理和定位能力，但不能替代运行时机制：

- 感知新鲜度
- 硬件就绪状态
- 任务状态
- 恢复策略
- 技能资源管理

这些机制必须作为系统结构存在，而不是临时写进 prompt。

### 8.2 与 COSA-like Agentic OS 的关系

Hey Robot 和 COSA-like Agentic OS 的共同点是：都认为机器人不能只靠单模型，必须有管理认知、技能、记忆、感知、backend control 和执行的系统层。

区别在于，COSA 面向完整人形机器人自主和全身控制，而 Hey Robot 面向 XLeRobot 上的 foundation-first 三层 runtime。COSA 强调类小脑基础模型和 whole-body control，Hey Robot 强调 semantic Skill OS、VLA capability backend、任务状态、主动感知、执行反馈和恢复机制。

因此，Hey Robot 可以描述为：COSA-aligned foundation backend runtime for constrained medium-horizon XLeRobot tasks。

### 8.3 为什么交互体验是系统能力

对长期部署的机器人产品而言，交互体验不是前端附属功能。用户在真实环境中不会一次性给出完整、无歧义、永不变化的任务描述。系统必须理解“这个”“那个”“继续刚才的”“先停一下”等任务内语言，并把这些话映射到当前任务、活跃技能、最近场景证据和机器人状态。Hey Robot 因此把多轮对话看作运行时问题，而不是只通过更长的聊天历史解决。

### 8.4 局限性

当前系统局限包括：

- 单机器人部署。
- 技能目录有限。
- 不解决密集杂物。
- 不解决任意物体抓取。
- 清理任务只覆盖少量简单垃圾物体。
- 恢复剧本仍是手工设计。
- 语义记忆还不是完整世界模型。
- 交互状态目前主要覆盖任务相关对话，不追求开放域闲聊。
- 真实机器人成功率强依赖机械臂、夹爪、相机、VLA capability service 和后续 foundation services 的稳定性。

### 8.5 未来工作

未来工作包括更丰富的物体和位置记忆、任务相关指代解析、从执行日志中学习恢复策略、物体级姿态估计、自动技能契约挖掘、交互式任务控制台，以及更强的学习型操作策略。多机器人协作和完整开放世界记忆不是当前阶段的优先目标。

## 9. 结论

Hey Robot 提出了一个面向半开放中程机器人任务的交互式具身 Agent 运行时。它的主张是克制但实用的：当机器人任务需要组合多个已落地技能时，可靠性不仅取决于 LLM 规划，还取决于主动感知、持久任务状态、技能契约、执行反馈、类型化恢复和多轮交互连续性。

通过在 XLeRobot / mock XLeRobot 上评估取水瓶、桌面清理、多物体重排、任务中断和目标纠偏等任务，本文希望证明：从 LLM 工具调用 demo 走向可靠的多步骤具身行为，需要一个明确的运行时层；从可运行系统走向产品，还需要把交互状态作为一等公民。

## References

1. Ahn, M. et al. "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances." CoRL, 2022.
2. Brohan, A. et al. "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control." arXiv:2307.15818, 2023.
3. Driess, D. et al. "PaLM-E: An Embodied Multimodal Language Model." ICML, 2023.
4. Huang, W. et al. "Inner Monologue: Embodied Reasoning through Planning with Language Models." CoRL, 2022.
5. Liang, J. et al. "Code as Policies: Language Model Programs for Embodied Control." ICRA, 2023.
6. Quigley, M. et al. "ROS: an open-source Robot Operating System." ICRA Workshop, 2009.
7. Yao, S. et al. "ReAct: Synergizing Reasoning and Acting with Language Models." ICLR, 2023.
8. Colledanchise, M. and Ogren, P. "Behavior Trees in Robotics and AI." CRC Press, 2018.
