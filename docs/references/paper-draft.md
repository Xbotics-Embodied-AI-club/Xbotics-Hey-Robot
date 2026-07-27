# Hey Robot：面向交互式长程机器人任务的 Embodied Agent Harness

**Hey Robot: An Embodied Agent Harness for Interactive Long-Horizon Robot Tasks**

<p align="center"><em>Embodied Agent Harness · Fast–Slow Dual System · Distributed Model Services</em></p>

> 系统论文初稿 · 2026-07-27
>
> 本文描述 Hey Robot 的目标问题、系统设计、当前实现和评估协议。文中区分：
> **implemented**（代码和测试已存在）、**integrated**（模型/环境已接入）、
> **conditionally validated**（在明确配置和条件下有成功 artifact）以及
> **to be validated**（尚待真机或系统性实验验证）。RoboCasa365 的单条件成功不被外推为
> 通用真机泛化或开放世界长程性能。

## 摘要

交互式机器人任务要求系统同时处理开放语言目标、视觉观测、物理动作、用户纠偏和长程
任务连续性。单次 Tool Calling 或端到端策略通常缺少持久任务状态、动作唯一性、资源互斥、
终止语义和崩溃恢复机制。本文提出 Hey Robot，一个面向真实机器人部署的 Embodied Agent
Harness。

Hey Robot 是一个分布式 Embodied Agent Harness，并采用快慢双系统。慢系统由 Agent、任务状态
和交互通道组成，负责理解目标、选择 Skill、处理用户反馈并基于执行结果继续决策。快系统由
有界 Skill、观测管线、Robot Runtime 和 driver 组成，负责在明确预算和安全边界内执行物理动作。
InternNav、VLA/VLN 和 LeRobot policy 通过 typed ModelService/gRPC contract 与 Agent 主系统
分割，可运行在独立进程、依赖环境和 GPU 上。两套系统通过 typed proposal、持久 task step、
Skill event 和结构化 outcome 连接，而不是依赖模型直接生成低层控制流。

当前实现提供单 Agent、单会话开放任务、persist-before-submit、幂等事件归并、启动恢复、
安全控制路径以及配置驱动的部署组装。InternNav 已接入 XLeRobot MuJoCo 仿真；LeRobot
policy 已通过统一 ModelService 接入 RoboCasa365，并在记录的 checkpoint、任务和 seed
上完成官方成功链路。下一阶段将验证 InternNav 和 LeRobot policy 在 XLeRobot 真机上的
观测、动作映射、取消、超时和安全闭环。

本文的研究重点不是提出新的 VLA 或 planner，而是研究一个最小、可观察、可恢复的 Harness
如何组织异构机器人能力，并如何将 Harness 的收益与模型本身的能力分离评估。

## 1. 引言

机器人系统正在从单一指令执行转向开放式交互。用户可能先提出一个多步骤目标，再在执行
过程中改变目标、补充约束或要求暂停。机器人还必须面对观测过期、执行延迟、共享资源、
动作不可随意回滚和进程重启。由此，机器人 Agent 的核心问题不仅是“下一步动作是什么”，
还包括：

1. 当前任务的唯一事实源在哪里？
2. 一个物理动作何时被接受、执行、完成或丢失？
3. 用户纠偏在什么安全边界生效？
4. Skill 结果如何进入下一次决策？
5. 进程重启后如何恢复任务而不盲目重放物理动作？

本文将这些问题归入 Harness 层。Harness 是围绕模型组织工具、上下文、工作流、持久状态、
权限、反馈和评估的系统，而不是另一个更大的 planner。对于机器人，Harness 还必须明确
慢系统的认知职责与快系统的物理执行职责。

### 1.1 研究问题

本文研究以下问题：

- 一个最小 Harness 能否支持跨多个 Skill 的持续任务？
- 用户在物理执行期间的纠偏能否在安全点影响后续决策？
- persist-before-submit、唯一 run 和幂等事件能否降低重启后的重复动作风险？
- 同一个 Agent-facing Skill contract 能否承载经典控制、InternNav 和 LeRobot policy？
- Harness 的任务连续性收益能否与 planner、VLA/VLN 和 embodiment 能力分开测量？

### 1.2 贡献

本文和开源实现的贡献包括：

1. 一个面向交互式长程机器人任务的最小 Harness 抽象；
2. 一个将 Agent proposal、Skill execution、Robot Runtime 和 ModelService 分离的实现；
3. 面向任务事实、执行事实和证据的持久化与恢复语义；
4. InternNav 仿真接入和 LeRobot/RoboCasa365 完整链路的条件化验证；
5. 一套用于比较交互、恢复、终止、观测和模型能力的分层评估协议。

## 2. 问题形式化

设用户目标为 (g)，在时刻 (t) 的系统状态为：

\[
s_t = (c_t, q_t, o_t, r_t, e_t),
\]

其中 (c_t) 是 canonical conversation，(q_t) 是持久任务状态，(o_t) 是机器人观测，
(r_t) 是当前 Skill run，(e_t) 是事件和证据记录。

慢系统根据上下文产生一个 proposal：

\[
p_t = \Pi(g, c_t, q_t, o_t),
\]

其中 (p_t) 要么是结构化用户回复，要么是至多一个物理 Skill 调用。快系统在预算 (B_t)、
资源约束 (R_t) 和安全门 (S_t) 下执行：

\[
(o_{t+1}, y_t, e_{t+1}) = \Omega(p_t, o_t, B_t, R_t, S_t).
\]

执行结果 (y_t) 不自动等价于完整任务成功。只有任务完成谓词、环境终态或明确的可信证据
支持时，任务才可以进入 completed 状态。

## 3. 系统设计

### 3.1 总体架构

```text
Channel / user event
        ↓
Gateway / identity / episode
        ↓
AutonomousAgentService
        ↓
AgentContextBuilder → AgentRunner → ToolRegistry
        ↓ typed proposal
AgentToolExecutor → TaskCoordinator → SkillClient
        ↓ SkillCommand
SkillWorker → Skill / OptionRunner → LocalRobotClient
        ↓
RobotRuntime → observation / safety / driver
        └──── structured SkillEvent / ToolOutcome ────→ Agent
```

AgentRunner 是无外部 I/O 的模型决策边界。它只接受规范化 ModelMessage 和已允许的 tool
schema，最多返回一个 proposal。AgentToolExecutor 将 proposal 分类为用户回复、普通 Harness
Tool 或物理 Skill。TaskCoordinator 负责提交前持久化、唯一 run、事件应用和任务续跑。

### 3.2 快慢双系统

慢系统维护目标、上下文、用户纠偏和任务生命周期。快系统执行短时域、有界、可取消的 Skill。
Agent、Skill Worker 和 Robot Runtime 可由 deployment profile 组合运行；InternNav、VLA/VLN
和 LeRobot policy 则以独立 ModelService 通过 gRPC 接入主系统。“快”和“慢”描述职责与
决策时域，不承诺硬实时保证。

慢系统不直接构造 driver action，快系统不扩展开放式用户目标。所有物理动作必须通过 Skill
surface、Robot Runtime 和安全门。

### 3.3 任务与恢复语义

一个 session 最多拥有一个开放任务。每个 task step 记录 proposal、run_id、tool_call_id、
结果、状态、序号和 evidence id。Skill submission 遵循：

```text
validate → persist pending step → submit once → apply ordered events
```

当服务重启后，仍处于非终态但已失去 worker 所有权的 run 被标记为 `execution_lost`。系统
恢复任务事实和最新结果，重新交给 Agent 判断，但不自动重放未知物理动作。

### 3.4 交互纠偏

物理 Skill 执行期间，用户输入被记录到 canonical conversation，并在安全边界等待当前操作
终止。Skill terminal event 到达后，Agent 使用最新对话、任务投影和 Skill outcome 重新决策。
该设计牺牲了直接打断正在运行的低层 policy 的即时性，换取物理动作时序的可解释性和安全性。

### 3.5 配置驱动与能力面

typed deployment config 选择：

- Channel 和身份绑定；
- Robot、driver 和 embodiment profile；
- ModelService endpoint、capability 和资源参数；
- `skills.tools` 暴露给 Agent 的能力；
- bus、runtime、media 和 episode 路径。

配置只负责组装和有界参数，不保存任务进度或运行状态。模块之间依赖小型 contract，替换
backend 应尽量只改变配置和 leaf implementation。

## 4. 模型与环境集成

### 4.1 InternNav

InternNav 以独立 VLN ModelService 接入 XLeRobot MuJoCo 仿真。服务使用目标 checkpoint、
图像观测、历史帧和导航指令，返回 pixel goal、heading 或 stop 等结构化导航结果。VLN
option 在 `max_steps` 内执行 observe-plan-act，移动动作最终经过 Robot Runtime。

当前状态：XLeRobot MuJoCo 仿真链路已接入并完成模型服务、观测转换和移动动作映射验证。
下一阶段需要在 XLeRobot 真机验证相机标定、视场、图像路径、离散动作步长、底盘响应和急停。

### 4.2 LeRobot policy 与 RoboCasa365

LeRobot policy 通过统一 RobotPolicy ModelService 运行在独立推理进程中。Hey Robot 负责
Agent/Skill、observation mapping、policy request、动作归一化、Robot Runtime 和环境交互；
RoboCasa backend 负责环境、frame、step 和官方成功谓词。

`configs/evaluation/robocasa365.yaml` 使用 `lerobot/pi052_robocasa`、12D action 和
明确的 camera/state mapping。当前已有一条条件化成功 artifact：记录的 CloseFridge
任务、target split、seed 1000 和固定环境条件下，官方成功谓词为 true，所有 action 均通过
Robot Runtime。

该结果证明完整集成链路在指定条件下可运行，不证明真机 embodiment 迁移、跨任务泛化或开放
世界长程任务成功率。

### 4.3 XLeRobot 真机迁移

XLeRobot 真机已有 native driver、底盘、SO101 机械臂、相机、观测和安全路径。当前默认
真机 profile 只开放场景检查和底盘基础动作。InternNav 与 LeRobot policy 的下一步验证包括：

1. 相机和 observation feature 对齐；
2. action dimensions、范围、频率和控制周期；
3. policy timeout、cancel、emergency stop；
4. 空载和低速动作安全；
5. 真实任务中的新观测和失败恢复；
6. 与 MuJoCo/RoboCasa 结果的差异归因。

## 5. 系统不变量与安全

核心不变量包括：

- 一份 canonical conversation；
- 一份 session-scoped task truth；
- 一个物理 run 的唯一提交身份；
- 一个 deployment 的唯一 Skill dispatch 路径；
- 物理动作串行占用共享资源；
- 普通文本不能隐式完成任务；
- 未知执行结果不能触发隐式重放；
- emergency stop 不依赖模型响应。

这些机制是软件安全防线，不等同于碰撞规避、功能安全认证或工业安全合规。

## 6. 评估协议

### 6.1 分层验证

| 层级 | 场景 | 目标 |
| --- | --- | --- |
| L0 | schema、配置、边界和事件 | 验证静态 contract |
| L1 | Mock / deterministic Skill | 验证交互、steer、失败回流 |
| L2 | MuJoCo + InternNav | 验证模型服务和导航闭环 |
| L3 | RoboCasa365 + LeRobot policy | 验证完整 policy/environment 链路 |
| L4 | XLeRobot bring-up | 验证真实观测、底盘、机械臂和安全 |
| L5 | XLeRobot + InternNav/LeRobot | 验证真机模型闭环 |
| L6 | 多步骤长程任务 | 验证任务连续性、恢复和用户纠偏 |

### 6.2 Golden Path

最小 Harness 的核心验收场景为三步任务：

1. 用户提交需要三个 Skill 的目标；
2. 第二步执行期间用户发送纠正；
3. 纠正影响第三步；
4. 另一轮测试在第二步制造进程中断；
5. 重启后将未知执行标记为 lost，不重放原动作；
6. Agent 基于事实继续或向用户请求确认。

该场景用于验证 Harness 机制，不能被 VLA 或 RoboCasa benchmark 的单次成功替代。

### 6.3 对照与指标

实验应分离以下变量：

- flat policy 与 Harness loop；
- 无持久任务与持久任务；
- 无用户纠偏与 safe-boundary steer；
- 无结构化 outcome 与结构化 outcome；
- 不同 termination、observation 和低层 policy；
- 仿真、RoboCasa 和 XLeRobot 真机 embodiment。

主要指标包括：任务完成率、子目标进展、错误完成率、重复动作率、恢复成功率、纠偏延迟、
任务持续时间、模型调用次数、Skill 延迟、急停响应和 evidence 完整性。

## 7. 当前结果与局限

当前可报告的事实是：

- Harness 主链和任务状态机制已有组件与集成测试；
- InternNav 已完成 XLeRobot MuJoCo 模型服务链路接入；
- LeRobot policy 已完成统一 ModelService 接入；
- RoboCasa365 已有明确条件下的官方成功 artifact。

当前不能报告的结论包括：

- XLeRobot 真机上的 InternNav 或 LeRobot policy 成功率；
- 开放世界家务或通用长程任务成功率；
- Harness 相对于更强模型或 flat policy 的统计显著收益；
- 通用碰撞规避、工业安全认证或跨 embodiment 自动迁移。

已知工程缺口还包括：VLA `no_action` / `max_steps` 的未知子目标不能被误判为任务成功，
动作后 observation 需要完整投影到下一轮 Agent context，Agent 层流式增量回调需要在完整
服务路径打通。这些属于现有 contract 的正确性和验证工作，不是新增产品功能。

## 8. 相关工作与定位

Hi Robot 展示了开放语言目标与 situated feedback 的层级处理；Hi-VLA 系统研究比较了
planner、VLA、termination、observation 和 memory；Harness VLA 展示了冻结 VLA 与固定
primitive 的组合；pi-agent-core 提供了清晰的 model/tool loop、steer/follow-up 和上下文
边界参考；RPent 展示了 Planner/Toolkit、逐步新观测和实验 artifact 的直接闭环。

Hey Robot 不复制这些项目的完整 runtime 或 benchmark-specific primitive，而是吸收其
边界原则，面向长期在线机器人服务增加任务持久化、物理资源约束、保守恢复和多 deployment
配置。

## 9. 结论

Hey Robot 将机器人 Agent 从单次工具调用提升为一个可交互、可持续、可恢复的 Embodied Agent
Harness。快慢双系统让慢系统负责目标和用户关系，让快系统负责有界物理执行；typed proposal、
持久 task step、Skill event 和 Robot Runtime 则将模型能力与物理事实分开。

当前实现已经完成基础 Harness、InternNav 仿真接入和 RoboCasa365 LeRobot policy 条件化
验证。下一阶段不是继续扩大架构，而是将已有模型接入 XLeRobot 真机，建立真实观测、动作、
安全和恢复的证据，并以分层对照实验测量 Harness 对交互式长程任务的实际贡献。

## References

1. Shi, L. X. et al. “Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models.” 2024.
2. Hu, J. et al. “What Matters in Orchestrating Robot Policies: A Systematic Study of Hierarchical VLA Agents.” arXiv:2606.10267, 2026.
3. Zhang, Y. et al. “Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents.” arXiv:2607.08448, 2026.
4. Weng, L. “Harness Engineering for Self-Improvement.” Lil’Log, 2026.
5. Brohan, A. et al. “RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control.” 2023.
6. Kim, M. J. et al. “OpenVLA: An Open-Source Vision-Language-Action Model.” arXiv:2406.09246, 2024.
7. Cadene, R. et al. “LeRobot: An Open-Source Library for End-to-End Robot Learning.” ICLR, 2026.
8. InternVLA-N1 / InternNav project materials. See `third_party/InternNav` and the VLN deployment guide.
