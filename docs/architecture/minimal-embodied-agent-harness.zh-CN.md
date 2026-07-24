# Hey Robot 最小化 Embodied Agent Harness 设计与演进路线

当前阶段聚焦 Tool/Skill 边界和 VLA 能力封装的文件级改动、测试及验收标准见
[《Hey Robot Tool/Skill 边界与 VLA 能力增量重构方案》](minimal-harness-refactoring-plan.zh-CN.md)。

## 1. 文档目的

Hey Robot 的总体方向是把机器人系统设计为：

1. 一个围绕基础模型组织观察、推理、工具调用、执行反馈、持久化和评估的
   **Embodied Agent Harness**；
2. 一个由高层慢速推理和低层快速执行组成的**快慢双系统**。

这两个方向值得保留，但它们不意味着项目初期就必须实现完整的分布式机器人平台。
本文件用于明确哪些是长期稳定的架构原则，哪些能力可以延后，并给出一条从最小实验闭环
逐步演进到真机系统的路线。

核心结论是：

> 保留分层边界，简化早期实现。Tool 是 Agent 可调用能力的统一接口；其中机器人
> Capability Tool 由 Skill/Capability 自动投影，普通 Harness Tool 则直接实现自己的
> 非物理能力。Implementation 再区分 VLA、WAM 和传统算法。

## 2. 研究依据

### 2.1 Harness Engineering for Self-Improvement

《Harness Engineering for Self-Improvement》把 Harness 定义为基础模型与真实环境之间的
部署和运行系统。它负责模型如何：

- 观察和管理上下文；
- 推理、规划和调用工具；
- 执行工作流并根据结果继续迭代；
- 保存轨迹、日志和其他持久化 artifact；
- 进行权限控制、结果验证和评估；
- 从成功与失败经验中逐步改进上下文、工作流乃至 Harness 本身。

这说明 Hey Robot 不应只是一个“LLM 调机器人 API”的薄封装。Agent loop、执行反馈、
持久化轨迹、权限和评估都属于其核心职责。

但论文同时强调 Harness 应当保持简单和通用。Harness 的价值来自稳定、清晰、可观察、
可评价的闭环，而不是组件或服务数量。自我改进也应建立在可靠 evaluator 和持久化轨迹
之上，而不是在项目初期提前构建庞大的自演化系统。

参考：
[Harness Engineering for Self-Improvement](../references/Harness%20Engineering%20for%20Self-Improvement.md)。

### 2.2 Hi-VLA

《What Matters in Orchestrating Robot Policies》使用类似 Options Framework 的统一控制循环
描述 Hierarchical VLA：

```text
高层 VLM
  观察当前状态和历史
  生成一个短时域语言子目标
          |
          v
低层 VLA / Policy
  在子目标约束下连续产生机器人动作
          |
          v
终止条件满足
  控制权返回高层 VLM
```

论文的消融实验表明，系统效果主要由以下因素共同决定：

- 高层 VLM 的推理能力；
- 低层 VLA 的动作质量和语言可控性；
- 低层 option 的终止条件；
- 提供给高层的观察表示；
- 跨 episode 经验的提取和使用方式。

最佳层次化系统在论文实验中的长时域和推理任务上显著优于 Flat VLA；但 Naive Hierarchy
与精心设计的 Hierarchy 之间也存在明显差距。这意味着“保留层次”是正确方向，同时也
意味着项目早期应该优先实验上述关键变量，而不是优先增加外围功能。

参考：[Hi-VLA](../references/Hi-VLA.md)。

## 3. 架构判断

### 3.1 保留的长期原则

Hey Robot 应长期保留以下原则：

1. Agent 不直接生成和下发关节或电机动作；
2. 高层 Agent 负责语义理解、子目标生成、恢复和长时域任务推进；
3. 低层 Policy/Skill 负责有界、短时域、可终止的机器人执行；
4. 观察、执行结果和证据必须返回 Agent，形成闭环；
5. 同一个语义能力可以由 VLA、WAM、传统算法或混合方法实现；
6. Robot Runtime 统一隔离具体 embodiment、仿真和真机；
7. 失败轨迹、成功轨迹和评估结果必须可以持久化和复现。

### 3.2 可以简化的部分

代码边界不等于部署边界。项目早期没有必要把 Agent、Skill、Model Service 和
Robot Runtime 全部部署为独立服务。

以下能力都可以在实验需要出现后再引入：

- 多进程消息总线；
- 独立 Skill controller 服务；
- 多 Agent；
- 多 channel 和复杂通知；
- 完整 Dashboard；
- 跨进程 durable worker 恢复；
- 多种机器人和仿真环境同时接入；
- 自动 Harness evolution；
- 复杂的 memory 管理系统。

对于新项目，早期可以在单进程中保持模块接口，通过直接函数调用完成闭环。Hey Robot 当前
已经具备 NATS、gRPC、Skill Worker 和持久化执行边界；它们不是本轮 VLA 增量重构的拆除
对象。只有新增模块才应避免在没有实验或部署需求时继续扩大分布式复杂度。

## 4. Tool、Skill 与 Capability 的统一定义

### 4.1 Tool 是 Agent 的统一动作空间

Tool 表示 Agent 可以调用的一切外部能力，不局限于机器人动作：

```text
Agent Tools
  |- Harness Tools
  |    |- read_file
  |    |- write_file
  |    |- search_memory
  |    |- complete_task
  |    `- request_human_help
  |
  `- Robot Capability Tools
       |- inspect_scene
       |- pick
       |- place
       `- navigate_to
```

普通 Harness Tool 可以直接调用受权限约束的 handler；Robot Capability Tool 只生成一个
物理能力提案，并进入 Skill Runtime 的校验、调度、安全和执行生命周期。因此，并非所有
Tool 都是 Skill，但每个需要暴露给 Agent 的 Skill 都可以投影为 Tool。

两类 Tool 共享面向模型的名称、描述和 JSON Schema，但执行路径不同：

```text
Agent Tool Call
       |
       +-- HarnessToolCall ------> permission ------> handler ------> ToolResult
       |
       `-- SkillCallProposal ----> skill policy ----> Skill Runtime -> SkillResult
```

这里的普通 Tool 也不是无条件执行。文件、网络、数据库和进程 Tool 仍然需要 workspace、
路径、网络域名、权限和超时限制，只是不需要机器人资源锁、动作终止条件和急停生命周期。

### 4.2 Skill/Capability 是机器人能力的唯一事实源

系统内部只维护一份语义能力定义：

```python
CapabilitySpec(
    name="pick",
    description="Pick up a visible object",
    input_schema={...},
    output_schema={...},
    resources=("robot_control", "camera"),
    timeout_sec=30,
    termination=...,
    evidence=...,
)
```

这里的 Skill 与 Capability 表达同一个概念：一个具有前置条件、执行过程、终止条件和
结果证据的 temporally extended action。为避免术语和代码重复，新代码宜逐步统一使用
一个名称；在完成术语迁移前，两者应被理解为同一层。

### 4.3 Robot Capability Tool 是 Agent-facing projection

机器人 Tool 不再拥有独立的机器人业务定义。它只把 CapabilitySpec 投影为模型可以调用的
function schema。这个约束不适用于 `read_file`、`complete_task` 等没有对应机器人 Skill 的
普通 Harness Tool：

```text
CapabilitySpec
    name + description + parameters
                 |
                 v 自动生成
Agent Tool Schema
```

因此下面两条信息不能分别维护：

- Tool 的参数 schema；
- Skill 的参数 schema。

Hey Robot 当前 `ToolRegistry` 从 Skill catalog 创建 `SkillTool` 的方向是正确的。后续应继续
删除机器人 Tool 与 Skill 之间重复的名称映射、参数模型和注册流程，使 CapabilitySpec
成为机器人能力的唯一事实源。

当前 `extra_tools` 已为普通 Tool 留出扩展入口，但 `ToolRegistry.proposal()` 只接受
`SkillCallProposal`、`CompleteTaskProposal` 和 `ControlTaskProposal`。重构若要支持文件、
记忆、检索或其他 LLM-based Agent Tool，需要增加通用的非物理 Tool 调用类型和 dispatcher，
而不是把这些 Tool 包装成 Skill。

### 4.4 Implementation 承担多实现差异

同一个 Capability 可以注册多个实现：

```text
pick
  |- pick.vla
  |- pick.wam
  |- pick.classic
  `- pick.hybrid
```

实现选择可以由部署配置、机器人能力、模型健康状态或实验策略决定，但 Agent 默认只看到
`pick`，不应该被迫知道底层使用 Pi0.5、ACT、WAM 或传统视觉伺服。

建议的实现接口是：

```python
class CapabilityImplementation(Protocol):
    id: str
    capability: str

    def supports(self, robot, runtime) -> bool: ...
    async def execute(self, context, arguments) -> CapabilityResult: ...
```

### 4.5 参数所有权

Tool/Skill 分层不能阻止 Agent 为 Skill 提供参数。CapabilitySpec 中公开的参数会成为 Tool
schema，模型产生的 arguments 经校验后原样进入 Skill handler。应按权限划分参数，而不是
按层级切断参数：

| 参数类型 | 示例 | 决定方 |
|---|---|---|
| 语义任务参数 | `object`、`target`、`instruction` | Agent |
| 有界执行参数 | `max_steps`、`max_attempts` | Agent 在 schema 安全范围内决定 |
| 实现选择参数 | `vla`、`wam`、`classic` | 部署或实验配置，默认不交给 Agent |
| 低层安全参数 | 关节限位、action scale、控制频率 | Skill Implementation / Robot Runtime |
| 运行上下文 | observation、run id、deadline | Harness 注入 |

正式语义 Tool 应使用明确的参数、上下限和 `additionalProperties: false`。只有受控的
primitive 或 model-specific 实验 profile 才允许更开放的模型参数。

## 5. Agent 应看到什么粒度的 Tool

### 5.1 过低的抽象

如果 Agent 默认看到：

```text
set_joint_angle
move_xyz
rotate_wrist
set_gripper
```

高层模型就会承担运动规划和执行细节，破坏快慢系统边界。这类 primitive 可以用于硬件
bring-up、调试和受控实验，但不应成为正式 Agent 的默认能力面。

### 5.2 过高的抽象

如果 Agent 只能调用：

```text
do_everything(task="整理整个房间")
```

低层 option 会过长，失败难以定位，观察和重新规划频率也不足。这相当于重新退化为
Flat VLA 或一个不可观察的子 Agent。

### 5.3 推荐粒度

语义能力应当是短时域、有界、可观察、可终止、可恢复的 option，例如：

```text
inspect_scene
approach_object
pick
place
open_drawer
navigate_to
manipulate_subgoal
stop
```

其中 `manipulate_subgoal` 可以作为 VLA 的通用语言子目标入口，但必须具有动作步数、时间、
成功检测或重新观察等终止边界，不能无限运行到整个用户任务结束。

## 6. Robot Capability 的暴露模式

Hey Robot 可以同时支持正式运行和研究实验，但通过 profile 隔离：

| Profile | Agent 可见能力 | 主要用途 |
|---|---|---|
| `semantic` | `pick`、`place`、`navigate_to` | 默认运行、跨模型、跨机器人 |
| `primitive` | `move_xyz`、`set_gripper` | bring-up、调试、控制实验 |
| `model-specific` | `pi05_rollout`、`wam_rollout` | 模型研究、消融实验 |

默认 profile 必须是 `semantic`。另外两种模式需要显式开启，并继续经过最小安全检查、
超时和停止机制。

RPent 当前把 `move_to`、`rotate_wrist`、`release`、`pi0_pick` 等直接暴露给 Agent，适合快速
开展特定 benchmark 实验，但会把具体模型和控制细节泄漏到高层策略。Hey Robot 可以吸收
这种实验效率，而不把它作为长期稳定的 Agent API。

## 7. 最小通用架构

逻辑上的最小主链只需要四个概念。下图描述职责而非强制部署拓扑：这些模块既可以位于
一个进程，也可以继续使用 Hey Robot 当前的 NATS、gRPC 和 Skill Worker 分布式部署。
Agent Tool Registry 同时组合普通 Harness Tool 和由 Capability 自动投影的机器人 Tool：

```text
+-------------------------------------+
| Agent Loop                          |
| reason -> call tool -> observe      |
+------------------+------------------+
                   |
+------------------v------------------+
| Agent Tool Registry                 |
| harness tools + capability tools    |
+---------+-------------------+-------+
          |                   |
          | direct handler    | SkillCallProposal
          v                   v
+-----------------+  +----------------+
| Harness Tool    |  | Capability     |
| Handler         |  | Registry       |
+-----------------+  +--------+-------+
                            |
                            | select implementation
                   +--------v----------+
                   | Capability Impl.  |
                   | VLA/WAM/Classic   |
                   +--------+----------+
                            | bounded actions
                   +--------v----------+
                   | Robot Runtime     |
                   | observe/act/stop  |
                   +-------------------+
```

### 7.1 最小 Agent Tool

第一版只需要：

```text
inspect_scene(question)
execute_subgoal(instruction)
complete_task(evidence_ids, recap)
stop(reason)
```

第一阶段可以只实现 `execute_subgoal.vla`。当实验需要比较传统控制或 WAM 时，再新增实现，
不改变 Agent-facing schema。

### 7.2 最小 Robot Runtime

```python
class RobotRuntime(Protocol):
    async def observe(self) -> Observation: ...
    async def execute(self, actions) -> ExecutionResult: ...
    async def stop(self, reason: str) -> None: ...
    async def reset(self) -> Observation: ...
```

第一版只支持一个仿真机器人。能力、观察和结果的数据结构保持通用，但不要提前实现所有
robot family 的字段和适配器。

### 7.3 最小 Harness 状态

每次决策写一条 JSONL 即可：

```json
{
  "task_id": "task-001",
  "step": 3,
  "observation": {"frame_id": 18},
  "agent_decision": "pick up the red cup",
  "capability": "execute_subgoal",
  "implementation": "pi05",
  "termination_reason": "success_detected",
  "success": true,
  "artifacts": ["images/frame-18.png"]
}
```

这已经满足最初的 Harness 需求：闭环、artifact、可观察、可回放和可评估。SQLite、事件总线
和 durable worker 应在 JSONL 无法满足真实需求后再加入。

## 8. 优先实验的问题

项目早期应围绕可证伪的架构假设开展实验。

### 8.1 Hierarchy 是否有效

比较：

- Flat VLA：用户完整指令直接交给 VLA；
- Naive Hierarchy：Agent 生成 subgoal，固定步数切换；
- Improved Hierarchy：结构化观察、成功检测和跨 episode 经验。

分别测量短时域、长时域和推理任务成功率。

### 8.2 Termination

比较：

- 固定 action steps；
- 固定时间；
- success detector；
- 混合策略：最短执行窗口 + success detector + 最大时间。

记录误报、漏报、重复动作和超时。Hi-VLA 表明 termination 是连接高层和低层的高杠杆机制，
应当先于复杂调度系统进行验证。

### 8.3 Observation representation

比较：

- raw image；
- scene caption；
- bounding box/object list；
- 图像加机器人 proprioception；
- 仿真 privileged state，仅作为对照上限。

### 8.4 Capability granularity

比较：

- 一个通用 `execute_subgoal`；
- `pick/place/open` 等语义 option；
- 直接 primitive tools。

评估成功率、Agent 决策次数、token、恢复次数、低层动作长度以及换模型后的 prompt 改动量。

### 8.5 多实现是否产生价值

先只选择一个能力，例如 `pick`，实现：

- `pick.vla`；
- `pick.classic`；
- `pick.hybrid`。

如果多实现没有带来可测量收益，就不应提前为所有能力建设复杂的策略路由系统。

## 9. 分阶段演进路线

### 阶段 0：最小闭环

范围：

- 一个仿真环境；
- 一个 Agent；
- 一个 VLA；
- 四个 Agent Tool；
- 单进程直接调用；
- JSONL、图片和视频 artifact；
- 明确的任务成功指标。

验收标准：能够稳定完成“观察—生成 subgoal—执行—重新观察—完成判断”的闭环。

### 阶段 1：Hi-VLA 消融实验

依次验证 planner、subgoal 粒度、termination、observation representation 和 memory。每项改动
必须有可重复的 benchmark 和对照组。

验收标准：层次化系统在长时域或推理任务上显著优于 Flat VLA，并能定位提升来源。

### 阶段 2：多实现 Capability

只对已经验证有价值的能力增加 VLA、WAM、传统或 hybrid 实现，并保持 Agent Tool schema
不变。

验收标准：同一 Agent 任务无需修改 prompt，即可切换实现并得到可比较结果。

### 阶段 3：真机与可靠性

在进入真机前逐步增加：

- action bounds 和 readiness；
- timeout、cancel 和 emergency stop；
- 资源互斥；
- durable command receipt 和幂等；
- 独立 Model Service；
- 必要的进程隔离。

验收标准：任何模型调用失败、超时或进程重启都不会无意重复物理动作，并且可以安全停止。

### 阶段 4：Harness 改进闭环

在积累足够轨迹后增加：

- 成功与失败轨迹归档；
- 跨 episode affordance 总结；
- prompt、termination 和 context 策略候选；
- held-out evaluation；
- 人工审核关键修改；
- 只接受在验证集上产生稳定收益的 Harness 更新。

Evaluator 和权限控制必须位于被优化 Harness 之外，避免 reward hacking 和测试集过拟合。

## 10. 当前阶段暂缓清单

在核心实验尚未证明价值前，不把以下内容作为主线目标：

- 同时完善全部 Web、CLI、语音和飞书能力；
- 多 Agent 协作；
- 通用分布式 Skill OS；
- 所有 foundation model 的统一路由；
- 所有机器人 embodiment；
- 自动修改生产 Harness；
- 复杂知识图谱或无限增长的 memory；
- 没有 benchmark 驱动的抽象和配置项。

这些能力可以保留已有代码，但新设计不应继续围绕它们扩张；默认开发和文档路径应聚焦最小
实验闭环。

## 11. 架构决策摘要

1. Hey Robot 继续定位为 Embodied Agent Harness，而不是机器人 API 聚合器；
2. 继续使用快慢双系统，高层产生 subgoal，低层执行短时域 option；
3. Tool 是 Agent 的统一动作空间，可以是普通 Harness Tool，也可以是 Robot Capability Tool；
4. Capability/Skill 是机器人能力的唯一事实源；
5. Robot Capability Tool 由 CapabilitySpec 自动生成，不重复定义机器人业务；
6. VLA、WAM、传统算法属于 Capability Implementation；
7. 默认 Agent 只看到语义能力，primitive 和 model-specific tool 仅用于受控实验；
8. 代码保持清晰边界；新实验可以简单组合，但不强制拆除已有分布式部署；
9. 优先建设 benchmark、termination、observation、trace 和 evaluator；
10. 以实验结果驱动新功能和新抽象；
11. 真机可靠性和 Harness 自我改进在最小闭环得到验证后逐步加入。

## 12. 重构约束与验收标准

### 12.1 重构时必须保持的约束

1. Agent Runner 只解析模型响应和产生类型化 Tool Call，不直接执行 IO；
2. 普通 Tool 和 Robot Capability Tool 共用一个 Agent-facing registry；
3. 普通 Tool 进入受权限约束的通用 dispatcher，不进入 Skill Runtime；
4. Robot Capability Tool 只产生 `SkillCallProposal`，不得直接生成 `RobotAction`；
5. CapabilitySpec 同时提供 Agent schema 和 Skill 参数校验，不维护两份 schema；
6. Capability Implementation 不影响 Agent-facing tool name；
7. Robot Runtime 不依赖 Agent、Tool Registry 或具体模型 provider；
8. primitive profile 必须显式开启，不能成为默认 Agent 工具面；
9. 所有物理执行必须有 timeout、cancel/stop 和结构化结果；
10. 每次 Tool Call、Skill 执行和终止原因必须可追踪。

### 12.2 建议的最小类型

```python
ToolCall = HarnessToolCall | SkillCallProposal | CompleteTaskProposal | ControlTaskProposal

class AgentTool(Protocol):
    name: str
    schema: dict
    def proposal(self, arguments: dict) -> ToolCall: ...

class HarnessTool(AgentTool, Protocol):
    async def execute(self, context, arguments: dict) -> ToolResult: ...

class CapabilitySpec:
    name: str
    description: str
    parameters: dict
```

这些类型表达必要边界即可。早期不要求每一种 proposal 都有独立服务、消息主题或持久化表。

### 12.3 完成一次重构切片的验收标准

每次重构应至少证明：

- 新增普通 Tool 不需要创建 Skill；
- 新增 Robot Skill 不需要重复编写 Tool schema；
- 同一个 `pick` Tool 可以在不修改 Agent prompt 的情况下切换 classic/VLA/hybrid；
- 非法 Tool 参数在任何 IO 或机器人动作前被拒绝；
- 普通 Tool 不能绕过其权限策略；
- Robot Capability Tool 不能绕过 Skill Runtime 和 Robot Runtime；
- 最小闭环可在现有部署中运行；是否增加单进程实验入口由实际实验需求决定；
- trace 能区分普通 Tool result、Skill lifecycle 和 Robot action result；
- 对应架构边界有自动化测试，而不只依赖文档约定。
