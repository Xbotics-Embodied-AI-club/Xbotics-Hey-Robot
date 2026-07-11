# Hey Robot 最小自主具身 Agent 重构方案

> 文档状态：准备实施  
> 版本：最小自主内核版  
> 更新日期：2026-07-12  
> Hey Robot 代码基线：`1c7d01b67ca28e91a6681e824833aca8871b3906`（main；分析以实际代码为准）  
> 目标：以尽可能少的机制验证长期目标、事件唤醒、有限推理、物理执行和真实反馈能否组成一个清晰的自主具身 Agent。  
> 文档角色：**唯一实施规范（source of truth）**。重构范围、协议、阶段、删除项和验收均以本文为准。

## 0. 文档定位

本文统一了三类输入：

1. 当前 Hey Robot 实际代码中的职责混合、fallback、兼容和同步物理闭环问题；
2. docs/nanobot-agent-design-reference-analysis.zh-CN.md 中适合具身 Agent 的设计经验；
3. Hey Robot 作为研究系统的首要原则：允许任务失败，但不允许掩盖失败。

nanobot 文档只保留参考代码、测试证据和设计来源，不是第二份实施计划。两份文档出现差异时，以本文为准。

本方案不追求功能齐全。判断一个组件是否进入第一版，只问三个问题：

~~~text
它是否让一次决策和物理结果更容易追踪？
它是否阻止不确定动作被重复执行？
它是否是自主闭环成立的必要条件？
~~~

三个答案都是否定的组件，不进入第一版。

### 0.1 nanobot 到底迁移什么

| nanobot 设计 | 对自主具身 Agent 的价值 | 本方案处理 |
|---|---|---|
| AgentLoop / Runner 分层 | 高 | 迁移“生命周期与单次模型运行分离”的边界 |
| request receipt、checkpoint、事件 trace | 高 | 迁移幂等记录和可观察性，但崩溃后不自动续跑模型或物理动作 |
| 显式 Tool registry 和窄依赖注入 | 高 | 迁移，并把生产模型工具缩减为两个 |
| session/context 组织 | 中 | 只保留当前 Goal 的结构化 action/evidence，不保留聊天历史修复 |
| sustained goal 的长期运行思想 | 中 | 重新设计为 GoalStore + Supervisor + SkillResult 事件唤醒 |
| memory/RAG/consolidation | 低，且当前会干扰实验 | 第一版不迁移；EvidenceLedger 是任务证据，不是长期记忆 |
| provider fallback、context repair、内部 continuation | 负价值 | 删除，不迁移 |
| 通用 subagent、playbook、动态工具 | 当前低价值 | 后置，不能进入最小内核 |

最关键的重新设计是：nanobot 的数字 Tool 可以在 Runner 内直接执行；Hey Robot 的物理 Tool 只能产生 ActionProposal，真实 SkillIntent 由 Supervisor 在持久状态和安全边界内提交。

## 1. 核心原则

~~~text
Fail explicitly.
Stop on uncertainty.
Never hide failure.
One external intent per deliberation.
At most one model request per deliberation.
No compatibility path.
No automatic recovery.
~~~

具体含义：

- 模型、协议、工具、技能、证据或持久化失败时，保存结构化失败；
- 物理动作是否执行无法确定时进入 BLOCKED，不重新发送；
- 不从自然语言回复、错误文本或最后一个工具结果猜测状态；
- 一次 deliberation 最多提出一个主动观测或物理技能，Supervisor 最多提交一个；
- 一次 deliberation 最多调用模型一次；合同已满足时可以零次；
- SkillResult 到达前不进行下一次物理决策；
- 不保留新旧双运行路径、legacy_mode、deprecated alias 或长期 facade；
- 旧配置不由主运行时兼容；需要保留时只允许离线一次性转换，主运行时只接受当前 schema；
- 第一版不做任何自动重试、自动 reobserve 或隐藏 continuation。

这里要区分“补丁”和“研究基础设施”。

应删除的补丁：

- provider fallback；
- 文本 tool protocol 兼容；
- orphan/missing tool result 自动修复；
- 自动 final answer 合成；
- 失败后切换另一条感知或执行路径；
- max iterations 后强行生成正常回复；
- 旧字段、旧 tool 名称和旧配置别名；
- 通过 prompt 文本推断恢复策略。

必须保留的研究基础设施：

- 结构化失败；
- immutable TaskContract；
- deliberation_id 幂等；
- ActionLedger；
- RobotExecutionGate；
- DeliberationStore 幂等 receipt；
- SkillCommandStore；
- 安全 gate；
- 硬预算停止；
- 完整 trace。

这些机制不会提高表面成功率。它们的作用是让失败位置和物理不确定性更清楚。

## 2. 第一版明确不做

- 多机器人；
- 多个并行 Goal；
- Agent 自己创建长期 Goal；
- Goal pause/resume；
- timer、周期任务和定时唤醒；
- 普通 mid-turn correction/follow-up 注入；
- 自动 REOBSERVE；
- 自动重试任何技能；
- 自动重发 DeliberationRequest 或 SkillIntent；
- Agent 进程崩溃后的模型调用续跑；
- 审批和确认闭环；
- Agent 长期记忆读写；
- search_memory 和 write_memory；
- RAG、embedding 和向量数据库；
- history consolidation；
- Dream、自修改和自动修改 Skill；
- Agent/Task Playbook；
- 通用 subagent；
- ProgressDetector 和无进展启发式控制；
- LLM 生成 Goal 最终报告；
- 动态 MCP/Tool 安装；
- 用 LLM 决定安全策略；
- 无人值守真机长时间运行；
- 为旧接口保留兼容层。

安全策略拒绝动作时，第一版直接返回结构化失败。用户需要修改目标或策略时，先取消旧 Goal，再创建新 Goal。

第一版只支持：

~~~text
一个机器人
一个非 terminal Goal
GoalCommand(create/cancel)
两个模型工具
一次一个 deliberation
一次最多一个外部 intent
失败默认停止
状态未知默认 BLOCKED
UNKNOWN 跨 Goal 锁住新的物理执行
完整 trace
~~~

## 3. 当前代码的主要问题

### 3.1 AgentRuntime 职责过多

src/hey_robot/cognition/runtime/runner.py 当前同时处理：

- provider 调用；
- 工具循环和并发；
- TaskContract；
- EvidenceLedger；
- perception grounding；
- 工具结果 fallback；
- final answer 推断；
- max-iteration finalization；
- 内部协议文本识别；
- execution feedback 文本解析。

严格 Runner 应只完成一次模型决策，并报告模型和工具实际发生了什么，不应解释任务是否完成。

### 3.2 RobotAgentCore 包含补救和展示逻辑

src/hey_robot/cognition/core.py 当前还负责：

- 请求和等待技能；
- 解析 execution feedback；
- 检查最近失败；
- 猜测最终文本是否表示完成；
- 从工具结果生成回复；
- 为未完成 turn 生成 fallback。

Core 应收敛为依赖组装和一次调用。

### 3.3 TaskRunManager 是聚合点

src/hey_robot/cognition/task_runtime.py 当前同时拥有：

- TaskRun；
- checkpoint；
- pending turn；
- Scene Memory；
- RobotState；
- execution feedback；
- recovery；
- continuation context；
- skill trace。

目标归属：

| 当前职责 | 最终归属 |
|---|---|
| TaskContract / task state | cognition/task/ |
| TaskEvaluator / Evidence | cognition/policy/task_evaluator.py |
| model/tool checkpoint | cognition/runtime/deliberation_store.py |
| skill trace / execution feedback | ActionLedger + SkillResultHandler |
| 最新 RobotStatus/Observation | Supervisor snapshot index |
| continuation context | 删除 |
| recovery tree | 删除 |
| pending correction | 第一版删除 |
| Scene Memory | 不进入自主控制路径 |

最终删除 TaskRunManager 聚合入口，不建立同名代理对象。

### 3.4 当前 autonomy 不是长期自主

src/hey_robot/cognition/autonomy.py 只在进程内保存目标和事件，没有：

- 持久化 GoalStore；
- 事件唤醒；
- Goal 状态机；
- deliberation 幂等；
- 动作对账；
- 预算；
- 重启恢复。

### 3.5 当前物理闭环阻塞在一个 turn 内

request_skill(wait_result) 会同步等待 SkillResult，模型可能在一个 turn 内连续请求多个技能。

结果是：

- turn 生命周期过长；
- 多个物理动作共享一个模型上下文；
- checkpoint 边界模糊；
- 进程重启后难以判断动作是否已经提交；
- 实验难以归因某一步为何失败。

### 3.6 当前跨服务协议还不够严格

src/hey_robot/protocol/messages.py 的 from_payload() 当前只对 Envelope 和少数 list 元素做特殊恢复，其余嵌套 union、tuple 和 dataclass 会原样留下；未知字段也不会被显式拒绝。现有 RobotObservation 没有主动观察动作的来源 skill_id，SkillResult 则已经有 skill_id、frame_id 和 observations。

因此第一版采用一个清晰规则：

- 普通 robot.observation 只更新最新快照；
- 主动观察以 inspect_scene 的 terminal SkillResult 为唯一完成事件；
- SkillResult.evidence 增加类型化 EvidenceFact；
- 新控制消息严格反序列化，不依赖 metadata 猜关联。

### 3.7 当前 Skill OS 幂等只覆盖内存 active run

src/hey_robot/skill_os/scheduler.py 的 SkillScheduler.add() 只检查当前 runs 字典；controller.py 在终态会 remove(skill_id)。Controller 重启或 run 结束后，再收到相同 skill_id，现有代码仍可能重新执行。

这说明“Supervisor 不重发”还不够。物理执行消费端必须持久化 command receipt，否则消息重复、服务重启和人工重放仍可能制造第二次动作。

## 4. 最小目标架构

~~~text
GoalCommand(create/cancel)
          |
          v
AutonomySupervisor
  AutonomyStore / ActionLedger / RobotExecutionGate / policy / Snapshot index
          |
          | DeliberationRequest
          v
RobotAgentService
  ContextBuilder
  StrictAgentRunner
  TaskEvaluator
          |
          +-- request_observation
          |
          +-- request_skill
                    |
                    v
             ActionProposal
                    |
                    v
          AutonomySupervisor
                    |
               SkillGateway
                    |
          ActionLedger PERSISTED
                    |
              skill.intent
                    |
                    v
Skill OS -> Foundation services -> Robot Runtime
                    |
               SkillResult
                    |
                    v
          AutonomySupervisor
~~~

这里有一个刻意的所有权约束：RobotAgentService 只产生一个类型化 ActionProposal，不直接发布 SkillIntent。Supervisor 在收到提案后重新检查 Goal 状态、termination_reason、硬预算和 RobotExecutionGate，再通过 SkillGateway 提交。这样停止中的 Goal 不会在模型返回较晚时继续发出物理动作，也不需要 RobotAgent 和 Supervisor 共同写 ActionLedger。

保持现有四层所有权：

~~~text
Agent / Cognition
  -> Skill Gateway / Skill OS
  -> Foundation Model services
  -> Robot Runtime
~~~

不做以下替换：

- 不用 nanobot 进程内 MessageBus 替换 NATS；
- 不把 RobotAction 暴露给 Agent；
- 不把 SkillSpec、安全、资源调度放进 prompt；
- 不把物理技能变成与普通数字工具同级的通用 Tool；
- 不让 Supervisor 调用 provider；
- 不让 StrictAgentRunner 导入 Skill OS。

## 5. 状态机

### 5.1 Goal 状态

~~~text
PENDING
  -> ACTIVE
  -> WAITING
  -> ACTIVE
  -> COMPLETED

ACTIVE / WAITING
  -> BLOCKED
  -> FAILED
  -> CANCELLED

BLOCKED
  -> FAILED
  -> CANCELLED
~~~

| 状态 | 含义 |
|---|---|
| PENDING | Goal 和合同已持久化，尚未开始第一次 deliberation |
| ACTIVE | 可以执行一次新的 deliberation |
| WAITING | 已提交唯一外部 skill/control，正在等待 terminal result |
| BLOCKED | 物理状态未知或机器人离线；第一版不会自动恢复 |
| COMPLETED | immutable TaskContract 已由证据满足 |
| FAILED | 明确失败 |
| CANCELLED | 用户显式取消 |

第一版没有 PAUSED 和 resume。BLOCKED Goal 可以被取消，但如果阻塞原因是 Action UNKNOWN，机器人级 execution-uncertain lock 仍然保留；完成显式对账前不能创建会发出动作的新 Goal。

termination_reason 一旦写入就不可清除，并立即成为 dispatch barrier。唯一允许的变更是 cancel/budget -> emergency 的安全升级；cancel 与 budget 同优先级，按 Supervisor 串行事件顺序 first-write-wins。它只决定确认 idle 后的 Goal 终态，不允许重新进入 ACTIVE。

禁止：

- terminal Goal 转回 ACTIVE；
- 没有 TaskEvaluation(SATISFIED) 就进入 COMPLETED；
- WAITING 时调用 LLM；
- BLOCKED 时自动重新发送动作。

### 5.2 Deliberation 状态

~~~text
SCHEDULED
  -> BUILDING
  -> BEFORE_MODEL_REQUEST
  -> MODEL_RESPONSE_RECEIVED
  -> TERMINAL

BUILDING
  -> TERMINAL
~~~

这也是 DeliberationStore 唯一使用的持久化 phase 词汇。BUILDING 包含 context build 和一次 TaskEvaluation；SATISFIED 或任意失败可以直接进入 TERMINAL。只有写入 BEFORE_MODEL_REQUEST 后才允许调用 provider，最多一次。MODEL_RESPONSE_RECEIVED 后只做严格解析和 ActionProposal 构造，没有物理 IO。进程重启发现任何非 terminal phase 都生成 AGENT_PROCESS_INTERRUPTED，不恢复运行。

### 5.3 Action 状态

~~~text
PERSISTED
  -> PUBLISHING | FAILED | CANCELLED

PUBLISHING
  -> PUBLISHED | COMPLETED | FAILED | INTERRUPTED | UNKNOWN | RECONCILED_IDLE

PUBLISHED
  -> ACCEPTED | RUNNING | COMPLETED | FAILED | INTERRUPTED | UNKNOWN | RECONCILED_IDLE

ACCEPTED
  -> RUNNING | COMPLETED | FAILED | INTERRUPTED | UNKNOWN | RECONCILED_IDLE

RUNNING
  -> COMPLETED | FAILED | INTERRUPTED | UNKNOWN | RECONCILED_IDLE

UNKNOWN
  -> COMPLETED | FAILED | INTERRUPTED | RECONCILED_IDLE
~~~

模型决定保存在 DeliberationResult 中，不是 Action 状态。只有 Supervisor 接受提案并写入 ActionLedger 后，ActionRecord 才从 PERSISTED 开始。

PUBLISHING 必须在调用 bus.publish 前持久化。进程在 PUBLISHING 中断时，消息可能已经被 Skill OS 接收，因此进入 UNKNOWN，不能重发。SkillResult 可以让 PUBLISHING/PUBLISHED 直接跳到 terminal；状态更新必须使用 compare-and-set，terminal 状态绝不能被随后返回的 publish 调用覆盖成 PUBLISHED。

UNKNOWN 是机器人级执行锁，不只是当前 Goal 的状态。可信的后续 SkillResult 可以跳过丢失的中间状态，例如 PUBLISHED -> COMPLETED；terminal SkillControlResult + idle confirmation，或人工操作员的权威对账，可以把 UNKNOWN 变为 RECONCILED_IDLE。取消 Goal 不会清除这个锁。

RECONCILED_IDLE 只证明执行已经停止，是 execution-terminal，但不证明动作成功，永远不能作为 TaskContract 的成功证据。

## 6. 最小协议

在 src/hey_robot/protocol/messages.py 增加以下类型。跨服务控制字段必须有明确类型，不能隐藏在 metadata 中。

### 6.1 Goal、合同和证据

第一版不使用任意字符串 DSL。所有成功条件是 all-of，只有以下三种正向条件：

| criterion_type | 允许的 predicate | 示例 |
|---|---|---|
| robot_state | equals | robot.location equals living_room |
| object_relation | at / near / inside / held_by | wand at dock |
| evidence_present | observed | kitchen observed scene |

不支持 OR、NOT、自然语言 predicate 或模型自定义 predicate。不存在可信反证时只能得到 INCONCLUSIVE，不能把“没有看到”推断成“不存在”。

subject_id 和 object_id 不是任意自然语言。Goal 创建时必须通过 deployment 的 EntityCatalog / RobotStateSchema 校验，例如 robot:main.location、room:living_room、object:wand、fixture:dock。未注册 ID 直接 GOAL_CONTRACT_INVALID。不同空间层级的事实可以同时成立，Evaluator 不根据另一个 object_id 自动推断冲突。

~~~python
CriterionType = Literal[
    "robot_state",
    "object_relation",
    "evidence_present",
]

CriterionPredicate = Literal[
    "equals",
    "at",
    "near",
    "inside",
    "held_by",
    "observed",
]


@dataclass(frozen=True)
class GoalBudgets:
    max_wall_time_sec: float = 1800.0
    max_deliberations: int = 20
    max_skills: int = 12
    min_battery_percentage: float = 20.0


@dataclass(frozen=True)
class SuccessCriterion:
    criterion_id: str
    criterion_type: CriterionType
    subject_id: str
    predicate: CriterionPredicate
    object_id: str
    max_age_sec: float


@dataclass(frozen=True)
class EvidenceFact:
    evidence_id: str
    goal_id: str
    source_kind: Literal["robot_status", "skill_result"]
    source_id: str
    observed_at: float
    frame_id: int | None
    subject_id: str
    predicate: CriterionPredicate
    object_id: str
    artifacts: tuple[ArtifactRef | ImageRef, ...] = ()


@dataclass(frozen=True)
class GoalCommand:
    envelope: Envelope
    command_id: str
    action: Literal["create", "cancel"]
    goal_id: str | None = None
    objective: str = ""
    contract_template_id: str | None = None
    success_criteria: tuple[SuccessCriterion, ...] = ()
    budgets: GoalBudgets = field(default_factory=GoalBudgets)
~~~

创建 Goal 时必须提供 success_criteria，或显式选择已注册的 contract_template_id。系统不从 objective 文本猜测完成条件。新自主控制消息不提供 metadata 逃生口；控制字段必须进入明确 schema。

command_id 必填并在 GoalStore 中唯一，用于重复消息去重。action=create 时 Envelope.robot_id 必填，goal_id 必须为空并由 Supervisor 生成，objective 必填；action=cancel 时 goal_id 必填，criteria/template/budgets 不参与更新。取消从不修改原合同。

原始 RobotObservation 只是传感器快照，不能直接证明 object_relation。inspect_scene 等观察技能必须通过 SkillResult.evidence 输出 EvidenceFact；TaskEvaluator 不能解析 summary、metadata 或模型文本生成事实。

source_kind == skill_result 时 source_id 必须等于对应 skill_id；source_kind == robot_status 时 source_id 使用 status:<robot_id>:<frame_id>，且 frame_id 必填。没有稳定来源 ID 的数据只能进入 trace，不能进入 EvidenceLedger。

感知置信度阈值属于感知服务/SkillSpec：低于阈值时不要发出语义事实，只返回原始 artifact 和 INCONCLUSIVE 所需的缺证据状态。TaskEvaluator 不再实现第二套置信度启发式。

RobotStatus 增加 location_id、motion_state、battery_percentage 等显式字段。Supervisor 在接收 status 时调用 task/evidence.py 中的纯函数 project_robot_status()，只从这些类型化字段生成 robot_state EvidenceFact，并在更新 snapshot 的同一事务写入当前非 terminal Goal 的 EvidenceLedger；没有 Goal 时只更新 snapshot。projector 不读取 metrics、summary 或自然语言 task 字段。

### 6.2 Deliberation 和动作提案

~~~python
@dataclass(frozen=True)
class GoalSnapshot:
    goal_id: str
    version: int
    task_id: str
    contract_id: str
    contract_hash: str
    objective: str
    success_criteria: tuple[SuccessCriterion, ...]
    status: Literal[
        "pending",
        "active",
        "waiting",
        "blocked",
        "completed",
        "failed",
        "cancelled",
    ]
    termination_reason: Literal["cancel", "budget", "emergency"] | None = None


@dataclass(frozen=True)
class ActionSnapshot:
    skill_id: str
    deliberation_id: str
    intent_kind: Literal["skill", "observation"]
    name: str
    objective: str
    arguments: dict[str, Any]
    status: Literal[
        "persisted",
        "publishing",
        "published",
        "accepted",
        "running",
        "completed",
        "failed",
        "interrupted",
        "cancelled",
        "unknown",
        "reconciled_idle",
    ]


@dataclass(frozen=True)
class BudgetState:
    elapsed_wall_time_sec: float
    deliberations_used: int
    skills_used: int
    battery_percentage: float | None


@dataclass(frozen=True)
class DeliberationRequest:
    envelope: Envelope
    deliberation_id: str
    trigger_event_id: str
    goal: GoalSnapshot
    robot_status: RobotStatus | None
    robot_observation: RobotObservation | None
    actions: tuple[ActionSnapshot, ...]
    evidence: tuple[EvidenceFact, ...]
    latest_skill_result: SkillResult | None
    budget_state: BudgetState


@dataclass(frozen=True)
class ActionProposal:
    intent_kind: Literal["skill", "observation"]
    skill_name: str
    objective: str
    arguments: dict[str, Any]
~~~

DeliberationRequest 是完整、自包含的系统消息，不伪装成 UserTurn。RobotAgentService 只读取该请求，不直接读取 GoalStore、ActionLedger 或 Supervisor 内存。

ActionProposal 不是 SkillIntent，没有 skill_id，也没有发布权限。它只表达模型选择；Supervisor 是唯一可以把它变成真实 SkillIntent 的组件。

### 6.3 失败、评估和结果

~~~python
@dataclass(frozen=True)
class FailurePayload:
    stage: str
    code: str
    component: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskEvaluationPayload:
    outcome: Literal["satisfied", "inconclusive"]
    reason: str
    evidence_ids: tuple[str, ...] = ()
    missing_criteria_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeliberationResult:
    envelope: Envelope
    deliberation_id: str
    request_hash: str
    goal_id: str
    task_id: str
    status: Literal[
        "completed",
        "action_proposed",
        "failed",
    ]
    proposal: ActionProposal | None = None
    failure: FailurePayload | None = None
    evaluation: TaskEvaluationPayload | None = None


@dataclass(frozen=True)
class GoalEvent:
    envelope: Envelope
    event_id: str
    goal_id: str
    task_id: str
    status: Literal[
        "pending",
        "active",
        "waiting",
        "blocked",
        "completed",
        "failed",
        "cancelled",
    ]
    active_skill_id: str | None = None
    active_control_id: str | None = None
    termination_reason: Literal["cancel", "budget", "emergency"] | None = None
    evaluation: TaskEvaluationPayload | None = None
    failure: FailurePayload | None = None
~~~

约束：

- action_proposed 必须有且只能有一个 proposal；
- completed 必须有 evaluation.outcome == satisfied；
- failed 必须有 FailurePayload；
- Agent 不返回 blocked；物理未知和机器人离线由 Supervisor 判定；
- FailurePayload.details 只保存诊断数据，任何状态迁移只能读取 stage/code 和显式字段；
- 第一版不让模型生成 Goal 最终回复。UI 根据 GoalEvent、EvidenceFact 和最后一个 SkillResult 展示结果，展示文本不参与完成判定。

from_payload() 必须递归恢复嵌套 dataclass、tuple 和 optional union；未知字段、缺失必填字段、非法枚举和类型错误直接失败。所有新消息增加 encode/decode round-trip 和 negative tests。

### 6.4 Topics

新增：

~~~text
goal.command
goal.event
agent.deliberation
agent.deliberation.result
skill.control
skill.control.result
~~~

继续使用：

~~~text
skill.intent
skill.event
skill.result
robot.status
robot.observation
runtime.event
~~~

runtime.event 只用于 trace、指标和 UI progress，不参与物理状态判断。

skill.control 和 skill.control.result 是 cancel / emergency stop 使用的类型化控制通道，不是 LLM tool，也不能绕过持久化账本。

V1 的 SkillIntent 只有一类来源：AutonomySupervisor。现有 SkillIntent 增加以下必填字段：

~~~python
goal_id: str
task_id: str
deliberation_id: str
intent_kind: Literal["skill", "observation"]
~~~

切换时删除 SkillIntent.origin 和 metadata；技能执行输入只允许进入通过 SkillSpec 校验的 arguments，诊断信息进入 trace。NATS 权限只允许 AutonomySupervisor 发布 skill.intent，普通 UserTurn、RobotAgentService 和其他 system service 都没有发布权限。停止动作只能使用 skill.control。

SkillResult 增加 evidence: tuple[EvidenceFact, ...]；summary 和 metrics 只用于展示、诊断和实验指标，TaskEvaluator 永远不读取它们决定完成。

SkillResult.status 收敛为 completed / failed / interrupted / unknown：completed 要求 success is True；failed/interrupted 要求 success is False；unknown 要求 success is None。非法组合在协议边界失败。

completed / failed / interrupted 都是 terminal 承诺：Skill OS 已停止该 run、释放资源，并能证明机器人不再执行该 skill。做不到这一点只能发布 unknown，不能用 failed 掩盖仍可能运行的动作。

RobotStatus.state 至少收敛为 idle / executing / error / offline / unknown。只有 state == idle、skill_id is None，并且 SkillCommandStore 没有 active receipt 时，才可作为解除 execution lock 的候选证据；通用 metrics 字段不能证明 idle。

## 7. TaskContract 和完成判定

当前 TaskContract.required_skill 把“完成目标”等同于“执行一个技能”，无法表达长期家庭任务。

目标结构：

~~~python
@dataclass(frozen=True)
class TaskContract:
    contract_id: str
    task_id: str
    goal_id: str
    objective: str
    success_criteria: tuple[SuccessCriterion, ...]
    schema_version: int
    contract_hash: str
~~~

规则：

1. 新合同没有 required_skill；
2. 合同只定义完成条件，不定义执行步骤；
3. 第一版没有任务 DAG 和自动子目标树；
4. Goal 创建时生成稳定 task_id、contract_id 和 hash；
5. GoalStore 把 Goal 和合同写入同一个 snapshot；
6. 每个 deliberation 使用同一个 contract hash；
7. 第一版修改成功标准必须取消旧 Goal 并创建新 Goal；
8. 无 criteria/template 的 Goal 以 GOAL_CONTRACT_REQUIRED 拒绝；
9. 模型 final text 永远不是物理完成证据；
10. 所有 criteria 采用 all-of；第一版不支持 OR、NOT 和自然语言谓词。

例如：

~~~text
目标：检查 wand 是否在 dock，不在则放回
完成条件：object_relation(wand, at, dock)
证据：匹配对象和位置关系的最新视觉/状态证据
~~~

多房间巡视也不需要任务 DAG：

~~~text
objective：巡视客厅、餐厅和厨房
criteria（all-of）：
  evidence_present(living_room, observed, scene)
  evidence_present(dining_room, observed, scene)
  evidence_present(kitchen, observed, scene)
~~~

合同不包含固定的 inspect -> pick -> place 序列。如何达到目标仍由 Agent 决定。

TaskEvaluator 只读取 TaskContract 和 EvidenceFact，并输出：

~~~text
SATISFIED
INCONCLUSIVE
~~~

确定性求值规则：

1. 只接受 goal_id 匹配、source_id 能对应 ActionLedger/RobotStatus、消息来源服务身份受信任且未超过 max_age_sec 的 EvidenceFact；
2. 每个 criterion 都存在 (subject_id, predicate, object_id) 精确匹配事实时为 SATISFIED；
3. 其余情况全部为 INCONCLUSIVE，包括看到同一实体位于另一个空间层级或另一个位置；
4. 相同 evidence_id 对应不同 payload、来源非法或 evaluator 异常时结构化失败，不能选择一个看起来合理的结果；
5. 不根据缺失事实推断反面。

每个 deliberation 在调用模型前只评估一次：

- SATISFIED：直接返回 completed，不调用模型；
- INCONCLUSIVE：调用模型选择唯一的下一动作；
- 模型提交一个 tool：产生 ActionProposal；
- 模型只返回文本：FAILED(MODEL_STOPPED_BEFORE_GOAL)。

第一版不调用模型生成最终报告。UI 可以展示结构化 GoalEvent、证据引用和 SkillResult.summary，但这些展示内容不能反向修改 Goal 状态。

不自动 reobserve，不让 Supervisor 或 Runner 重新解释。

## 8. StrictAgentRunner：一次 slice 一次决策

### 8.1 目标接口

~~~python
@dataclass(frozen=True)
class AgentRunRequest:
    messages: tuple[ReasoningMessage, ...]
    allowed_tools: frozenset[str]
    deadline: float
    run_id: str
    deliberation_id: str


@dataclass(frozen=True)
class AgentRunResult:
    status: Literal["returned", "action_proposed", "failed"]
    final_text: str | None
    stop_reason: str
    tool_calls: tuple[ToolCallRecord, ...]
    proposal: ActionProposal | None
    failure: AgentFailure | None
    usage: dict[str, int]
~~~

returned 只表示模型正常返回文本，不表示任务完成。

### 8.2 Runner 只负责

- provider request；
- provider response 校验；
- tool name/arguments 校验；
- 执行最多一个无物理 IO 的 tool adapter；
- deadline；
- usage；
- checkpoint callback；
- trace event。

第一版没有内部模型工具循环。每次 Runner 只请求模型一次，结果只能是 final text 或一个 tool call。两个工具只产生 ActionProposal，不直接提交外部 intent，因此：

- 不实现 tool batch；
- 模型一次返回多个 tool calls 时直接 MULTIPLE_ACTION_PROPOSALS 失败；
- tool 参数转换为 proposal 后立即结束 Runner；
- 不把 tool result 再交给同一个 slice 的模型；
- Supervisor 接受 proposal 后才可能提交 SkillIntent；
- SkillResult 到达后由 Supervisor 创建新的 deliberation。

### 8.3 Runner 不负责

- TaskContract；
- EvidenceLedger；
- Goal 完成判定；
- observation freshness 策略；
- recovery；
- 用户回复 presentation；
- provider fallback；
- final answer 合成；
- tool result 文本解释；
- orphan/missing result 修复；
- SkillGateway 和 ActionLedger；
- SkillIntent 发布；
- 任何内部 continuation。

Provider API 和 tool-call 格式差异只能存在于 providers/。Provider 不支持声明能力时明确失败，不切换文本协议或另一个 provider。

### 8.4 严格失败

| 情况 | 结果 |
|---|---|
| Provider timeout | PROVIDER_TIMEOUT |
| Provider error | PROVIDER_ERROR |
| 空响应 | EMPTY_MODEL_RESPONSE |
| 非法响应 | INVALID_MODEL_RESPONSE |
| 未知工具 | UNKNOWN_TOOL |
| 参数错误 | INVALID_TOOL_ARGUMENTS |
| 多个工具调用 | MULTIPLE_ACTION_PROPOSALS |
| 合同未满足时只返回文本 | MODEL_STOPPED_BEFORE_GOAL |
| Tool 抛异常 | TOOL_FAILED |

所有情况都返回结构化失败，不生成伪正常回复。

## 9. Agent 工具：8 个缩减为 2 个

自主 RobotAgent 模型只看到：

~~~text
request_observation(question)
request_skill(skill, objective, slots)
~~~

### 9.1 request_observation

- 只表示主动获取新感知证据；
- 固定转换为 ActionProposal(intent_kind="observation", skill_name="inspect_scene")；
- 不访问 NATS、SkillGateway、ActionLedger 或相机 IO；
- proposal 构造成功后返回 STOP_SLICE；
- 不包含 wait policy；
- 不调用第二条直接 IO/caption fallback；
- 参数非法就是 INVALID_TOOL_ARGUMENTS。

第一版把 terminal SkillResult 作为主动观测的唯一完成事件。SkillResult 必须携带 frame_id 和 observation/image 引用。普通 robot.observation 只更新最新快照，不单独唤醒 Agent。

这避免 SkillResult 和 RobotObservation 两个事件分别唤醒一次。

### 9.2 request_skill

- 只允许非 observe/perception 类 SkillSpec；
- SkillGateway 再次校验 safety level/category；
- motion 所需观测不新鲜时由安全 gate 拒绝并明确失败，不自动改调 request_observation；
- inspect_scene 等感知技能从该入口直接拒绝；
- 安全拒绝产生结构化失败，不进入审批 fallback；
- 固定转换为 ActionProposal，不直接提交 SkillIntent；
- proposal 构造成功后返回 STOP_SLICE；
- 不等待 SkillResult；
- 不暴露 wait policy。

### 9.3 删除的工具

| 当前工具 | 第一版处理 |
|---|---|
| get_robot_status | 删除；RobotStatus 是 DeliberationRequest 输入 |
| get_task_context | 删除；Goal、Action、Failure、Budget 是显式输入 |
| propose_skill | 删除；第一版安全拒绝即失败 |
| wait | 删除；WAITING 是状态，不是工具 |
| search_memory | 删除；第一版不研究长期记忆 |
| write_memory | 删除；模型不能写长期事实 |
| request_perception | 重构并重命名为 request_observation |
| request_skill | 保留语义，删除 wait policy 和字符串结果 |

### 9.4 Tool 框架简化

生产工具显式注册：

~~~python
def build_agent_tools(deps: AgentToolDependencies) -> ToolRegistry:
    return ToolRegistry(
        [
            RequestObservationTool(deps.skill_catalog),
            RequestSkillTool(deps.skill_catalog),
        ]
    )
~~~

删除：

- ToolLoader 和 pkgutil 自动扫描；
- 生产路径的 adhoc/dynamic tool 注册；
- 大而全的 ToolContext；
- Registry 和 Executor 的重复 coercion/validation；
- tool 内的 fallback、presentation 和 recovery。

每个 tool 只注入只读 SkillCatalog。参数只在一个边界校验。内部返回 ActionProposal，只在 provider tool-result 边界序列化。物理安全和资源状态由 Supervisor 调用 SkillGateway 时再次基于最新快照校验。

STOP_SLICE 只是 Runner 内部控制值，含义是“已经得到本 slice 的唯一决策，立即结束本次模型运行”。它不是 wait policy，也不表示物理动作已经提交。

## 10. SkillGateway 和 ActionLedger

### 10.1 提交流程

~~~text
模型决定一个 tool call
  -> RobotAgent 校验并返回 ActionProposal
  -> Supervisor 校验 Goal 仍为 ACTIVE 且 termination_reason 为空
  -> 检查 hard budget 和 RobotExecutionGate == READY
  -> SkillGateway 校验 SkillSpec / safety / resources
  -> 由 deliberation_id 确定性生成 skill_id
  -> 同一事务写 ActionRecord(PERSISTED) + Goal(WAITING)
  -> 写 ActionRecord(PUBLISHING)
  -> 发布 skill.intent
  -> 写 ActionRecord(PUBLISHED)
~~~

一个 slice 最多产生一个 ActionProposal；一个被接受的 proposal 最多对应一个 SkillIntent，包括 observe SkillIntent。

Supervisor 是唯一 ActionLedger writer 和 SkillGateway caller。RobotAgentService 不拥有物理提交权限。这条边界使以下竞态变得简单：

- cancel 在模型运行期间到达：Goal 先变为 CANCELLED，迟到 proposal 被忽略，不会发动作；
- 重复 DeliberationResult：相同 deliberation_id 只能找到同一 ActionRecord；
- SkillResult 极快到达：Goal 和 ActionRecord 已在 publish 前原子持久化，不会出现“结果先于等待状态”的空窗。

proposal acceptance 使用 Goal.version 和 RobotExecutionGate.version 做同一 SQLite 事务的 CAS。验证完成后若 cancel/emergency/watchdog 已改变任一版本，proposal 直接失效，不允许基于旧检查结果继续发布。

### 10.2 ActionRecord

~~~python
@dataclass
class ActionRecord:
    version: int
    deliberation_id: str
    goal_id: str
    task_id: str
    robot_id: str
    skill_id: str
    intent_kind: Literal["skill", "observation"]
    skill_name: str
    objective: str
    arguments: dict[str, Any]
    status: ActionStatus
    intent_persisted_at: float | None = None
    publish_started_at: float | None = None
    intent_published_at: float | None = None
    terminal_at: float | None = None
    result: SkillResult | None = None
    terminal_result_hash: str | None = None


@dataclass
class RobotExecutionGate:
    robot_id: str
    version: int
    state: Literal["ready", "stop_pending", "uncertain"]
    control_id: str | None = None
    reason: str | None = None
    updated_at: float = 0.0
~~~

所有自主权威状态使用同一个 SQLite 数据库：

~~~text
runtime/<deployment>/autonomy.sqlite3

goals
goal_commands
actions
evidence
outgoing_deliberations
control_commands
robot_execution_gate
~~~

新内核使用新的 experiment/deployment runtime 目录。生产进程不读取旧 TaskRun、pending turn 或 autonomy JSON；需要保留历史时只做离线导出，不写运行时兼容读取器。SQLite schema_version 不匹配直接拒绝启动。

规则：

- skill_id 使用 deployment namespace + deliberation_id 的 UUIDv5 确定性生成，不能随机重建；
- goal_commands.command_id 有唯一约束，重复 create/cancel 返回原处理结果；
- actions.deliberation_id 和 actions.skill_id 都有唯一约束；
- goals 对 robot_id 建立“最多一个非 terminal Goal”的唯一约束；
- Goal WAITING 和 ActionRecord PERSISTED 在一个事务中提交；
- PUBLISHING 必须先于 bus.publish；
- bus.publish 返回后只允许 CAS(PUBLISHING -> PUBLISHED)；record 已被快速 SkillResult 置为 terminal 时不得回写；
- 重启发现 PERSISTED 且从未进入 PUBLISHING：FAILED(DISPATCH_INTERRUPTED)，不继续发布；
- 相同 skill_id + 相同 terminal_result_hash 视为 replay；
- 相同 skill_id + 不同 terminal_result_hash 是 IDEMPOTENCY_CONFLICT，Action UNKNOWN、gate UNCERTAIN；非 terminal Goal 再进入 BLOCKED；
- 在 PUBLISHING 前明确失败可标为 FAILED 或 CANCELLED；
- PUBLISHING 后无法证明投递结果时标为 UNKNOWN；
- UNKNOWN 使 Goal BLOCKED，并把 RobotExecutionGate 置为 UNCERTAIN；
- STOP_PENDING 或 UNCERTAIN 都阻止该机器人接受任何新 SkillIntent；
- cancel、budget stop 和 emergency stop 的 SkillControl 允许在 STOP_PENDING/UNCERTAIN 时发送，因为它们只减少执行风险；
- UNKNOWN 不自动重发，取消 Goal 也不解除执行锁；
- SkillControlResult completed + idle confirmed 把 STOP_PENDING 置回 READY；
- UNKNOWN 收到可信 terminal SkillResult 时可转为对应 terminal，并在没有 STOP_PENDING control 时把 gate 置回 READY；原 BLOCKED Goal 不自动恢复；
- 操作员只有在 Skill OS receipt、RobotStatus 和机器人 idle 状态形成权威证据后，才能把 UNCERTAIN 置回 READY，并记录 RECONCILED_IDLE。

### 10.3 Skill OS 持久幂等

当前 src/hey_robot/skill_os/scheduler.py 只拒绝仍在内存中的重复 skill_id。终态 run 被移除或 SkillController 重启后，同一 SkillIntent 仍可能再次执行，因此现状不能满足物理幂等。

第一版必须新增 SkillCommandStore：

~~~text
runtime/<deployment>/skill_receipts.sqlite3
~~~

Skill OS 在校验和执行前先持久化 command receipt：

- payload hash 使用与 DeliberationRequest 相同的规范化 JSON 规则，并排除 Envelope.timestamp 等传输字段；

- 相同 skill_id、相同 payload hash：不再次执行；active 时重发当前状态，terminal 时重发原 SkillResult；
- 相同 skill_id、不同 payload hash：发布 IDEMPOTENCY_CONFLICT，拒绝执行并触发 execution lock；
- terminal SkillResult 先持久化，再发布；
- Controller 重启发现非 terminal receipt 时不恢复动作；有权威运行证据则继续等待，否则发布 UNKNOWN；
- SkillControl 以 control_id 为幂等键，使用相同 payload-hash 和 terminal-result 规则。

这不是自动恢复。它只保证重复投递不会制造第二次物理动作。

## 11. DeliberationStore：幂等 receipt，不做隐藏恢复

Supervisor 在发布前：

~~~text
生成 deliberation_id
  -> GoalStore 保存 active_deliberation_id + trigger_event_id
  -> 保存完整 DeliberationRequest（媒体只保存引用）和 hash
  -> 发布 agent.deliberation
~~~

outgoing_deliberations 对 (goal_id, trigger_event_id) 建唯一约束。Goal create 使用 command_id 作为 trigger；terminal SkillResult 使用 skill_id + terminal status 形成稳定 trigger。重复事件不能生成第二个 deliberation_id。

request_hash 使用规范化 JSON（固定 schema version、sorted keys、稳定 tuple/list 编码）计算；媒体只包含不可变引用和 sha256，不把本地临时路径变化计入控制哈希。

terminal_result_hash 只覆盖控制语义：ID、status、success、proposal/failure code、evaluation、frame/evidence ID 和 artifact sha256；排除 envelope timestamp、summary、metrics、trace 文本和诊断 details。诊断文案变化不构成 split-brain，控制字段变化才构成冲突。

Supervisor 只通过一个 schedule_deliberation() 事务完成调度：

~~~text
校验 Goal 是 PENDING，或 WAITING 且匹配 Action 已 terminal
  -> 校验 termination_reason 为空
  -> 校验 hard budget 仍可用
  -> 生成 deliberation_id 和完整 request
  -> Goal PENDING/WAITING -> ACTIVE
  -> 写 active_deliberation_id
  -> 增加 deliberation budget counter
  -> 写 outgoing_deliberations(request + hash)
  -> COMMIT
  -> 发布 agent.deliberation
~~~

SkillResult success 的事件事务必须同时完成：校验 terminal hash、Action terminal、Evidence 写入、Goal WAITING -> ACTIVE，以及 outgoing_deliberations 插入。SkillResult failed 则在同一事务写 Action terminal + Goal FAILED。termination_reason 非空时只更新 Action/Evidence，不创建新 deliberation。

这样重复 SkillResult 只有两种情况：事务未发生时可以正常处理；事务已提交时由 terminal hash 和 (goal_id, trigger_event_id) 唯一约束识别 replay。不存在“Action 已写完但唤醒永久丢失”的中间状态。

outgoing request 提交后发布失败或进程在 publish 前中断，Goal 直接 FAILED(DELIBERATION_DISPATCH_INTERRUPTED)；不自动重发。该失败不涉及物理动作。

RobotAgent 只使用一个持久化 DeliberationStore，同时承担幂等 receipt 和模型阶段记录：

~~~python
@dataclass
class DeliberationRecord:
    deliberation_id: str
    request_hash: str
    run_id: str | None
    goal_id: str
    task_id: str
    phase: str
    model_response_ref: str | None
    terminal_result: DeliberationResult | None
    terminal_result_hash: str | None
    updated_at: float
~~~

持久化到 runtime/<deployment>/agent_deliberations.sqlite3。它由 RobotAgentService 单独拥有，不与 Supervisor 共享写权限。

phase：

~~~text
SCHEDULED
BUILDING
BEFORE_MODEL_REQUEST
MODEL_RESPONSE_RECEIVED
TERMINAL
~~~

规则：

- 首次 SCHEDULED 请求启动 Runner；
- 重复的非 terminal 请求不启动第二个 Runner；
- 已有 terminal result 时只重发同一规范化 hash 的结果；
- ActionProposal 只存在于 terminal DeliberationResult；
- Agent 启动时发现任何非 TERMINAL record，都持久化并发布 AGENT_PROCESS_INTERRUPTED；
- 重启后不重新调用模型，也不从半截 checkpoint 猜测 proposal；
- Agent 先持久化 terminal_result，再发布；
- 相同 deliberation_id + 相同 terminal_result_hash 视为 replay；
- 相同 deliberation_id + 不同 terminal_result_hash 是 DELIBERATION_PROTOCOL_ERROR，不能 IGNORE。

Supervisor 等待 DeliberationResult 超时后直接 FAILED(DELIBERATION_TIMEOUT)，不自动重发请求。重复消息只可能来自传输层或人工故障注入，DeliberationStore 仍必须幂等处理。第一版宁可暴露一次投递失败，也不增加自动 continuation。

## 12. Context validation 和 trace

不新增独立 ContextInspector 层。ContextBuilder 只从不可变 DeliberationRequest 构造上下文，在返回模型请求前完成测量和严格校验，但不修复：

自主 deliberation 的上下文固定由以下结构组成：

~~~text
system policy
GoalSnapshot + immutable TaskContract
BudgetState
latest RobotStatus
latest relevant RobotObservation / observation SkillResult
DeliberationRequest.actions
DeliberationRequest.evidence
latest SkillResult
pre-run TaskEvaluation
~~~

不注入完整聊天历史、长期 memory、旧 Goal、continuation guidance 或恢复文本。历史 evidence 只放 ID、结构化字段和 artifact 引用，不重复内嵌旧图像。Goal budgets 已限制 action/evidence 数量，因此第一版直接保留当前 Goal 的结构化记录；如果仍超出 context budget，就明确失败，不增加总结或裁剪补丁。

- 消息角色；
- tool-call/result 配对；
- orphan/missing result；
- context/token 预算；
- TaskContract；
- GoalSnapshot；
- RobotStatus/Observation；
- tool schema 数量。

禁止：

- 自动裁剪当前 turn；
- backfill tool result；
- 自动总结；
- 自动 offload；
- 自动改变消息顺序。

超限或协议污染直接失败并保存原始证据。

统一 trace 至少记录：

~~~text
goal created/cancelled
deliberation scheduled/started/terminal
model request/response
tool validation
intent persisted/published
skill event/result
task evaluation
budget decision
progress fingerprint
failure
~~~

第一版只实现一个具体的 RunTraceWriter，不建立 hook/plugin/SDK 抽象。GoalStore、ActionLedger、DeliberationStore 和 SkillCommandStore 是权威存储，写入失败必须停止对应控制流；RunTraceWriter 是派生观测，写失败只记录本地错误和指标，不能修改 Goal 状态。这样不会同时宣称 trace 既是 best-effort 又是控制依据。

## 13. AutonomySupervisor

Wake 和 budget 实现为 autonomous/policy.py 中的纯函数，不创建多个 manager/service。Supervisor 提供当前 Goal、事件、快照和计数，policy 返回结构化决定。

Supervisor 对每个 robot_id 串行处理控制事件；所有 Goal/Action/Gate 更新仍带 version 字段并在 SQLite 事务中 CAS，不能只依赖进程内锁。这样 emergency/cancel、迟到 proposal、fast SkillResult 和 watchdog 只有一个状态迁移能成功。

### 13.1 订阅

~~~text
goal.command
agent.deliberation.result
skill.event
skill.result
skill.control.result
robot.status
robot.observation
~~~

### 13.2 职责

- create/cancel Goal；
- 保证每个机器人最多一个非 terminal Goal；
- 维护最新 RobotStatus/Observation 索引；
- 更新 ActionLedger；
- 校验并接受 ActionProposal；
- 通过 SkillGateway 提交唯一 SkillIntent；
- 执行最小 WakePolicy；
- 检查预算；
- 发布 DeliberationRequest；
- 重启 reconciliation；
- watchdog；
- BLOCKED/FAILED 通知。

watchdog 只是检查已存在的 deadline、robot heartbeat 和 hard budget。它不创建 Goal、不周期性唤醒模型、不重新发布消息，也不选择恢复动作，因此不等同于 timer/periodic goal 功能。

不负责：

- 构造 prompt；
- 调用 provider；
- 选择 skill；
- 执行 RobotAction；
- 修改安全 gate；
- 自动恢复；
- 自动创建新 Goal。

### 13.3 最小 WakePolicy

| 事件 | 当前状态 | 决定 |
|---|---|---|
| Goal create，gate READY | PENDING | schedule_deliberation(command_id)；原子转 ACTIVE |
| Goal create，gate STOP_PENDING/UNCERTAIN | 无 | REJECT(ROBOT_EXECUTION_UNCERTAIN) |
| DeliberationResult completed，termination_reason 为空 | ACTIVE | COMPLETED |
| DeliberationResult action_proposed，termination_reason 为空且 gate READY | ACTIVE | 校验 proposal；原子写 Goal WAITING + ActionRecord；发布 SkillIntent |
| DeliberationResult failed，termination_reason 为空 | ACTIVE | FAILED |
| 非 active deliberation_id 的 result，且 hash 与已存结果一致 | 任意 | IGNORE_REPLAY |
| 相同 deliberation_id 但 result hash 冲突 | 任意 | DELIBERATION_PROTOCOL_ERROR；已有动作时 gate UNCERTAIN，非 terminal Goal 再 BLOCKED |
| 匹配的 SkillResult success，termination_reason 为空且 budget 可用 | WAITING | 原子写 result/evidence 并 schedule_deliberation(trigger) |
| 匹配的 SkillResult success 后 budget 已耗尽 | WAITING | 原子写 result/evidence + Goal FAILED(BUDGET_EXHAUSTED)，不再 schedule |
| 匹配的 SkillResult failed/interrupted，termination_reason 为空 | WAITING | FAILED |
| 匹配的 SkillResult unknown，termination_reason 为空 | WAITING | Action UNKNOWN + gate UNCERTAIN + BLOCKED |
| UNKNOWN Action 后到达可信 terminal SkillResult，termination_reason 为空 | BLOCKED | Action terminal + gate READY；Goal 保持 BLOCKED，不自动恢复 |
| 任意匹配 SkillResult，termination_reason 非空 | WAITING/BLOCKED | 只更新 Action/Evidence，不唤醒、不覆盖 termination |
| SkillControlResult completed + idle confirmed | WAITING/BLOCKED | gate READY；cancel -> CANCELLED，budget -> FAILED(BUDGET_EXHAUSTED)，emergency -> FAILED(EMERGENCY_STOPPED) |
| SkillControlResult failed/unknown 或未确认 idle | WAITING/BLOCKED | Action UNKNOWN + gate UNCERTAIN + BLOCKED |
| robot.observation | 任意 | UPDATE_ONLY |
| robot.status heartbeat | 任意 | 更新 snapshot，并通过 projector 原子写 status evidence |
| Robot offline，无 committed action | ACTIVE/WAITING | BLOCKED |
| Robot offline，存在 PUBLISHING/PUBLISHED/ACCEPTED/RUNNING action | ACTIVE/WAITING | Action UNKNOWN + gate UNCERTAIN + BLOCKED |
| Emergency stop | 任意 | 先持久化 gate STOP_PENDING + control；若 Goal 非 terminal 再写 termination/emergency；terminal Goal 不变；禁止新 dispatch |
| Goal cancel，proposal 尚未提交 | PENDING/ACTIVE | CANCELLED；迟到 result 不再 dispatch |
| Goal cancel，Action 仍为 PERSISTED | WAITING | Action CANCELLED，Goal CANCELLED |
| Goal cancel，Action 已开始 publish | WAITING | 原子写 termination/cancel + gate STOP_PENDING + control，再发送 interrupt |
| hard budget 耗尽，无 nonterminal Action | PENDING/ACTIVE/WAITING | FAILED(BUDGET_EXHAUSTED)；迟到 result 不再 dispatch |
| hard budget 耗尽，Action 可能运行 | WAITING | 原子写 termination/budget + gate STOP_PENDING + control；确认 idle 后 FAILED |
| gate STOP_PENDING/UNCERTAIN | 任意 ActionProposal | REJECT(ROBOT_EXECUTION_UNCERTAIN) |
| 重复 SkillResult，且 terminal hash 一致 | 任意 | IGNORE_REPLAY |
| terminal Goal 的首个迟到 SkillResult | terminal | 更新 Action late audit；不唤醒、不改变 Goal |
| 相同 skill_id 但 terminal hash 冲突 | 任意 | IDEMPOTENCY_CONFLICT + gate UNCERTAIN；非 terminal Goal 再 BLOCKED |
| 无关 goal/skill event | 任意 | IGNORE |

Observation 不直接唤醒。模型只有在上一 SkillResult 后的新 deliberation 中才能决定是否调用 request_observation。

termination_reason 和 RobotExecutionGate 规则优先于普通 DeliberationResult/SkillResult。stop control 一旦持久化，即使目标 SkillResult 先到，也必须等待 SkillControlResult 或权威 idle；反过来 control result 先到时可以终结 Goal，随后到达的 SkillResult 只作审计记录，不再改变 Goal。

Supervisor 必须验证 DeliberationResult 的 deliberation_id、Goal/task ID、request hash 和状态约束。只有 active Goal 的 action_proposed result 能进入 dispatch。ActionRecord 与 Goal WAITING 在 publish 前已经原子写入，因此 SkillResult 即使立即返回，也只能更新一个已存在的 record。缺失或不一致时进入 BLOCKED/STATE_UNKNOWN，不能补写一条“看起来应该存在”的记录。

### 13.4 Budget

第一版只保留：

- 总墙钟时间；
- deliberation 次数；
- skill 次数；
- 最低电量。

DeploymentConfig 只增加一个显式 AutonomySpec，不再把这些字段塞入 agents.main.settings：

~~~python
@dataclass(frozen=True)
class AutonomySpec:
    enabled: bool = False
    robot_id: str | None = None
    hard_max_wall_time_sec: float = 3600.0
    hard_max_deliberations: int = 40
    hard_max_skills: int = 24
    min_battery_percentage: float = 20.0
~~~

GoalCommand budgets 只能收紧 deployment hard caps，不能放宽。旧 autonomy 配置字段直接报错。

提交物理 skill 前 battery_percentage 为 None 或 RobotStatus 过期时，Supervisor 不假定安全，直接 BLOCKED(ROBOT_OFFLINE/STATE_UNKNOWN)。request_observation 本身仍需满足 SkillSpec 的最低安全前提。

预算在以下位置检查：

~~~text
Goal 开始前
每次 DeliberationRequest 前
每次 SkillIntent 前
每次 SkillResult 后
watchdog tick
~~~

预算耗尽：

~~~text
没有 nonterminal Action
  -> Goal FAILED(BUDGET_EXHAUSTED)
  -> 忽略迟到 proposal

Action 可能仍在运行
  -> termination_reason = budget
  -> RobotExecutionGate STOP_PENDING
  -> 持久化并发布 interrupt SkillControl
  -> 保持 WAITING
  -> idle confirmed: Goal FAILED(BUDGET_EXHAUSTED)
  -> stop unknown: Goal BLOCKED + gate UNCERTAIN
~~~

预算绝不能让 Goal 在机器人仍可能运动时直接进入 terminal。

### 13.5 重复行为只记录，不自动解释

第一版删除 ProgressDetector。墙钟、deliberation 和 skill 三个硬上限已经保证系统最终停止；额外的 fingerprint 启发式容易把持续变化的 frame_id 误判为进展，或把真实但缓慢的动作误判为无进展。

trace 仍记录重复 skill、相同 arguments、相同 evidence 和合同状态未变化，供仿真实验分析。系统不会据此切换策略、自动重新观察或生成新的恢复分支；循环最终以 BUDGET_EXHAUSTED 明确失败。

## 14. 失败模型

### 14.1 FailureStage

~~~text
GOAL_CONTRACT
CONTEXT_BUILD
MODEL_REQUEST
MODEL_PROTOCOL
TOOL_VALIDATION
TOOL_EXECUTION
ACTION_DISPATCH
SKILL_EXECUTION
SKILL_CONTROL
TASK_EVALUATION
PERSISTENCE
SUPERVISION
~~~

### 14.2 第一版 FailureCode

~~~text
GOAL_BUSY
GOAL_REQUIRED
GOAL_CONTRACT_REQUIRED
GOAL_CONTRACT_INVALID
ROBOT_EXECUTION_UNCERTAIN
ROBOT_OFFLINE
CONTEXT_BUDGET_EXCEEDED
PROVIDER_TIMEOUT
PROVIDER_ERROR
EMPTY_MODEL_RESPONSE
INVALID_MODEL_RESPONSE
UNKNOWN_TOOL
INVALID_TOOL_ARGUMENTS
MULTIPLE_ACTION_PROPOSALS
DELIBERATION_PROTOCOL_ERROR
TOOL_FAILED
SAFETY_REJECTED
SKILL_REJECTED
SKILL_FAILED
SKILL_TIMEOUT
INVALID_EVIDENCE
IDEMPOTENCY_CONFLICT
MODEL_STOPPED_BEFORE_GOAL
DELIBERATION_TIMEOUT
DELIBERATION_DISPATCH_INTERRUPTED
AGENT_PROCESS_INTERRUPTED
DISPATCH_INTERRUPTED
STATE_UNKNOWN
BUDGET_EXHAUSTED
INTERRUPTED
EMERGENCY_STOPPED
PERSISTENCE_FAILED
~~~

### 14.3 失败映射

| 失败 | Goal 结果 |
|---|---|
| Goal contract 缺失/非法 | 不创建 Goal，明确拒绝 |
| Provider/协议/tool 错误 | FAILED |
| Deliberation publish/timeout / Agent process interrupted | FAILED |
| Deliberation result/hash/state 不一致 | FAILED(DELIBERATION_PROTOCOL_ERROR)；若已存在动作记录则 gate UNCERTAIN，非 terminal Goal 再 BLOCKED |
| Evidence 非法、互相矛盾或 evaluator 异常 | FAILED(INVALID_EVIDENCE) |
| Safety/Skill 拒绝 | FAILED |
| Skill 明确失败/超时 | FAILED |
| Budget exhausted，无 active action | FAILED(BUDGET_EXHAUSTED) |
| Budget exhausted，action 可能运行 | STOP_PENDING；idle 后 FAILED，未知则 BLOCKED |
| Observation skill 失败 | FAILED |
| 模型在 INCONCLUSIVE 时停止行动 | FAILED |
| publish/动作状态未知 | BLOCKED |
| 相同 skill_id 对应不同 payload | BLOCKED(IDEMPOTENCY_CONFLICT) + execution lock |
| Robot offline 且可能有 active action | BLOCKED + execution lock |
| Robot offline 且没有已提交 action | BLOCKED |
| Emergency stop 已确认 | FAILED(EMERGENCY_STOPPED) |
| Emergency stop 结果未知 | BLOCKED + execution lock |
| Persistence failure | FAILED；若动作可能已发出则 BLOCKED |

SKILL_TIMEOUT 只有在 Skill OS 已经完成取消/停止并发布 terminal SkillResult 时才是明确 FAILED。Supervisor 自己等待超时不能证明机器人停止，必须转成 STATE_UNKNOWN、BLOCKED 和 execution lock。

第一版没有 RecoveryPolicy 和 retryable 自动行为。

## 15. 用户输入

第一版自主 Goal 入口只有：

~~~text
/goal create ...
/goal cancel <goal_id>
REST Goal API create/cancel
~~~

UNKNOWN 对账使用独立的 operator-only 管理入口，例如 /execution reconcile <skill_id> 或对应 REST Admin API；它不是 GoalCommand，也不是 Agent tool。

规则：

- 普通 UserTurn 不会隐式创建 Goal，也看不到任何机器人执行 tool；
- 普通 UserTurn 中的物理动作请求返回 GOAL_REQUIRED，引导用户显式创建 Goal；
- 已有非 terminal Goal 时，新 create 返回 GOAL_BUSY；
- 新的普通动作指令不修改 active Goal；
- 用户必须 cancel 后再创建新目标；
- emergency stop 使用独立高优先级通道，不经过 LLM；
- 状态查询可以由 UI/API 读取 GoalStore、ActionLedger 和 RobotStatus，不需要 Agent tool。

这是有意的破坏性变更：旧的“普通聊天消息直接驱动机器人动作”路径删除，不提供自动转 Goal 的 adapter。这样所有物理动作都能经过同一个合同、账本和 execution gate。

cancel 不是 LLM tool，但也不能成为账本之外的特殊旁路。使用显式协议：

~~~python
@dataclass(frozen=True)
class SkillControl:
    envelope: Envelope
    control_id: str
    action: Literal["interrupt", "emergency_stop"]
    target_skill_id: str | None
    goal_id: str | None
    reason: str


@dataclass(frozen=True)
class SkillControlResult:
    envelope: Envelope
    control_id: str
    action: Literal["interrupt", "emergency_stop"]
    target_skill_id: str | None
    status: Literal["completed", "failed", "unknown"]
    robot_idle_confirmed: bool
    error: str | None = None
~~~

Supervisor 在一个事务中写 Goal.termination_reason（若有非 terminal Goal）、RobotExecutionGate=STOP_PENDING 和 control_commands(PERSISTED)，再写 control PUBLISHING，最后发布 skill.control。Skill OS 通过 SkillCommandStore 幂等消费，先停止 active run，再执行必要的 stop_motion，并持久化 SkillControlResult 后发布 skill.control.result。

cancel 产生的 control_id 由 GoalCommand.command_id + target_skill_id 确定性生成；emergency stop 的 control_id 由其独立系统命令 ID 生成。

control_commands 使用 PERSISTED -> PUBLISHING -> PUBLISHED -> COMPLETED / FAILED / UNKNOWN，同样禁止 UNKNOWN 重发。control_id 有唯一约束，重复结果只更新同一记录。

处理 SkillControlResult 时，control terminal、target Action、RobotExecutionGate 和 Goal terminal/BLOCKED 必须在一个事务中提交，不能先把 Goal 结束再单独清 gate。

action == interrupt 时 target_skill_id 必填；action == emergency_stop 时允许为空并表示停止该机器人全部 active execution。SkillControlResult.status == completed 时 robot_idle_confirmed 必须为 True；failed/unknown 时必须为 False。非法组合按 UNKNOWN 处理。

control completed 时，已有 terminal SkillResult 的 Action 保持原 terminal 状态；没有 terminal SkillResult 的 target Action 记为 RECONCILED_IDLE，表示“确认已停止，但原动作结果未知”。之后到达的 SkillResult 只保存为 late audit result，不改变 Goal 或 gate；若同一 skill_id 的 terminal payload 自身冲突，仍按 IDEMPOTENCY_CONFLICT 处理。

取消规则：

1. 模型仍在运行且尚无 ActionRecord：直接 CANCELLED；迟到 proposal 被忽略；
2. Action 只有 PERSISTED 且尚未 PUBLISHING：Action CANCELLED，Goal CANCELLED；
3. Action 已到 PUBLISHING/PUBLISHED/ACCEPTED/RUNNING：同一事务写 termination_reason=cancel、RobotExecutionGate=STOP_PENDING 和 control record，然后发送 interrupt；
4. target SkillResult 先到时只更新 Action/Evidence，仍等待 control terminal；
5. SkillControlResult completed + idle confirmed 后才进入 CANCELLED；
6. interrupt 投递或停止结果无法确定：Action UNKNOWN、gate UNCERTAIN、Goal BLOCKED；
7. BLOCKED Goal 之后即使被取消，UNKNOWN record 和 gate UNCERTAIN 仍然存在，并继续阻止新 Goal 发动作。

emergency stop 使用同一个 skill.control 协议和 receipt store，只是优先级最高并允许在 gate UNCERTAIN 时发送。Supervisor 必须先写 gate STOP_PENDING 和 control record，之后才能 publish；这一持久 gate 会拒绝较晚返回的 ActionProposal。确认停止后，非 terminal Goal FAILED(EMERGENCY_STOPPED)；停止结果不确定时保持 BLOCKED。不存在含义模糊的 SAFE_ABORT 状态。

若 gate 已是 STOP_PENDING，emergency 只把 termination_reason 安全升级为 emergency，并复用正在进行的 stop control，不发布第二条并行停止命令；若 gate 是 UNCERTAIN，才创建新的 emergency_stop control。

解除 UNKNOWN 只允许操作员执行显式 reconcile 操作。系统必须同时记录目标 Action、权威 RobotStatus/Skill receipt、机器人 idle 证据和操作者身份，然后把 Action 标为 RECONCILED_IDLE、gate 置回 READY。它不把原动作伪装成成功，也不恢复旧 Goal；新的 Goal 使用自己的 goal_id，因此旧证据不会复用。

这样明确区分：

~~~text
interactive query/presentation（无物理工具）
autonomous goal
internal deliberation
~~~

不使用 metadata 偷偷区分模式。

## 16. 文件级改造

### 16.1 新增

~~~text
src/hey_robot/cognition/runtime/result.py
src/hey_robot/cognition/runtime/deliberation_store.py
src/hey_robot/cognition/runtime/trace.py

src/hey_robot/cognition/task/__init__.py
src/hey_robot/cognition/task/contract.py
src/hey_robot/cognition/task/evidence.py
src/hey_robot/cognition/task/state.py

src/hey_robot/cognition/policy/__init__.py
src/hey_robot/cognition/policy/task_evaluator.py

src/hey_robot/cognition/autonomous/__init__.py
src/hey_robot/cognition/autonomous/goal.py
src/hey_robot/cognition/autonomous/store.py
src/hey_robot/cognition/autonomous/supervisor.py
src/hey_robot/cognition/autonomous/policy.py
src/hey_robot/cognition/autonomous/action_ledger.py

src/hey_robot/skill_os/command_store.py
~~~

### 16.2 重点修改

| 文件 | 修改 |
|---|---|
| protocol/messages.py | Goal/Deliberation/Evidence/SkillControl、严格 RobotStatus/SkillResult 和嵌套反序列化 |
| protocol/topics.py | 6 个新 topic |
| config/model.py | 显式 AutonomySpec / EntityCatalog，旧字段报错 |
| app/runner.py | 只启动一个 AutonomySupervisorService |
| cognition/runtime/runner.py | 收敛为 StrictAgentRunner |
| cognition/runtime/tool_executor.py | tool 只构造 ActionProposal 和 STOP_SLICE |
| cognition/core.py | 删除 fallback、文本解析和完成猜测 |
| cognition/robot_agent.py | 消费 DeliberationRequest，发布 DeliberationResult |
| cognition/skill_gateway.py | 删除 wait policy；只允许 Supervisor dispatch |
| cognition/task_runtime.py | 拆分后删除 TaskRunManager 聚合入口 |
| cognition/task_supervisor.py | 提取必要 watchdog 后删除旧 service |
| cognition/container.py | 注入新 Runner/Evaluator/DeliberationStore |
| skill_os/controller.py | skill/control receipt、重复命令回放、UNKNOWN |
| skill_os/scheduler.py | 不再把内存 active run 当作唯一幂等依据 |
| skill_os/builtins/perception.py | inspect_scene 输出类型化 EvidenceFact，不靠 summary |
| robot_runtime/simulation/skill_adapter.py | 把仿真真值映射为受信 EvidenceFact |
| skill_os/event_sink.py 和 foundation adapter | 把感知/技能输出写入 SkillResult.evidence |
| Web/CLI | Goal create/cancel、UNKNOWN reconcile 和状态展示 |

### 16.3 最终删除

- cognition/autonomy.py；
- TaskSupervisorService；
- TaskRunManager 聚合入口；
- 旧单 required_skill TaskContract；
- ToolLoader 和生产自动发现；
- get_robot_status.py；
- get_task_context.py；
- propose_skill.py；
- pending confirmation store 和 confirmed 重放路径；
- wait.py；
- search_memory.py；
- write_memory.py；
- 旧 request_perception.py；
- SkillGateway wait policy；
- SkillIntent.interrupt 控制复用和 _handle_interrupting_intent 旁路；
- SkillIntent.metadata 控制字段旁路；
- SkillIntent.origin 的 interactive/system 物理执行分支；
- _successful_tool_fallback_result；
- _synthesize_final_answer_after_tool；
- _fallback_reply_from_failed_tool；
- _internal_protocol_retry_guidance；
- _final_answer_from_tool_result；
- _safe_tool_result_reply；
- _fallback_reply_for_unfinished_turn；
- _looks_like_internal_final_response；
- execution feedback 文本字段解析；
- 字符串 failure marker recovery；
- autonomous 路径的历史自动裁剪、backfill 和 context repair；
- 被忽略的旧配置参数；
- 旧 tool 名称 alias；
- legacy protocol prompt 文本。

删除前先有结构化替代，但不保留 deprecated alias。

cognition/memory、memory_context 和旧 episode/recovery 模块不得被新的 autonomous 路径导入。切换后若 rg 证明它们已无其他有效所有者，就直接删除，不保留“以后可能用到”的休眠兼容代码。

## 17. 分阶段实施

建议总工作量约 24～38 个工程日，不含 VLA/VLN 模型本身的调试。真正高风险的工作不是写 Supervisor，而是合同证据、物理幂等和一次性切换。

开发期间允许新内核先存在于隔离测试中，但生产运行时始终只有一条路径：

~~~text
切换前：只有旧运行时
切换提交：接入新运行时，同时删除旧入口
切换后：只有新运行时
~~~

禁止 legacy_mode、双写、运行时 fallback 和新旧配置开关。

### Phase 0：实验基线

预计：2～3 天。

工作：

1. 固定 Mock/MuJoCo 场景、随机种子、模型和配置；
2. 保存 5～10 条 golden trace；
3. 记录当前 stop_reason/failure 分布；
4. 列出所有 fallback、legacy symbol 和依赖测试；
5. 增加架构守卫测试。

验收：

- 全量测试通过；
- trace 可定位 model/tool/skill/status；
- 明确哪些测试依赖伪正常 fallback；
- 不修改生产行为。

### Phase 1：严格单次 Agent 内核

预计：5～8 天。

工作：

1. AgentFailure/AgentRunResult；
2. Provider、ToolExecutor、Core 统一失败；
3. StrictAgentRunner，一次最多一个 provider request；
4. ActionProposal 和两个无 IO tool adapter；
5. 新 TaskContract、EvidenceFact 和 TaskEvaluator；
6. presentation 只翻译，不修改状态；
7. 在新内核内删除 grounding 自动插入、final answer 合成和文本状态猜测。

验收：

- Provider timeout 不再是普通完成；
- 空响应不再生成 fallback；
- tool 参数错误不再被文案覆盖；
- 每个失败只有一个 stage/code；
- Runner 不导入 TaskContract、RobotStatus、SkillSpec；
- Runner 测试只使用 fake provider/tools；
- 单次 run 不会发出第二个 provider request；
- 模型 final text 不能完成物理目标；
- SATISFIED 时零次模型调用；
- tool 只产生 ActionProposal，不访问 NATS。

说明：本阶段不把新合同适配回旧 TaskRunManager。它只通过独立单元测试和 scripted kernel driver 验证，避免为了过渡引入兼容层。

### Phase 2：协议、权威存储和物理幂等

预计：7～11 天。

工作：

1. Goal/Deliberation 协议和 round-trip；
2. deliberation_id/DeliberationStore；
3. SuccessCriterion/EvidenceFact 严格反序列化；
4. RobotStatus projector、inspect_scene、仿真 adapter 和 SkillResult 输出类型化 evidence；
5. autonomy.sqlite3、GoalStore 和 ActionLedger；
6. ActionProposal 到 SkillIntent 的 Supervisor dispatch；
7. PERSISTED/PUBLISHING/PUBLISHED、CAS 状态更新和 terminal hash；
8. SkillCommandStore 和持久 skill_id 去重；
9. skill.control、ControlRecord 和 cancel/emergency 流程；
10. execution-uncertain lock 和人工 reconcile；
11. ContextBuilder 严格校验和 RunTraceWriter；
12. scripted provider + mock Skill OS 集成测试。

验收：

- 一个 slice 最多一个 ActionProposal；
- Supervisor 接受后最多一个 SkillIntent；
- 重复 DeliberationRequest 不重复调用模型；
- SkillIntent 发布前 Goal WAITING 和 ActionRecord 已原子持久化；
- Controller 重启后重复 skill_id 不会再次执行；
- PUBLISHING 中断进入 UNKNOWN，不自动重发；
- cancel during deliberation 不会发布迟到 proposal；
- UNKNOWN 即使取消 Goal 也继续阻止新动作。

说明：本阶段仍只运行 scripted integration harness，不给生产入口增加临时开关。

### Phase 3：一次性生产切换

预计：7～11 天。

工作：

1. 接入 AutonomySupervisor、Snapshot index 和 watchdog；
2. app/runner.py 只启动新 Supervisor；
3. robot_agent.py 消费 DeliberationRequest 并返回 DeliberationResult；
4. Web/CLI 接入 Goal create/cancel/reconcile；
5. Skill OS 接入 skill.control 和 SkillCommandStore；
6. 同一切换删除旧 autonomy.py、TaskSupervisorService、TaskRunManager；
7. 同一切换删除旧 TaskContract、wait policy 和八工具旧注册；
8. 同一切换删除 fallback、legacy prompt/protocol、旧配置字段和 alias；
9. 删除普通 UserTurn 直接发布物理动作和 SkillIntent.origin 分支；
10. 更新所有生产配置和 NATS ACL，不提供兼容转换。

验收：

- Goal 创建后启动第一次 deliberation；
- WAITING 时不调用 LLM；
- 匹配 SkillResult 只唤醒一次；
- Action terminal、Evidence、Goal ACTIVE 和 outgoing deliberation 在一个事务中提交；
- Observation 只更新 snapshot；
- terminal Goal 不再唤醒；
- 重启后 Goal 仍存在；
- UNKNOWN 不重发；
- 硬预算能停止循环；
- 每个机器人最多一个非 terminal Goal；
- 没有双 autonomy；
- 没有 wait policy；
- 没有 provider/text fallback；
- 没有旧工具名 alias；
- 旧配置直接失败；
- 模型只看到两个工具；
- 普通 UserTurn 不能绕过 Goal/Supervisor 发物理动作；
- rg 不再发现目标 legacy symbol。

### Phase 4：家庭仿真验证和静态审计

预计：5～8 天完成首轮，之后持续实验。该阶段不再承担行为删除，只做验证和缺陷修正。

顺序：

~~~text
确定性 Supervisor
  -> Scripted Provider + Mock Robot
  -> 真实 LLM + Mock Robot
  -> 真实 LLM + MuJoCo
  -> 最后单独测试 VLA/VLN
~~~

同时运行 no-legacy surface、依赖边界、重复消息、进程崩溃、UNKNOWN lock 和长时间资源测试。只有实验数据反复证明某个失败具有唯一且可验证的恢复语义时，才在新的研究分支讨论恢复；主分支第一版仍然失败即停止。

## 18. 测试

建议目录：

~~~text
tests/protocol/
  test_goal_deliberation_roundtrip.py
  test_skill_control_roundtrip.py

tests/cognition/runtime/
  test_failures.py
  test_strict_runner.py
  test_context_validation.py
  test_deliberation_store.py
  test_single_proposal_slice.py

tests/cognition/task/
  test_contract.py
  test_task_evaluator.py
  test_evidence_freshness.py
  test_evidence_conflicts.py
  test_robot_status_projector.py

tests/autonomy/
  test_goal_store.py
  test_goal_state_machine.py
  test_policy.py
  test_action_ledger.py
  test_supervisor.py
  test_reconciliation.py
  test_deliberation_flow.py
  test_cancel_races.py
  test_termination_barrier.py
  test_budget_during_running.py
  test_emergency_from_blocked.py
  test_execution_uncertain_lock.py

tests/skill_os/
  test_command_store.py
  test_persistent_skill_idempotency.py
  test_skill_control.py

tests/integration/
  test_autonomous_goal_nats_flow.py
  test_autonomous_goal_restart.py
  test_autonomous_goal_simulation.py

tests/architecture/
  test_autonomy_boundaries.py
  test_two_agent_tools_only.py
  test_no_agent_fallbacks.py
  test_no_legacy_autonomy.py
~~~

关键断言：

- runtime 不导入 autonomous supervisor；
- runtime 不导入 Skill OS；
- policy 不发布 NATS；
- Supervisor 不导入 provider；
- cognition 不构造 RobotAction；
- RobotAgentService 不发布任何 SkillIntent；
- NATS ACL 只允许 AutonomySupervisor 发布 skill.intent；
- Skill OS 拒绝缺少 goal/task/deliberation ID 的 SkillIntent；
- 普通 UserTurn 的物理请求不能形成 SkillIntent；
- 一个 deliberation 不能产生两个 ActionProposal；
- Supervisor 对一个 deliberation 最多发布一个 SkillIntent；
- fast SkillResult 先到时，publish 返回不能把 terminal Action 倒退为 PUBLISHED；
- 相同 deliberation_id 不启动两个 Runner；
- 重复 GoalCommand.command_id 不创建第二个 Goal 或第二个 control；
- observe skill 只能通过 request_observation；
- Action UNKNOWN 不能回到 PUBLISHING/PUBLISHED；
- cancel during deliberation 不发布迟到 proposal；
- budget/emergency stop pending 时不发布迟到 proposal；
- active action 未确认 idle 时 Goal 不得 terminal；
- ControlResult 的 control/action/gate/Goal 更新必须原子提交；
- 取消 BLOCKED Goal 不清除 execution-uncertain lock；
- SkillController 重启后重复 skill_id 不再次执行；
- 相同 skill_id + 不同 payload 必须进入 IDEMPOTENCY_CONFLICT；
- 相同 deliberation_id/skill_id 的冲突 terminal result 不能被当作 replay 忽略；
- TaskEvaluator 不读取 summary、metadata 或模型文本；
- raw RobotObservation 不能直接满足 object_relation；
- inspect_scene 和仿真 adapter 必须输出可追溯 EvidenceFact；
- 未注册 entity ID 不能进入合同或 EvidenceLedger；
- RobotStatus projector 不能读取 metrics/summary 生成事实；
- presentation 不能修改结构化 status；
- terminal Goal 不能 transition；
- SkillResult 事务任一写点崩溃后不能留下“已 terminal 但未 schedule”的 Goal；
- Agent tool 集合严格等于两个。

## 19. 家庭仿真任务

基础任务：

~~~text
观察当前房间并报告
前往客厅
靠近餐桌
寻找杯子并靠近
将 wand 放回 dock
~~~

长任务：

~~~text
巡视客厅、餐厅和厨房并报告状态
在多个房间寻找杯子
检查 wand 是否在 dock，不在则寻找并放回
~~~

失败任务：

~~~text
寻找不存在的物体
前往不可达位置
相机失效时导航
低电量时执行长任务
VLN 重复输出相同转向
Skill success 但合同相关状态不变化
~~~

每个 autonomous Goal fixture 必须显式提供 SuccessCriterion，不从任务文本推断。

故障注入：

| 故障 | 期望 |
|---|---|
| Provider timeout | Goal FAILED |
| 非法 tool call | MODEL_PROTOCOL failure |
| 多 tool calls | MULTIPLE_ACTION_PROPOSALS |
| Action 处于 PUBLISHING 时 Supervisor 崩溃 | Action UNKNOWN，BLOCKED，不重发 |
| publish 返回前 SkillResult 已 terminal | CAS 保留 terminal，不回写 PUBLISHED |
| DeliberationRequest 重复 | 不启动第二个 Runner |
| SkillIntent 在 SkillController 重启后重复 | receipt replay，不再次执行 |
| SkillResult 重复 | 只处理一次 |
| 相同 skill_id 的冲突 SkillResult | IDEMPOTENCY_CONFLICT + gate UNCERTAIN；非 terminal Goal BLOCKED |
| SkillResult 丢失 | WAITING 超时后 Action UNKNOWN + BLOCKED + gate UNCERTAIN |
| Robot service 在 active action 时重启 | BLOCKED + execution lock，不重发 |
| Camera 黑帧 | Observation skill FAILED |
| Observation 不更新 | INCONCLUSIVE；最终由模型停止或硬预算 FAILED |
| Emergency stop 已确认 | Goal FAILED(EMERGENCY_STOPPED) |
| Emergency stop 结果未知 | BLOCKED + execution lock |
| deliberation 期间 Goal cancel | CANCELLED，迟到 proposal 不发布 |
| active skill 期间 Goal cancel | control receipt + interrupt；确认停止后 CANCELLED |
| active skill 期间 hard budget 耗尽 | interrupt；确认停止后 FAILED(BUDGET_EXHAUSTED) |
| BLOCKED/UNKNOWN 时 emergency stop | 允许 stop control；确认 idle 后 FAILED，未知则保持 BLOCKED |
| 取消含 UNKNOWN 的 BLOCKED Goal | Goal 可 CANCELLED，但新动作仍被 lock 拒绝 |
| Supervisor 重启 | GoalStore 恢复 |

长时间：

~~~text
30 分钟：基本闭环
2 小时：重复事件、状态和资源增长
8 小时：空闲、watchdog、重启和资源泄漏
~~~

指标：

- Goal completion rate；
- 正确完成声明率；
- FailureStage/FailureCode 分布；
- 重复 SkillIntent 数；
- Action UNKNOWN 数；
- 重复 ActionProposal 和相同 arguments 次数；
- Skill success 但 criteria 未变化次数；
- 每 Goal deliberation/skill/token/time；
- 重启恢复率；
- terminal Goal 误唤醒；
- 运行时内存增长。

## 20. 最终系统是什么样子

### 20.1 一次真实运行

以“检查 wand 是否在 dock，不在则放回”为例：

~~~text
1. UI 创建 Goal，并同时持久化不可变合同 wand at dock。
2. Supervisor 保存完整 DeliberationRequest，唤醒一次 RobotAgent。
3. TaskEvaluator 发现证据不足；模型只能提出 request_observation。
4. RobotAgent 返回 observe ActionProposal，不接触机器人。
5. Supervisor 确认 Goal 仍 active，生成稳定 skill_id，先写账本再发布 inspect_scene。
6. Goal 在 WAITING 期间不调用 LLM。
7. Skill OS 返回带 EvidenceFact 的 SkillResult。
8. Supervisor 再唤醒一个新 deliberation。
9. 若事实显示 wand 在 table，模型提出 pick/place 类 request_skill。
10. 每个 SkillResult 后重复同样的有限 slice。
11. wand at dock 的新鲜事实满足合同后，TaskEvaluator 零次调用模型，Goal COMPLETED。
12. UI 展示 GoalEvent、证据和技能摘要；展示文本不改变完成事实。
~~~

任何一步明确失败就 FAILED；物理执行是否发生无法证明就 BLOCKED 并锁住后续动作；系统不会为了“看起来更智能”自动重试、重新观察或换一条路径。

最终代码的核心可以压缩为七个明确所有者：

~~~text
AutonomySupervisor
AutonomyStore
RobotAgentService
StrictAgentRunner
TaskEvaluator
SkillGateway
Skill OS + SkillCommandStore
~~~

没有 TaskRunManager 总管、没有第二套 autonomy、没有八工具兼容层，也没有 memory/RAG 参与控制。

### 20.2 完成标准

重构完成必须同时满足：

1. Goal 在没有后续用户输入时，可以由 SkillResult 唤醒下一 slice；
2. WAITING 时不调用 LLM；
3. 每次 deliberation 最多一个 ActionProposal，Supervisor 最多提交一个 SkillIntent；
4. 每次 deliberation 最多调用模型一次；
5. 自主 RobotAgent 模型只看到 request_observation 和 request_skill；普通 UserTurn 没有物理工具；
6. 任一失败都有唯一 stage/code；
7. 模型 final text 不能证明物理完成；
8. Goal 完成只依据 immutable contract 和证据；
9. Agent/Supervisor/SkillController 重启不会重复动作；
10. 状态不确定时 BLOCKED，并形成跨 Goal 的机器人执行锁；
11. Goal 不会在 action/control 仍可能运行时进入 terminal；
12. 不可完成目标因明确失败或硬预算停止；
13. terminal Goal 不会再次唤醒；
14. trace 可以解释每次唤醒、决策、动作和停止；
15. 主路径没有 fallback、wait policy、legacy alias 或双实现；
16. 没有 TaskRunManager 聚合入口；
17. 家庭仿真连续 2 小时无无限循环、同一 skill_id 重复执行和持续内存增长。

## 21. 推荐第一步

不要先实现 Supervisor。

先完成：

~~~text
Phase 0 固定当前真实行为
  -> Phase 1 完成严格单次 Agent 内核
  -> Phase 2 完成协议、权威存储和物理幂等
~~~

如果一次单独 turn 的失败还不能准确表达，Supervisor 只会把模糊行为放大成长时间循环。

最终系统应当可以用一句话解释：

> Supervisor 持久化一个有明确成功合同的 Goal；每次只唤醒一个有限 Agent slice；Agent 最多提出一个主动观测或物理技能；Supervisor 负责唯一提交；Skill OS 返回真实结果；系统据此继续、完成、阻塞或明确失败。
