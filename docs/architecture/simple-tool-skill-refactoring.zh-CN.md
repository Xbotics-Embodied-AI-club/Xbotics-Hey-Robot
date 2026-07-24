# Hey Robot Tool 与 Skill 总体重构目标

## 1. 项目目标

Hey Robot 的核心不是一次机器人函数调用，而是面向长程任务的 Embodied Agent Harness。
系统需要让 Agent 在真实世界中持续执行“观察、决策、动作、验证、恢复”，同时保持实现简单、
边界稳定。系统已经由多个解耦模块和独立服务组成；重构目标是固定清晰的模块职责和部署边界，
而不是让每个模块同时支持单进程与分布式两套部署路径。

总体目标如下：

- Agent/Cognition 作为慢系统，负责目标理解、任务推进、证据判断和恢复决策；
- Skill、VLA/VLN 和 Robot Runtime 作为快系统，负责有界执行、实时控制和安全；
- Tool 是 Agent 可见的接口，Skill 是可执行的机器人能力；
- classic、VLA 和 hybrid 共享同一套 Skill 生命周期和执行协议；
- 每个模块只有一种明确的生产部署方式：进程内模块直接组合，独立服务使用固定的跨进程协议；
- 系统由配置组合 Policy、Embodiment、Environment 和 Deployment，不为每个组合增加专用 Runtime；
- 任务状态、Skill 事件和执行结果可以持久化并在进程重启后协调恢复；
- Robot Runtime 始终拥有最终动作准入、安全检查和紧急停止权；
- 先完成最小可运行闭环，再根据真实部署需求增加复杂度。

主要设计依据：

- [Harness Engineering for Self-Improvement](../references/Harness%20Engineering%20for%20Self-Improvement.md)
- [Harness VLA](../references/harness_VLA.md)
- [Agent 与 Skill 边界](agent-skill-boundaries.md)
- [ModelService RPC](model-service-rpc-proto.md)

## 2. 设计原则

### 2.1 简单内核

核心 Harness 只保留以下概念：

```text
Tool
Skill
SkillCommand
SkillEvent
SkillResult
SkillClient
SkillRunner
RobotClient
ModelRouter
TaskStore
RunStore
```

新功能应优先组合这些概念。只有在现有模型无法表达真实需求时才增加新抽象。

### 2.2 配置驱动，机制与差异分离

代码只实现稳定机制，部署配置声明可变差异。learned policy 部署由四个正交维度组合：

```text
Policy      = checkpoint + LeRobot policy class + processor
Embodiment  = observation feature mapping + action space mapping
Environment = RoboCasa / simulation / real robot
Service     = module ownership + fixed production transport
```

RoboCasa365、WAM、SO101、仿真和真机不对应不同的 Policy Executor。新增一个 LeRobot 已支持的模型或
新的机器人形态时，应优先只增加配置、feature/action mapping 或 LeRobot plugin，不复制模型加载、
session、推理和错误处理代码。

配置不是另一套运行时逻辑。composition root 负责读取配置、构造 adapter，并在启动时校验 checkpoint、
processor、observation feature、action dimension、action space 和服务边界是否兼容。无效组合必须
fail fast，不能在第一次物理动作时才失败。

### 2.3 Tool 与 Skill 一一投影

每个允许 Agent 使用的 Skill 投影为一个独立 Tool：

```text
inspect_scene(...)
navigate_to(...)
pick(...)
place(...)
manipulate(...)
stop_motion(...)
```

Tool 只负责参数解析并生成 Skill 调用提案，不包含机器人控制、模型路由、重试或任务状态机。
Agent 不直接构造 RobotAction，也不依赖驱动 primitive。

### 2.4 一个 Skill 执行入口

所有 Skill 都通过 SkillRunner 执行。SkillRunner 统一负责：

- 参数校验；
- timeout 和 deadline；
- cancel；
- 资源互斥；
- 嵌套 Skill；
- 事件序列；
- 终态结果；
- run artifact 持久化。

classic、VLA、VLN、hybrid 只是 handler 实现差异，不得形成不同的 Runtime 或任务协议。

### 2.5 异步快慢双系统

Agent 提交 Skill 后不等待完整机器人动作同步结束：

```text
Slow system
  Agent -> persist pending step -> submit SkillCommand -> return waiting

Fast system
  Worker -> SkillRunner -> Robot/Model -> emit SkillEvent

Resume
  terminal SkillEvent -> persist result -> resume Agent decision loop
```

长时间 VLA/VLN loop 必须是有界 Skill。每次执行需要明确 deadline、最大步数、终止原因和证据。

### 2.6 分层解耦

目标依赖方向为：

```text
Cognition
  -> SkillClient + Skill definitions

Skill Runtime
  -> RobotClient + ModelRouter + RunStore

Robot Runtime
  -> Driver + safety/readiness/control plane

Foundation Models
  -> policy runtime backend (LeRobot) / VLN backend
```

下层不得导入 Cognition。Skill handler 不得直接依赖 NATS、gRPC、具体机器人 Driver 或 Agent。

### 2.7 模块部署边界唯一

领域接口仍与具体库解耦，但不要求每个接口都实现 local/remote 双 adapter。现阶段生产拓扑固定为：

- Agent、TaskCoordinator、SkillRunner 和 Robot Runtime 由 `DeploymentRunner` 在同一进程组合，
  使用直接调用或进程内 async queue；
- LeRobot 等 Foundation Model Service 是独立服务，使用 gRPC；
- RoboCasa environment backend 是独立服务，使用 gRPC；
- Gateway、Channel、通知和跨服务事件使用 NATS；
- TaskStore/RunStore 由所属进程直接持久化，不通过消息总线保存事实。

模块一旦确定为进程内组件，就不再为其预留远程 adapter；确定为独立服务，就只维护其正式 RPC
contract。NATS 不进入 Skill handler，gRPC 不进入 Agent Tool，协议装配只发生在 composition root。

### 2.8 安全在 Agent 之外

Agent 的判断不能替代确定性安全规则：

- SkillRunner 限制资源、timeout 和 cancellation；
- Robot Runtime 检查 primitive、参数范围、readiness、frame freshness 和控制权；
- emergency stop 必须绕过普通资源排队并直接进入 Robot Runtime 高优先级控制面；
- Task completion 必须由独立 verifier 根据动作后的 observation/evidence 判断。

## 3. 目标系统结构

模块是代码和职责边界，不自动等于进程边界。现阶段生产拓扑固定如下：

| 模块 | 生产部署 | 通信方式 |
|---|---|---|
| Agent、TaskCoordinator、SkillRunner、Robot Runtime | 同一个 `DeploymentRunner` 进程 | 直接调用、进程内 async queue |
| LeRobot/Foundation ModelService | 独立服务进程 | gRPC |
| RoboCasa environment backend | 独立服务进程 | gRPC |
| NATS | 独立基础设施 | Gateway、Channel、通知和跨服务事件 |
| TaskStore、RunStore | Core Harness 进程本地 | SQLite、JSONL 和 artifact 文件 |

配置可以选择 checkpoint、robot、environment、endpoint 和是否启用服务，但不能把同一模块切换到
另一套隐藏的生产实现。部署形态需要改变时，必须先修改这张边界表并删除被替代路径，而不是长期兼容两套。

```text
Channel / API
      |
      v
Gateway ---- conversation/event bus ---- Autonomous Agent
                                           |
                                      independent Tools
                                           |
                                     TaskCoordinator
                                           |
                                      SkillClient
                                           |
                               in-process Skill Worker
                                            |
                                       SkillRunner
                                      /      |       \
                                RobotClient ModelRouter RunStore
                                     |          |
                               Robot Runtime   independent gRPC ModelService
                                     |
                              Driver + safety plane
```

该结构不是同一模块的两种版本。Core Harness 是一个进程内组合；ModelService 和 RoboCasa backend
是独立服务。测试可使用 fake/in-process implementation，但不将测试替身提升为第二条生产部署路径。

Robot Policy ModelService 的配置形态应收敛为：

```yaml
robots:
  robocasa365:
    type: robocasa
    driver: grpc
    settings:
      target: grpc://127.0.0.1:9092

model_services:
  robocasa365:
    type: robot_policy
    robot_id: robocasa365
    target: grpc://127.0.0.1:9091
    provides: [manipulate]
    timeout_sec: 1800
    settings:
      runtime: lerobot
      policy_path: lerobot/pi052_robocasa
      policy_device: cuda
      embodiment: robocasa
      robot_type: robocasa
      state_dimensions: 16
      camera_names: [camera1, camera2, camera3]
      observation_features:
        observation.images.robot0_agentview_left: observation.images.camera1
        observation.images.robot0_agentview_right: observation.images.camera2
        observation.images.robot0_eye_in_hand: observation.images.camera3
      media_root: runtime/robocasa365.agent/media
      action_space: robocasa_12d
      action_dimensions: 12
      prompt_mode: environment_root
```

同一个 executor 用于 RoboCasa benchmark、其他仿真评测和真机控制。切换到 WAM 或 SO101 时，替换
checkpoint、feature mapping、action space 和服务 endpoint 配置；只有协议确实不同的边界才增加
adapter。比如 PI0.5 与 MuJoCo EGL 需要依赖隔离，由既有独立 ModelService 进程解决，不产生第二种
executor 或第二条 Harness 调用路径。

LeRobot 是 policy runtime，不是 VLA 的子类型，因此代码固定放在
`foundation/backends/lerobot/`。VLA、WAM、ACT、Diffusion 等 family 不进入 Hey Robot 的服务类型、
目录结构或 executor 分支；`robot_policy` 服务从 checkpoint 的 `config.json.type` 读取 policy type，
并统一交给 LeRobot factory。配置不得重复声明 `policy_type`，也不保留 `vla_policy` 旧服务类型。

## 4. Tool 与 Skill 模型

### 4.1 Skill

一个 Skill 至少声明：

```text
name
description
parameters
handler
resources
timeout
supported_robots
required_actions
required_models
dependencies
```

其中：

- `required_actions` 是 Skill 可能调用的 Robot Runtime primitive；
- `required_models` 是 VLA/VLN 等模型能力；
- `dependencies` 是可能通过 `ctx.run()` 调用的子 Skill；
- 部署校验必须递归验证完整依赖闭包。

### 4.2 SkillResult

SkillResult 必须明确表达：

- success 和 terminal status；
- 面向 Agent/用户的 summary；
- 结构化 data；
- evidence、observation 和 artifact 引用；
- failure mode、error 和 retryable 语义；
- termination reason；
- 动作后是否需要重新观察。

### 4.3 Skill 事件

每个 run 形成单调递增的事件流：

```text
accepted -> running -> progress* -> completed | failed | cancelled
```

消费者按 `(run_id, sequence)` 幂等处理。终态不可逆，重复事件不得重复推进 Agent。

## 5. Classic、VLA 与 Hybrid

### 5.1 Classic

Classic Skill 使用确定性算法或 Robot Runtime primitive。它与 VLA Skill 使用同一 Runner、
事件、持久化和 Tool surface。

### 5.2 VLA/VLN

VLA/VLN 是有界 Skill handler：

```text
observe -> infer -> validate action -> execute -> fresh observe -> terminate/continue
```

必须满足：

- 模型只输出结构化 action 或 action chunk；
- Robot Runtime 对每个 action 做最终准入；
- 每轮使用新的 observation；
- cancel 能终止模型请求并停止机器人；
- 达到 max_steps 不等同于根任务完成；
- observation、model output、action 和 result 可追溯。

### 5.3 一个 LeRobot Policy Executor

生产代码只保留一个 `LeRobotPolicyExecutor`，职责限定为：

- 通过 LeRobot 标准 factory 加载 checkpoint 和 policy class；
- 通过 LeRobot pre/post processor 处理 observation 和 action；
- 管理 episode/session reset；
- 推理单个 action 或 action chunk；
- 响应 cancel/close，并返回统一的结构化结果。

统一结果至少包含：

```text
action_space
embodiment
actions
horizon
dt
done
raw metadata
```

该 executor 不拥有相机、机械臂或 RobotClient，不执行物理动作，也不包含 RoboCasa、WAM 或 SO101
专用控制循环。Environment adapter 将 observation 映射为 policy feature；Robot Runtime 根据
`action_space` 和 embodiment mapping 将输出映射到 driver primitive，并执行最终安全准入。

一个模型可进入通用执行链的条件是：它能通过 LeRobot 注册表和 `PreTrainedConfig` 加载、能构造
标准 processor、提供标准 action 推理接口，并且 observation/action contract 能映射到目标
embodiment。不能满足时，应先扩展 LeRobot plugin 或显式 adapter，而不是在 Hey Robot 中增加平行 executor。

### 5.4 Hybrid

Hybrid Skill 通过 `ctx.run()` 组合 classic 与 VLA/VLN Skill。依赖必须显式声明并在启动时验证，
不得在 handler 中形成不可见的运行时依赖。

## 6. 通信与部署策略

### 6.1 NATS

NATS 继续用于 Gateway、Channel、通知和已经存在的跨服务事件。Core Harness 内的 Skill command、
Skill event 和 emergency stop 不经过 NATS。

只有出现已经验证、必须将 Skill Worker 独立部署的需求时，才重新评审边界；在此之前不为假设需求
维护第二套 Skill transport。

### 6.2 RPC

gRPC 继续用于边界清晰、请求响应明确的大负载服务：

- VLA/VLN ModelService；
- 独立 RoboCasa environment service。

RPC 不承担长程任务事实来源，也不替代 SkillEvent 流。

### 6.3 持久化

第一阶段使用简单文件和 SQLite：

```text
runtime/<deployment>/
  conversations.sqlite3
  tasks.sqlite3
  runs/<run_id>/
    events.jsonl
    result.json
  media/artifacts/
```

TaskStore 是慢系统任务事实来源，RunStore 是快系统 run 事实来源，两者由 Core Harness 进程持有。
大型 observation、model output 和 action trace 由统一 media store 保存，RunStore 只记录 artifact reference。
消息总线只负责已有跨服务消息，不保存任务或 Skill run 事实；独立服务不得共享这个 SQLite writer。

## 7. 范围边界

近期不引入：

- 通用工作流 DSL；
- 独立调度服务；
- 多层 capability/operation/skill taxonomy；
- 同一个模块同时维护进程内和分布式两套生产部署路径；
- 为尚未存在的部署预留多套兼容 Runtime；
- 按 checkpoint、benchmark、机器人型号或部署方式复制 Policy Executor；
- 在 Policy Executor 中持有硬件或实现机器人控制循环；
- 与本地语义不同的“临时”远程协议；
- 没有故障恢复测试的消息 bridge。

具体剩余工作、文件修改和验收顺序见
[Tool 与 Skill 代码级重构计划](simple-tool-skill-refactoring-code-plan.zh-CN.md)。
