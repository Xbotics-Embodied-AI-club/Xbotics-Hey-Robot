# Hey Robot 系统深度分析：以当前代码为准

> 分析日期：2026-07-11  
> 分析原则：代码、配置、测试和协议定义优先；README 与既有文档只作为线索。  
> 目标读者：具备计算机与 AI 本科基础，希望理解具身智能系统工程实现的开发者。

## 1. 先给结论

Hey Robot 不是“让大模型输出电机角度”的简单 Demo，也不是一个通用聊天 Agent 外接机器人 API。它更准确的定位是：

> 一个围绕真实机器人异步执行、状态观测、技能调度、失败恢复和多渠道交互设计的 Embodied Agent Harness（具身 Agent 运行框架）。

系统把能力拆成四层：

1. **Agent / Cognition**：理解用户目标，组织上下文，让 LLM 在工具循环中选择技能，并判断是否需要继续、恢复或结束任务。
2. **Skill OS**：把“导航到桌子”“操作物体”等语义能力变成有参数、有资源约束、有超时和失败语义的可调度任务。
3. **Foundation Model**：以独立 gRPC 服务运行 VLA、VLN 等模型，返回短时域决策结果。
4. **Robot Runtime**：把统一的机器人动作落实到 Mock、MuJoCo 或真实 XLeRobot 硬件，并持续发布状态与图像观测。

四层不是普通的同步函数调用栈，而是主要通过 NATS 消息协作。模型服务单独使用 gRPC。这样做的核心价值不是“微服务”这个标签，而是让慢速语言推理、GPU 模型、机器人控制和用户交互不必共享同一个执行节奏与依赖环境。

当前系统的工程完成度明显高于概念原型：主代码约 295 个 Python 文件、4.6 万行，测试约 162 个文件、2.9 万行；完整测试在关闭覆盖率插件后共收集并通过 **1366 项**。但它仍属于研发和联调阶段，而不是工业级自主机器人产品，主要限制包括：

- 自治仍然由用户 turn 驱动，不是持续唤醒 LLM 的后台目标循环；
- VLN 是局部、迭代式视觉导航规划，不等于 SLAM 或全局避障；
- VLA 已有真实模型加载和 action chunk 路径，但端到端效果取决于模型、数据、状态定义和具体硬件标定；
- 默认 Web、NATS 和 gRPC 开发配置缺少完整的生产认证边界；
- 本地 JSON/JSONL 状态适合单机可审计运行，不适合直接当作分布式事务存储；
- Docker Compose 存在配置路径和服务 ID 漂移，按默认值启动并不可靠；
- 测试中 MuJoCo 偶尔报告数值不稳定警告，测试通过不等于所有物理轨迹稳定。

## 2. 我是如何判断的

本分析重点追踪了以下事实来源：

- CLI 和本地编排入口：`src/hey_robot/cli/main.py`、`src/hey_robot/app/runner.py`
- 跨服务协议：`src/hey_robot/protocol/messages.py`、`topics.py`
- 用户入口：`src/hey_robot/gateway/service.py`、`src/hey_robot/channels/`
- Agent 主链路：`src/hey_robot/cognition/robot_agent.py`、`loop.py`、`core.py`
- LLM 工具循环：`src/hey_robot/cognition/runtime/runner.py`
- 技能入口和调度：`src/hey_robot/cognition/skill_gateway.py`、`src/hey_robot/skill_os/controller.py`
- 机器人执行：`src/hey_robot/robot_runtime/service.py`、`runtime.py`、各 driver
- VLA/VLN 服务：`src/hey_robot/foundation/`、`proto/hey_robot/model_service/v1/model_service.proto`
- 部署事实：`configs/*.yaml`、`docker-compose.yml`、`pyproject.toml`
- 架构约束和行为证据：`tests/architecture/`、`tests/integration/` 及完整测试集

这种阅读顺序很重要。单看类名容易得到“系统功能很多”的印象；只有沿着消息发布、订阅、状态落盘和测试断言走一遍，才能判断某个能力是主链路、实验路径，还是只存在于接口层。

## 3. 最值得记住的整体模型

可以把系统理解成一个“慢速经理 + 技能执行部门 + 专项模型 + 机器人现场”的组织：

```text
用户（Web / Voice / Feishu / CLI）
        |
        v
Gateway：统一身份、会话和消息格式
        |
        | NATS: user.turn
        v
Agent：理解目标、看状态、选择下一项技能
        |
        | NATS: skill.intent
        v
Skill OS：校验合同、抢占资源、执行和超时管理
        |                         |
        | NATS: robot.action      | gRPC ModelService
        v                         v
Robot Runtime                 VLA / VLN
        |                         |
        +-----------+-------------+
                    |
                    v
       status / observation / event / result
                    |
                    v
      Agent 继续规划、恢复或回复用户
```

这里有两个互相正交的视角：

- **快慢系统**描述决策时间尺度。LLM、长期任务和恢复属于慢系统；VLA/VLN、技能闭环和机器人执行更接近快系统。
- **四层架构**描述代码所有权。Agent 不拥有驱动，Skill OS 不负责自然语言推理，模型服务不拥有根任务状态，Robot Runtime 不理解用户意图。

“快系统”不代表硬实时。当前控制和编排大量使用 Python `asyncio`、NATS 和普通操作系统调度，不能当成确定性实时控制器。

## 4. 真实部署形态

统一入口是 `hey-robot`。CLI 会动态转发到 `agent`、`gateway`、`robot`、`task-supervisor`、`model-service`、`run` 等子命令。

最常用的 `hey-robot run --config ...` 由 `DeploymentRunner` 在一个 asyncio 进程中按配置创建：

- `RobotService`
- 可选 `HumanFollowService`
- `SkillControllerService`
- `TaskSupervisorService`
- 一个或多个 `RobotAgentService`
- `GatewayService`

需要注意：这些服务虽然默认在同一进程，却各自创建 NATS client，通过消息协议协作。因此准确说法是：

> 协议上可拆分部署，默认主运行时本地一体化。

NATS broker 是外部进程。VLA/VLN ModelService 也是外部进程，因为两组依赖在 `pyproject.toml` 中被声明为冲突组：VLA 与 VLN 需要不同版本的 Transformers、Hugging Face 生态和其他 GPU 依赖，应使用不同虚拟环境或容器。

## 5. 一条用户指令如何走完整个系统

以“靠近桌上的杯子”为例。

### 5.1 Gateway：把外部消息变成系统内部消息

Web、语音、飞书和 CLI 各自解析输入，但最终都产生 `UserTurn`。`GatewayService` 随后：

1. 解析或绑定统一 `user_id`；
2. 选择目标 `agent_id` 和默认 `robot_id`；
3. 根据用户、渠道、会话等维度分配 episode；
4. 把原始 turn 写入 episode JSONL；
5. 更新交互状态；
6. 发布 NATS `user.turn`。

贯穿消息的 `Envelope` 类似分布式追踪上下文，保存 `trace_id`、`episode_id`、`turn_id`、渠道身份、`robot_id`、`agent_id` 和时间戳。后续状态、技能和回复都尽量沿用它，因此 Web 任务页能够把一次物理任务的多个异步事件重新聚合起来。

### 5.2 Agent Service：并发控制和 turn 生命周期

`RobotAgentService` 订阅用户 turn、机器人状态、观测、技能事件和技能结果。

收到 turn 后，它先做三件工程上很重要的事：

- 去重，避免重复消息触发两次物理动作；
- 检查机器人是否已经被活动技能租用；
- 同时获取 episode 锁和 robot 锁，避免同一会话或同一机器人并发执行冲突 turn。

如果机器人正忙，系统不会简单丢弃新消息。`BusyTurnHandler` 会区分：

- 只读状态查询：直接用缓存状态回答；
- 修正、追问和重试：排队到下一个安全边界；
- interrupt / emergency stop：发布中断 intent，并释放租约。

正式 turn 由 `RobotAgentLoop` 以显式状态机运行：

```text
restore -> build -> run -> save -> done
```

- `restore`：标记和恢复任务上下文；
- `build`：组合历史、长期记忆、场景记忆、恢复信息和最新机器人快照；
- `run`：进入 `RobotAgentCore` 和 LLM 工具循环；
- `save`：保存任务状态、checkpoint 和机器人 episode 状态。

显式状态机的意义是让恢复和审计成为 turn 生命周期的一部分，而不是散落在异常处理代码中。

### 5.3 RobotAgentCore：规则优先，模型其次

Agent Core 并不是一上来就调用 LLM。实际顺序是：

1. 任务和渠道安全检查；
2. 对 stop、reset、home、gripper 等确定性短命令进行规则路由；
3. 构造任务、记忆、感知、自治目标和恢复上下文；
4. 才进入 `AgentRuntime` 的模型工具循环。

这是一种合理的机器人设计：能确定处理的高优先级命令不必承受模型随机性和网络延迟。

当前 Agent 自动发现并加载的核心工具有 8 个：

| 工具 | 作用 |
|---|---|
| `get_robot_status` | 查询机器人状态 |
| `get_task_context` | 查询当前任务和恢复上下文 |
| `propose_skill` | 请求用户确认动作 |
| `request_perception` | 主动获取视觉证据 |
| `request_skill` | 唯一正式物理技能入口 |
| `search_memory` | 搜索长期记忆 |
| `wait` | 等待异步状态 |
| `write_memory` | 写入结构化记忆 |

测试明确守卫：Cognition 层不能绕过 `SkillGateway` 直接构造技能请求，也不能依赖 `RobotAction` 或 driver primitive。

### 5.4 AgentRuntime：不是一次问答，而是工具循环

`AgentRuntime` 把 system prompt、任务、机器人状态、记忆和工具 schema 交给模型。模型可以：

1. 返回文本；
2. 调用一个或多个只读工具；
3. 调用 `request_skill`；
4. 等待技能结果后，在同一 turn 内继续调用下一项工具；
5. 最后给出用户可见结论。

运行时包含若干重要治理逻辑：

- 总 turn 时间预算和单次 provider timeout；
- 最大迭代次数；
- 工具参数 JSON Schema 校验与适度类型纠正；
- 只读且可并发的工具分批执行，独占工具串行执行；
- 工具调用与 tool result 必须成对；
- 拒绝把模型生成的“伪工具协议文本”当作已执行动作；
- 对需要视觉依据的回答插入 grounding 感知；
- 根据任务合同和证据账本判断根目标是否真正完成；
- provider 失败后，若已有可靠工具结果，尽量生成保守的 fallback 回复。

因此，LLM 的职责是选择和组合能力，而不是自行宣布“动作已完成”。完成判定还要参考真实的技能结果、状态和证据。

### 5.5 SkillGateway：Agent 与物理世界之间的单一闸门

`request_skill` 最终进入 `SkillGateway.submit()`。这里会：

1. 检查 skill 是否存在于当前部署能力面；
2. 校验 objective 和参数；
3. 在恢复状态下只允许观察、停止、复位、松爪等安全能力；
4. 应用语音渠道确认策略；
5. 对运动能力检查相机健康度和新鲜度；
6. 禁止没有新感知证据的连续运动；
7. 创建 `SkillIntent` 并发布到 NATS。

它支持三种等待语义：

- `wait_result`：等待最终 `SkillResult`，允许 LLM 在同一 turn 内继续规划；
- `wait_acceptance`：技能受理即返回，适合短命令；
- `return_handle`：只返回 `skill_id`。

当前长任务闭环主要依赖 `wait_result`。它让一次用户指令内部可以形成多轮“观察 -> 动作 -> 反馈 -> 再观察”，但也意味着整个 LLM turn 可能持续较长时间。

### 5.6 Skill OS：把能力当成有合同的任务

每个 `BaseSkill` 都带一个 `SkillSpec`。它是当前技能合同的事实来源，主要描述：

- 输入 schema 和必填参数；
- 所需资源，例如 camera、base、arm、gripper；
- 支持的机器人类型；
- 是否对 Agent 可见；
- 安全级别；
- 超时、可中断性；
- 所需模型服务；
- driver primitives；
- 成功标准、失败模式、恢复建议和证据输出。

`SkillControllerService` 收到 `SkillIntent` 后会校验合同、机器人 readiness 和资源冲突，创建 `SkillRun`，再通过插件 runtime 执行。一个 skill 可以：

- 直接组合 Robot Runtime primitive；
- 调用 ModelService，再把模型结果适配为 primitive；
- 运行 Human Follow 这类持续闭环插件。

资源调度不是单一全局锁。camera 是共享资源，而 base、arm、gripper 通常互斥；左右臂参数还可以实例化为不同资源。因此不同手臂理论上能并行，同一手臂动作会冲突。

### 5.7 Robot Runtime：最终只有这里接触执行器

Skill OS 把 primitive 编码进 `RobotAction.metadata.skill`。`RobotService` 订阅 `robot.action`，交给目标 `RobotRuntime`：

1. RobotRuntime 再做一次确定性安全检查；
2. 处理内置感知技能或调用 driver；
3. driver 执行 Mock、MuJoCo 或真机动作；
4. 返回 `RobotStatus`；
5. 立即刷新并发布 `RobotObservation`。

Robot Service 还运行 merged observation loop。每个周期只调用一次 driver `observe()`，同时产出两条数据路径：

- `robot.observation`：结构化、小消息，图像只保存引用；
- `robot.camera.frame.<robot_id>`：给 Human Follow 等低延迟消费者使用的原始帧包。

这个设计避免 Agent 消息携带大块 base64 图像，也避免不同消费者各自重复抓取相机。

## 6. 协议为什么是系统的骨架

系统的主要消息类型如下：

| 消息 | 含义 |
|---|---|
| `UserTurn` | 用户输入 |
| `AgentReply` | Agent 回复或通知 |
| `SkillIntent` | 请求执行一个技能 |
| `SkillEvent` | 技能生命周期事件 |
| `SkillResult` | 技能最终结果 |
| `RobotAction` | 发给机器人运行时的动作 |
| `RobotStatus` | 机器人状态与执行结果 |
| `RobotObservation` | 图像引用、本体状态和其他观测 |

对应 NATS topic 包括 `user.turn`、`agent.reply`、`skill.intent`、`skill.event`、`skill.result`、`robot.action`、`robot.status`、`robot.observation` 等。

默认 NATS 使用 Core Publish/Subscribe，不自动持久化或重放。代码支持可选 JetStream，但当前 `_ensure_js_stream()` 只在首次 topic 上创建 stream 并缓存“已就绪”，多 subject 场景需要谨慎验证。系统真正的 durable state 主要依靠本地文件存储，而不是消息总线。

## 7. VLA 与 VLN 的真实位置

### 7.1 ModelService 边界

ModelService 的 proto 只有三个 RPC：

- `GetHealth`
- `ExecuteSkill`
- `CancelSkill`

`ModelServiceRegistry` 按 `provides + robot_id` 路由。调用前，Skill OS 会检查服务是否 online、loaded、busy。取消技能或超时也会转发到模型服务。

这是一个刻意收窄的边界：模型服务只负责一次能力调用，不拥有用户会话、根任务、记忆和机器人全局调度。

### 7.2 VLN：局部视觉规划器，不是完整导航栈

当前 VLN 后端是 `InternVLA-N1-System2`。典型链路为：

```text
当前图像 + 语言目标
  -> VLN ModelService
  -> pixel goal / heading / stop / look-down
  -> navigation adapter
  -> turn_base / move_base / stop_motion
  -> 刷新图像
  -> 下一次规划
```

`navigate_to` 和 `approach_object` 会进行多步迭代，并可在每个 primitive 后获取新观测。它具备“视觉闭环导航”的系统形态，但没有在主代码中看到完整 SLAM、地图维护、全局路径规划和通用动态避障栈。因此最准确的能力描述是：

> 基于当前视觉的局部、短步长、迭代式导航规划。

### 7.3 VLA：已有真实推理链路，但模型效果不是框架能保证的

当前公开语义能力是统一的 `manipulate`，不是旧文档中的 `pick_object/place_object`。实验配置指向：

```text
models/smolvla-so101-pick-orange
```

VLA executor 支持：

- LeRobot policy 加载；
- action chunk policy；
- 图像、任务文本和显式机器人 state 预处理；
- state/action 归一化与反归一化；
- 将关节和夹爪输出适配为 Robot Runtime primitive；
- mock inference 和独立 HTTP policy endpoint 路径。

这说明代码不再只是空接口。但“能加载模型并生成动作”与“能在真机稳定完成开放世界操作”是两回事。实际成功率仍取决于：

- 训练数据与当前相机视角是否一致；
- state/action schema、单位和关节顺序是否一致；
- 机器人标定和延迟；
- 物体、背景、光照的分布偏移；
- action chunk 长度与闭环重规划频率。

所以应把 VLA 评价为“工程链路已实现、任务效果仍需按模型和场景验收”。

## 8. 当前技能能力面

内置 registry 共注册 23 个语义或实现技能。重要的 Agent 可见语义技能包括：

| 能力 | 类型 | 依赖 |
|---|---|---|
| `inspect_scene` | 感知 | camera |
| `look_around` | 主动感知 | camera + base |
| `navigate_to` | 导航 | VLN ModelService |
| `approach_object` | 导航 | VLN ModelService |
| `human_follow` | 视觉跟随 | camera + base |
| `stop_motion` | 停止 | Robot Runtime |
| `reset_posture` | 复位 | arm + gripper |
| `manipulate` | VLA 操作 | VLA ModelService + camera + arm + gripper |
| `pick` / `place` | SO101 桌面操作 | IK 与仿真/驱动 primitive |
| `pick_wand_from_dock` / `place_wand_to_dock` | XLeRobot/SO101 mobile 专用操作 | 几何、IK 与 dock primitive |

`move_base`、`turn_base`、`base_velocity_step`、`set_arm_pose`、`move_arm_joints`、`set_gripper`、`detect_marker` 等被标为实现或联调能力，默认不应在 production 模式直接暴露给 Agent。

不过仓库中的主要 real/sim 配置目前都使用 `skills.mode: bringup`，并显式开放若干底层 skill。这说明当前部署配置的目标仍然是开发、验收和硬件联调，而不是收敛后的生产能力面。

## 9. 不同配置实际能做什么

### 9.1 Mock

`mock.dev.yaml` 和 `mock.test.yaml` 使用有世界状态的 Mock Robot。它能模拟物体可见性、扫描后出现、抓取/放置结果、相机故障、电池和脚本化失败，不只是一个永远返回成功的 stub。它适合验证 Agent、恢复和消息链路，但不能证明物理控制有效。

### 9.2 XLeRobot 真机

Windows、Ubuntu 和 S600 真机配置主要开放：

- 场景观察与 marker 检测；
- 短步底盘运动；
- Human Follow；
- 停止和复位；
- 机械臂命名姿态、关节和夹爪控制。

这些主配置没有启用 ModelService，因此不能把默认真机配置描述为已经拥有 VLA/VLN。

### 9.3 普通 MuJoCo 仿真

Windows/Ubuntu 仿真配置除基础能力外，还开放 wand dock 的取放技能。MuJoCo driver 包含底盘、双臂、夹爪、相机和 dock weld 等较具体的场景逻辑，属于项目当前较完整的物理闭环验证环境。

### 9.4 实验 VLA + VLN 仿真

`xlerobot.sim.vla_vln.yaml` 同时声明：

- `vln_nav`：提供 `navigate_to`、`approach_object`；
- `manipulate`：提供 `manipulate`。

它是研究/集成配置，不代表这些模型已在所有硬件和任务上完成验收。

## 10. 感知、记忆、任务和恢复

### 10.1 感知不是把图片直接塞给 LLM

`ObservationPipeline` 会检查空图、shape、黑帧等质量问题，把图像保存到 local media store，再在消息中只携带 `ImageRef`。Scene Captioner 可以把图像转成对象、风险、摘要和置信度，Scene Memory 再按 episode 保存这些证据。

运动技能前的相机 gate 使用状态中的 camera health 和 age：健康图像通常要求不超过 15 秒，超过 30 秒或明确无效会阻止运动，并要求先执行 `inspect_scene`。

### 10.2 记忆分为多种，而不是一个聊天历史数组

主要持久化包括：

- episode 对话：`*.jsonl` 和 `*.meta.json`
- 任务：`*.task.json`
- Agent checkpoint：`*.agent_checkpoint.json`
- 机器人 episode 状态：`*.robot_state.json`
- 任务事件：`*.events.jsonl`
- 场景记忆：`*.scene.jsonl`
- 长期记忆：`runtime/.../memory/long_term.jsonl`
- 技能生命周期：`skills.json`、`skill_events.jsonl`
- Runtime 事件：`events.jsonl`

长期记忆支持 entity、place、task result、skill experience、user preference、scene anchor 和 task lesson，并带有去重与相关性排序，而不是简单全文拼接。

### 10.3 Task Supervisor 负责监控，不负责自主续跑

Task Supervisor 周期检查：

- 活动 skill 是否超时；
- status / observation 是否过期；
- camera quality 是否阻塞；
- episode 是否要求 recovery。

异常会更新 recovery 策略、发布 watchdog 事件并通知用户。但是它不会在没有新用户 turn 时自动再次唤醒 LLM 执行下一步。

`AutonomyManager` 保存目标和最近事件，并把它们注入 prompt；但目标列表本身当前主要是进程内状态，没有看到独立持久化和后台调度循环。因此配置中的 `mode: autonomous` 应理解为“允许 Agent 自主使用工具完成一个 turn”，而不是“机器人永久自主运行”。

## 11. 安全设计：多层确定性 gate

系统的安全链可以概括为：

```text
用户任务/渠道安全
  -> Agent 工具前置安全 hook
  -> SkillGateway 感知与恢复 gate
  -> Skill contract / readiness / resource gate
  -> RobotRuntime 电池、急停和状态 gate
  -> driver 参数和硬件约束
```

已实现的典型规则包括：

- 语音运动默认要求显式确认；
- 开门等当前不支持的高风险任务直接拒绝；
- emergency stop、collision、protective stop 时阻止普通动作；
- skill objective 必须原子化，禁止用一条复合目标偷偷串联多个动作；
- 连续运动之间要求新的感知证据；
- 相机不可用或过期时阻止运动；
- recovery required 时限制可执行技能；
- 电池严重不足时只保留停止等安全路径；
- 技能资源冲突、参数缺失和 readiness 不满足时拒绝受理；
- 中断和超时会尝试取消正在运行的 ModelService。

这些规则能显著降低 LLM 误调用风险，但不是工业功能安全系统。代码中没有完整的碰撞预测、安全认证控制器、独立硬急停回路证明、全局避障或实时保证。真机仍必须保留物理急停/断电能力并隔离网络。

## 12. 多渠道与前端

Gateway 支持：

- CLI
- Web
- Voice
- Feishu

Web 是 FastAPI + 原生 HTML/CSS/JS，提供：

- `/chat`：对话和实时进度；
- `/tasks`：任务列表；
- `/tasks/{episode_id}`：任务详情；
- `/admin`：运行时摘要；
- `/ws`：实时消息；
- `/turn`：HTTP 输入 fallback；
- task、history、event、identity binding 等 API。

任务视图不是另一个数据库，而是聚合 TaskRun、机器人状态、Scene Memory、Skill Store 和 Interaction State 的查询层。

身份系统能把不同渠道 sender 映射为统一用户，并支持绑定码。这解决了“用户从飞书换到 Web 后是否还是同一任务”的连续性问题。不过 Web 路由本身没有看到完整登录鉴权和角色授权中间件，默认配置有时监听 `0.0.0.0`，因此不能直接暴露到不可信网络。

## 13. Provider 层

Agent 推理通过项目自建的 provider 抽象访问 OpenAI、DeepSeek、DashScope、Ark/Doubao、MiniMax、OpenRouter 或自定义 OpenAI-compatible endpoint。

默认兼容路径使用 Chat Completions；直接 OpenAI GPT-5/o 系列或显式 reasoning effort 可以切换 Responses API。Provider 层还处理：

- tool schema 转换；
- strict tools；
- required tool choice 差异；
- provider timeout 和错误归一化；
- fallback model 链；
- 图片编码；
- DeepSeek 特殊兼容处理。

项目没有把 Agent 编排交给 OpenAI SDK。SDK 只是模型客户端，工具循环、状态、权限、记忆和证据判断都在本仓库中实现。

## 14. 代码质量与测试能说明什么

### 14.1 已验证事实

执行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q --no-cov
```

结果：

```text
1366 passed in 99.35s
```

覆盖的高价值场景包括：

- 四层 import 边界；
- Agent 不能绕过 SkillGateway；
- proto 与 generated gRPC contract 一致；
- NATS 协议消息 round-trip；
- Agent 多轮工具调用、超时、grounding 和证据完成判定；
- Skill 合同、资源冲突、超时、中断和取消；
- gRPC ModelService 集成；
- Mock、MuJoCo 和 XLeRobot 运行时行为；
- VLA payload、action chunk 和 schema；
- Voice、Web、飞书、身份和通知；
- 场景、任务和长期记忆。

此外，选取 168 个架构、部署、集成和核心运行时测试执行时，行为断言全部通过；命令最终退出 1 是因为只运行子集导致全局覆盖率为 48.99%，没有达到 `pytest.ini` 对完整套件设置的 85% 门槛，不是测试失败。

### 14.2 不能从测试推出的结论

- 大部分单元测试使用 fake、mock 或固定输入，不能证明真实环境成功率；
- ModelService 集成测试不能替代真实大模型权重和 GPU 推理验收；
- 仿真通过不能替代真机动力学、延迟和标定测试；
- 测试中出现过 `Nan, Inf or huge value in QACC` 的 MuJoCo 警告，说明至少某些测试轨迹存在数值不稳定，即使断言最终通过；
- 本次完整覆盖率模式运行被中止，因此本文不声称当前完整套件实际达到 85% 覆盖率，只能确认仓库配置要求该门槛。

## 15. 以代码为准发现的文档与部署漂移

### 15.1 现有架构文档中的 VLA 信息已过时

`docs/architecture/system-architecture.md` 仍称实验 VLA 的 `model_path` 为空，并使用 `pick_object/place_object` 等旧能力名。当前代码和配置实际是：

- `model_path: models/smolvla-so101-pick-orange`
- ModelService `provides: [manipulate]`
- Skill registry 公开 `manipulate`

### 15.2 Docker Compose 的 runtime 默认配置路径不存在

`docker-compose.yml` 的 runtime 默认命令引用：

```text
/app/configs/deployment/mock.dev.yaml
```

仓库实际文件是：

```text
/app/configs/mock.dev.yaml
```

按默认值启动 runtime 容器会找不到配置。

### 15.3 Docker Compose 的 VLN 默认配置与 service ID 不匹配

Compose 默认给 VLN 使用 `xlerobot.sim.ubuntu.yaml` 和 service ID `vln_nav`，但该配置当前没有 `model_services.vln_nav`。包含 `vln_nav` 的是 `xlerobot.sim.vla_vln.yaml`；另一个 VLN 配置 `xlerobot.sim.home_vln.yaml` 使用的 ID 是 `vln_home`。

### 15.4 Docker 文档也继承了同一漂移

`docs/operations/xlerobot-sim.md` 的 Compose 默认变量表仍把 VLN config 写成普通 Ubuntu 仿真配置，需要与真实 Compose 和配置文件一起修订。

### 15.5 代码中的“自治”注释比实际能力更强

`AutonomyManager` 的类注释提到 always-on operation，但当前实现本身只是目标/事件上下文容器，没有后台调度循环。实际系统仍由 turn 驱动。对外描述应避免把它写成持续自主 Agent。

## 16. 当前最重要的工程风险

### 高优先级

1. **默认容器部署不可复现**：Compose 路径和 VLN service ID 漂移会直接导致默认启动失败。
2. **网络边界偏开发态**：gRPC 使用 `insecure_channel`；Web、NATS、ModelService 端口可被映射到宿主机，默认示例没有形成统一认证方案。
3. **真机安全不能只依赖软件 gate**：当前 gate 很有价值，但 Python/NATS/LLM 链路不是安全认证系统。
4. **VLA/VLN 能力容易被过度描述**：接口完整不代表开放世界任务已稳定，必须按模型、场景和硬件给出实测指标。

### 中优先级

5. **本地文件状态限制横向扩展**：多个实例共享 episode/runtime 目录时缺少数据库级事务和锁语义。
6. **Core NATS 默认不持久**：服务离线期间消息可能丢失，重启恢复主要依赖本地 checkpoint，而不是消息重放。
7. **长 turn 的延迟和故障面较大**：`wait_result` 让闭环自然，但一个 turn 可能同时经历多次 LLM、模型服务和机器人动作。
8. **MuJoCo 数值稳定性需要专项验收**：测试警告值得通过固定 seed、轨迹和物理量阈值单独治理。

### 架构债务

9. `AgentRuntime` 和 `TaskRunManager` 文件较大，承担了较多策略；继续扩展时需要防止状态机、证据、恢复和展示逻辑再次耦合。
10. 部分用户可见字符串和实现注释在终端读取时曾表现为乱码，但仓库的 UTF-8 防乱码测试通过，说明更可能是 Windows PowerShell 输出编码问题；运维环境仍应统一 UTF-8。

## 17. 对系统成熟度的专业判断

| 维度 | 判断 | 理由 |
|---|---|---|
| 架构边界 | 较成熟 | 四层边界有代码和测试守卫 |
| Agent 工具运行时 | 较成熟 | 有协议治理、预算、证据和错误恢复 |
| Skill 调度 | 较成熟 | 合同、资源、超时、中断链路完整 |
| Mock/仿真验证 | 较成熟 | 场景和测试丰富，MuJoCo 有具体实现 |
| 真机基础控制 | 可用但需现场验收 | 有驱动、诊断、状态与安全 gate |
| VLA 工程接入 | 已实现、实验性 | 真实 policy 路径存在，效果依赖模型与标定 |
| VLN 工程接入 | 已实现、实验性 | 有闭环局部规划，但不是完整导航栈 |
| 持续自治 | 初级 | 目标上下文存在，缺少后台认知循环 |
| 分布式生产部署 | 初级 | 本地文件状态、默认无统一认证、Compose 漂移 |
| 工业安全 | 不具备 | 缺少硬实时和功能安全认证边界 |

## 18. 建议的改进顺序

1. **先修部署事实源**：修正 Compose 默认路径和 VLN config/service ID，并增加自动化测试验证所有 Compose 命令引用的配置和服务都存在。
2. **建立能力验收矩阵**：对 Mock、MuJoCo、真机分别记录每个 semantic skill 的成功率、延迟、失败模式和恢复成功率。
3. **收紧生产 profile**：新增真正的 `production` 配置，只暴露 Agent 可见语义技能，不直接开放底层 primitive。
4. **完善网络安全**：Web 身份认证、NATS 用户/证书、gRPC TLS 或 mTLS、端口最小暴露、密钥管理。
5. **强化消息可靠性**：明确哪些 topic 必须 durable，验证 JetStream 多 subject 配置、consumer identity、ack 和幂等语义。
6. **治理长任务**：把超长 `wait_result` 工作流逐步演进为可恢复的异步 task actor，同时保持用户可查询和可中断。
7. **补全自治定义**：若目标是 always-on，需要增加持久化 goal store、事件触发器、后台调度、资源配额和无人值守安全策略。
8. **专项治理仿真稳定性**：把 QACC NaN/Inf 从警告提升为可量化测试指标，定位具体模型、关节和控制参数。

## 19. 推荐阅读代码的顺序

如果要进一步学习或二次开发，建议按下面顺序阅读：

1. `src/hey_robot/protocol/messages.py`：先理解系统共同语言。
2. `src/hey_robot/app/runner.py`：看部署时有哪些服务。
3. `src/hey_robot/gateway/service.py`：理解用户、episode 和消息入口。
4. `src/hey_robot/cognition/robot_agent.py`、`loop.py`：理解一次 turn。
5. `src/hey_robot/cognition/core.py`、`runtime/runner.py`：理解 LLM 工具循环。
6. `src/hey_robot/cognition/skill_gateway.py`：理解 Agent 为什么不能直接动机器人。
7. `src/hey_robot/skill_os/base.py`、`controller.py`：理解技能合同和调度。
8. `src/hey_robot/robot_runtime/runtime.py`、`service.py`：理解动作与观测闭环。
9. `src/hey_robot/foundation/`：理解 VLA/VLN 服务边界。
10. `tests/architecture/` 和 `tests/integration/`：理解哪些边界是项目明确承诺的。

## 20. 最后总结

Hey Robot 最有价值的部分不是某个单独模型，而是它认真处理了具身 Agent 最容易被 Demo 忽略的工程问题：异步物理执行、观测新鲜度、技能合同、资源冲突、中断、失败反馈、恢复、跨渠道会话和审计。

从计算机系统角度看，它是一套事件驱动、协议优先、状态可恢复的机器人 Agent 运行时；从 AI 角度看，它把 LLM 放在高层决策位置，把 VLA/VLN 放在受合同约束的专项能力位置；从机器人角度看，它仍然需要依靠确定性安全门、驱动和现场验收来约束学习系统。

因此，对当前版本最准确的评价是：

> 架构和软件闭环已经比较完整，Mock/仿真与大量自动化测试提供了扎实基础；真机通用能力、模型效果、持续自治、生产安全和分布式部署仍处于需要专项工程化与量化验证的阶段。
