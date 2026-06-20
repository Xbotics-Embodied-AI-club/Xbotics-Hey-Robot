# Hey Robot 架构总览

## 1. 项目定位

Hey Robot 是一个面向真实机器人部署的 embodied agent runtime。

它不是单纯的 LLM loop，而是围绕真实机器人任务构建的一套 runtime shell，负责：

- task state
- scene memory
- active perception
- skill contract
- execution feedback
- recovery
- health/reporting
- robot execution boundary

当前主线目标是 `XLeRobot`。系统已经形成三层 COSA-like runtime：

- 上层：Agent Runtime，负责任务理解、task state、scene memory、active perception、recovery 和多轮交互。
- 中层：Skill OS / Capability Runtime，负责 skill contract、resource gate、lifecycle、backend resolution 和 capability routing。
- 下层：Robot / Backend Control Runtime，负责 robot driver、safety、真实硬件或仿真执行；VLA 等 foundation 能力通过独立 capability service 接入。

当前系统优先保证 XLeRobot 的真实可用能力边界：Agent 通过统一 skill surface 调用可验证的感知、底盘、机械臂、夹爪和安全动作。`vla_manipulation` 已注册，capability service 可单独部署；但它未加入 `skills.enabled`，默认不进入 Agent 可调用面。

一个 deployment 通常会绑定：

- 一个 default agent
- 一个 default robot

同时通过不同 channel 接入同一套 Agent Runtime，例如：

- Web
- CLI
- Voice
- Feishu

`Teleop` 可以作为未来 channel 扩展，但不是当前主线架构入口。

## 2. 当前架构

```text
User Channel
  -> GatewayService
  -> bus topic: user_turn
  -> RobotAgentService
      -> RobotAgentRuntimeContainer
      -> AgentTurnSessions
      -> RobotAgentLoop
          restore
          build task/context/memory/recovery
          active perception gate
          run RobotAgentCore
            -> AgentRuntime
            -> request_capability / request_perception
            -> SkillGateway
  -> bus topic: skill_intent
          save checkpoint/task state
      -> TaskRunManager
      -> SceneRuntime
      -> AgentNotificationRuntime
  -> SkillControllerService
      -> SkillContractRuntime
      -> CapabilityRuntime
  -> bus topic: robot_action
  -> RobotService / RobotRuntime
      -> XLeRobotDriver / SO101Driver / LeKiwiDriver / MockRobotDriver
  -> bus topics: RobotObservation / RobotStatus / SkillEvent / SkillResult
  -> GatewayService
  -> User Channel
```

当前 XLeRobot real/sim 配置启用 11 个非 VLA skill。启用面包括感知、底盘、跟随、安全、机械臂和夹爪；`vla_manipulation` 已注册，并有 capability service 配置路径，但默认不加入 `skills.enabled`。

## 3. 关键边界

- `RobotAgentService`
  - service shell
  - 负责 bus subscription、publish 和 lifecycle 编排

- `RobotAgentLoop`
  - user turn 的产品级 state machine

- `RobotAgentCore`
  - LLM/tool execution 与 skill-level decision

- `TaskRunManager`
  - task、checkpoint、execution feedback、recovery 的 durable state

- `SceneRuntime`
  - scene memory、scene evidence query 和 active perception gate

- `SkillContractRuntime`
  - skill schema、precondition、resource conflict、timeout、readiness gate

- `CapabilityRuntime`
  - 当前按 skill/capability 配置路由到已部署 model service
  - VLA 路径使用 `vla_manipulation`；该 skill 需显式加入 `skills.enabled` 后 Agent 才能调用

- `RobotRuntime`
  - 真实 robot driver 外层的 runtime boundary

- `RobotService / RobotRuntime / PerceptionService`
  - 负责 robot status、robot observation 和 raw camera frame 发布
  - 供 Agent、perception skill、VLA camera adapter 等 consumer 使用

## 4. Plan、Memory、Recovery

当前系统不是单纯的 agent loop，而是一个带 durable task state 的 embodied runtime。

### Plan

系统里有两层 plan：

- `Agent/Core plan`：LLM/tool decision 和 runtime skill call。
- `Task plan`：持久化 task state、subgoal、skill binding、retry、feedback 和 recovery state。

真正的 durable plan boundary 是 `TaskRunManager + RobotTaskStateStore`。LLM 可以灵活推理，但 runtime 必须记录实际尝试过什么、结果是什么、失败后应该如何恢复。

### Memory

当前 memory 由 `MemoryBroker` 统一路由，根据 task state（active / recovering / completed）选择相关 memory 层：

- `Task Memory`：当前任务、subgoal、attempt、feedback、recovery state。
- `Scene Memory`：近期 observation、frame evidence、confidence、freshness。
- `User Memory`：稳定的用户偏好、称呼、地点、日常约束（`MemoryRuntime` / LTM）。
- `Robot Memory`：校准事实、降级资源、重复失败模式。

`MemoryBroker` 不把全部 memory 一股脑塞给 Agent。根据 turn intent 和 task status 做选择性路由：active task 给完整 context，recovering task 只给 recovery state + last failure，completed task 只给 generic LTM。

### Recovery

Recovery 是 runtime 做出的 typed decision，Agent 负责解释、执行或向用户确认。当前 recovery types：

- `reobserve`：重新收集视觉证据再继续。
- `reposition`：调整视角再 inspect。
- `retry_with_adjustment`：带参数调整重试。
- `ask_operator`：请求用户补充信息或授权。
- `safe_abort`：停止任务，需要人工介入。
- `degraded_continue`：非关键资源降级时继续。

未解决 recovery 前，`block_actuation=True` 贯穿整个 Agent turn pipeline，阻止 Agent 提交新的 actuation skill。Recovery state 进入 `TaskSessionView`，对 UI 可见。

## 5. COSA 对齐关系

COSA 的核心方向可以抽象为三层：

```text
upper: cognition, memory, interaction, decision
middle: composable skills, scheduling, bridge to motion
lower: robust whole-body control / real-time motion generation
```

Hey Robot 当前已经形成对应的三层 runtime：

| COSA Layer | Hey Robot Component | 当前状态 |
| --- | --- | --- |
| upper cognition | `RobotAgentCore`, LLM runtime, prompt templates | 已有 |
| memory / world context | `SceneRuntime`, task memory, episode history | 已有，仍需增强 |
| task / recovery runtime | `TaskRunManager`, `AgentNotificationRuntime` | 已有 |
| skill bridge | `SkillIntent`, `SkillResult`, `SkillContractRuntime`, `SkillControllerService`, `CapabilityRuntime` | 当前 XLeRobot real/sim 配置启用 11 个非 VLA skill；每个 skill 单一执行边界 |
| robot execution | XLeRobot drivers, readiness gates, safety policy | 已有，仍需真实场景 hardened |
| foundation backend | VLA / future foundation services | VLA 已有 `vla_manipulation` 注册和 capability service 路径；只有加入 `skills.enabled` 后才进入 Agent 可调用面 |
| classic backend | deterministic primitive control | bring-up / fallback / ablation，不是最终 claim |

skill/backend decoupling 的当前状态：

- Agent 只能从 deployment 的 `skills.enabled` 调用 skill。
- 当前 XLeRobot real/sim 配置启用 11 个非 VLA skill。
- `vla_manipulation` 已注册；capability service 可部署，但只有加入 `skills.enabled` 后才进入 Agent 可调用面。
- memory、scene record、task context 属于 Agent/runtime 工具边界，不作为 robot skill 暴露。

## 6. XLeRobot 执行链路

XLeRobot 使用统一的 skill runtime。Agent 不直接接触串口、舵机 ID、相机后端或 policy client，只能通过当前 deployment 启用的 skill 请求机器人能力。

Agent 不直接感知：

- serial port
- servo id
- camera backend
- policy client

执行链路：

```text
Agent
  -> SkillIntent(name="<enabled_skill>")
  -> SkillControllerService
  -> SkillContractRuntime
  -> RobotRuntime / CapabilityRuntime
  -> XLeRobotDriver / CapabilityService
  -> SO101 arm / LeKiwi base / camera / battery
```

当前 XLeRobot real/sim 配置启用的 skill：

- `inspect_scene`
- `look_around`
- `detect_marker`
- `move_base`
- `turn_base`
- `human_follow`
- `stop_motion`
- `reset_posture`
- `set_arm_pose`
- `move_arm_joints`
- `set_gripper`

另外注册但当前不启用：

- `vla_manipulation`

因此当前 XLeRobot 能做短步底盘移动、转向、观察、marker 检测、跟随、急停/复位、机械臂命名姿态/关节控制和夹爪开合；不能做 VLA 自然语言抓取/放置、自动抓取闭环、语义导航、避障路径规划或 SLAM。

## 7. Task Architecture

所有用户交互、机器人观察、skill request、execution feedback、recovery decision 和 UI update 都附着到 `TaskSession`。

核心产品对象 `TaskSessionView`：

```python
TaskSessionView:
  episode_id, task_id, root_task, status, current_step
  active_skill_id, active_skill_name
  last_scene_summary, last_feedback_summary
  recovery_required, recovery_strategy, recovery_summary
  next_recommended_actions, timeline
```

用户可见的 task cockpit 在 `/cockpit`，通过 `TaskSessionQueryService` 聚合 task runs、robot state、scene memory、skill traces 和 recovery state 为单一视图。

当前代码中没有独立的 `request_quick_action` Agent 工具。用户、语音和 Web 入口进入系统后，仍通过 Gateway、Agent、`request_capability`、`SkillGateway` 和 SkillController 这条主链路提交机器人能力请求；底盘流式跟随等特殊控制使用单独的 service/topic，但不作为 LLM 可见工具暴露。

## 8. Camera Model

Camera 不归 VLA 独占，也不归某个 perception skill 独占。

当前代码中的相机与观察发布由 `RobotService / RobotRuntime / PerceptionService` 共同承担：

- `robot_observation`：结构化 observation，供 Agent、Gateway、SkillController 等订阅。
- `robot.camera.frame`：raw frame stream，供 human follow、VLA camera adapter 等低延迟 consumer 使用。

以下模块都可以作为 consumer 读取：

- `RobotRuntime`
- perception skill
- VLA / foundation adapter

这套设计的核心是：

- camera 是 shared perception source
- 每个 consumer 自己控制读取频率和 freshness

## 9. Capability 子系统

当前 capability 子系统已经统一收口到：

```text
src/hey_robot/capability/
  catalog/
  contract/
    v1/
  runtime/
  sensors/
  transport/
    grpc/
```

其中：

- `catalog/`
  - capability manifest
  - capability loader
  - capability policy
  - capability resolver

- `contract/`
  - generated protobuf contract

- `runtime/`
  - capability routing
  - execution request/result model

- `transport/grpc/`
  - gRPC client/server implementation

- `sensors/`
  - capability-related observation adapter

这部分已经不再保留旧的 `src/hey_robot/capabilities` 目录。

## 10. Capability RPC 原则

当前系统不是全面 RPC 化。

原则是：

- main chain 保持 event-driven
- capability boundary 改成 gRPC-only

保留 event-driven 的部分：

- `UserTurn`
- `SkillIntent`
- `SkillEvent`
- `SkillResult`
- `RobotObservation`
- `RobotStatus`

gRPC 化的部分：

- `GetHealth`
- `ExecuteCapability`
- `CancelCapability`

## 11. 推荐验证顺序

1. 验证 Python 3.12 环境与依赖安装
2. 验证平台与硬件映射
3. 验证 camera observation publishing
4. bring-up 验证：base、arm、gripper、perception（diagnostic profile）
5. 验证 foundation capability services（如 VLA manipulation）
6. 验证 semantic skill → single implementation 执行链路
7. 验证端到端 Agent task execution、execution feedback、typed recovery
8. 验证 task cockpit（`/cockpit`）展示 task state、timeline、scene evidence、recovery

## 12. 近期优先级

1. Skill contract hardening
   - 扩展 success criteria、expected observations、recovery hints。

2. Active perception loop
   - 在 manipulation/navigation 前增加 task-grounded affordance checks。
   - 强制 post-action verification。

3. Recovery playbooks
   - 增加 calibration、grasp failure、obstacle handling、controller fault 等真实场景 playbook。

4. Field-data loop
   - 记录 task attempts、observations、failures、feedback、operator corrections，用于后续改进 foundation skill reliability。

## 13. 目录概览

```text
configs/       deployment YAML
docs/architecture/        runtime and system design notes
docs/operations/          hardware and service deployment guides
frontend/views/           Web UI views（chat、tasks、admin）
frontend/shared/          Web UI shared CSS/JS
proto/                    protobuf contract sources
src/hey_robot/agents/     Agent runtime, loop, core, task runtime, turn policy
src/hey_robot/capability/ Capability catalog, contract, runtime, gRPC transport
src/hey_robot/memory/     MemoryBroker, SceneMemoryStore, MemoryRuntime (LTM)
src/hey_robot/health/     HealthReportService and deployment health payloads
src/hey_robot/perception/ Camera service and observation pipeline
src/hey_robot/robots/     Robot runtime and drivers
src/hey_robot/skills/     Skill catalog, contracts, controller（每个 skill 单一 implementation）
src/hey_robot/tasks/      TaskSession, recovery, view, report
tests/                    runtime, robot, capability, integration tests
scripts/model_downloads/  local model download scripts
scripts/audio/            audio device utilities
scripts/dev/              development maintenance and codegen scripts
scripts/ops/              platform and deployment checks
scripts/robots/           robot-specific diagnostics and generators
```
