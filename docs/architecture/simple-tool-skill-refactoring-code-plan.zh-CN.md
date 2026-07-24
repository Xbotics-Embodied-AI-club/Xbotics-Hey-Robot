# Hey Robot Tool 与 Skill 后续代码重构计划

## 1. 范围与目标

本文只记录尚未完成的工作。总体设计和固定部署边界见
[Tool 与 Skill 总体重构目标](simple-tool-skill-refactoring.zh-CN.md)。

重构目标是保持一个简单、通用、配置驱动的 Embodied Agent Harness：

- Agent/Cognition 是长程任务慢系统；
- SkillRunner、ModelService 和 Robot Runtime 是有界执行快系统；
- Tool 是 Agent 可见的 Skill 投影，Agent 不直接构造机器人动作；
- classic、VLA、VLN 和 hybrid 共用 SkillCommand/SkillEvent/SkillResult；
- 每个模块只有一种生产部署方式，不维护 local/remote 两套兼容实现；
- ModelService、Embodiment、Environment 和 endpoint 由配置组合；
- Robot Runtime 始终拥有最终动作准入、安全检查和急停权；
- SQLite、JSONL 和 artifact 文件是运行事实，NATS 只投影通知事件。

后续优先级：

```text
P1 长程恢复与安全
 -> P2 边界收敛与可观测性
 -> P3 仿真、长跑和真机验证
```

## 2. P1：完成文件级重启恢复

继续使用 SQLite、JSONL 和普通 artifact 文件，不引入额外数据库。

### 2.1 RunStore contract

扩展 RunStore：

```python
class RunStore(Protocol):
    def append_event(self, event: SkillEvent) -> None: ...
    def latest_event(self, run_id: str) -> SkillEvent | None: ...
    def result(self, run_id: str) -> SkillResult | None: ...
```

修改范围：

```text
src/hey_robot/persistence/run_store.py
src/hey_robot/skills/worker.py
src/hey_robot/skills/transport/local.py
src/hey_robot/cognition/runtime/task_coordinator.py
```

### 2.2 恢复规则

1. Worker 的 `status(run_id)` 先查 active task，再查 RunStore；
2. 已持久化终态直接返回，不重复执行；
3. 只有 non-terminal event、且当前 Worker 不拥有 active task 时，追加
   `failed/execution_lost`；
4. Agent startup reconcile 将该终态写入 TaskStore 并恢复慢系统决策；
5. 不自动重新提交可能已经执行过的物理动作；
6. 同一个 `(run_id, sequence)` 重放不得重复 resume Agent；
7. JSONL 最后一行损坏时保留已完成记录并报告 artifact corruption。

### 2.3 Worker 生命周期

- `_tasks` 只保存 active task，完成后立即移除；
- 幂等 submit 依赖 RunStore/receipt，不依赖无界 `_submitted` 字典；
- event subscriber 使用有界队列，并定义慢消费者策略；
- close 后不允许重新 start/submit；
- 30 分钟长跑时，内存不随 completed run 数量线性增长。

验收需要用新的 Store 和 Worker 实例模拟 completed、failed、cancelled、accepted/running、
submit 后崩溃和 terminal event 写入 TaskStore 前崩溃。

## 3. P1：实现真实 cancel 与 emergency stop

### 3.1 普通 cancel

普通 cancel 必须：

- 标记 run cancelled 并取消正在运行的 asyncio task；
- 调用 `ModelRouter.cancel(run_id)`；
- 阻止后续 action/action chunk；
- 写入唯一 cancelled 终态；
- 正确释放 ResourceManager 锁；
- ModelService 阻塞、断连或已经返回 action 时都保持幂等。

修改范围：

```text
src/hey_robot/skills/context.py
src/hey_robot/skills/runner.py
src/hey_robot/skills/worker.py
src/hey_robot/foundation/clients/router.py
```

### 3.2 Emergency stop

为 SkillClient 增加高优先级控制方法：

```python
async def emergency_stop(robot_id: str, *, reason: str) -> None: ...
```

LocalSkillClient 直接调用同进程 Robot Runtime control plane。该路径不得经过 NATS、gRPC、普通
Skill 队列或资源锁。必须证明模型请求阻塞且 Skill 持有 base/arm 锁时，急停仍能立即生效。

## 4. P1：声明和校验嵌套 Skill 依赖

为 Skill 增加：

```python
dependencies: tuple[str, ...] = ()
```

Builtin 必须显式声明嵌套调用，例如：

```text
pick(vla)       -> manipulate
place(vla)      -> manipulate
pick(hybrid)    -> approach_object, manipulate
```

DeploymentRunner 启动任何 Robot 或 Agent 前，递归验证 dependency 存在、无环、robot family、
required actions、required models 和 implementation 配置。不得推迟到 handler 首次运行时失败。

## 5. P1：完善 VLA/VLN 有界循环

### 5.1 Fresh observation

RobotClient 增加明确的等待语义：

```python
async def observe(
    robot_id: str,
    *,
    after_frame_id: int | None = None,
    timeout_sec: float | None = None,
) -> RobotObservation: ...
```

动作后必须等待 `frame_id > after_frame_id`。超时返回
`observation_timeout/observation_stale`，不得把旧帧送回模型。

### 5.2 Artifact 与终止语义

每个 bounded step 记录 observation refs/frame、model request 摘要、policy output、实际 action、
Robot Runtime result 和 termination reason。大图和数组只保存 artifact reference。

统一区分：

```text
model_done
environment_done
no_action
max_steps
cancelled
model_failed
action_failed
observation_stale
```

`max_steps` 只表示 option 有界结束，不表示根任务完成。benchmark/environment done 发生时，需要
让正在运行的 Skill 确定性收敛到终态，而不是由 launcher 关闭进程后留下 active Agent task。

## 6. P2：删除剩余兼容面

删除迁移期接口和死代码，不保留兼容解析：

```text
src/hey_robot/skills/clients.py
src/hey_robot/cognition/runtime/harness_store.py
AutonomousAgentService._on_skill_event transitional hook
SkillSurfaceConfig.enabled
DeploymentConfig 对 skills.enabled 的解析
DeploymentRunner._uses_native_skill_modules
```

同时清理测试、health/doctor、示例和文档中的旧配置或旧 Skill 假设。删除后增加架构测试，防止
兼容名字重新进入生产代码。

## 7. P2：收敛 Agent 与事件边界

近期保持一个 deployment 只有一个 enabled autonomous agent：

1. 配置校验显式限制单 Agent；
2. LocalSkillClient 由 composition root 单一持有和关闭；
3. Agent 不拥有共享 client 生命周期；
4. 只有出现真实多 Agent 需求时，才增加基于 `agent_id/task_id` 的显式 event routing。

建立只读 Skill event projection，使 Gateway/UI 能看到 accepted、progress 和 terminal event：

- RunStore 写入成功后再投影到 event bus；
- Gateway 不参与 ack、执行和恢复；
- 慢消费者不得阻塞 SkillRunner；
- event payload 不广播大图或大数组；
- UI 重连从 TaskStore/RunStore 查询当前状态。

## 8. P3：剩余验证

### 8.1 RoboCasa 稳定性

- cancel、ModelService crash 和 environment backend crash；
- 环境提前 done 时 Skill/Agent terminal state 收敛；
- b1/b2 长程分解条件与多任务、多 seed 回归；
- 常驻 ModelService 批量评测，避免每个 trial 重载 checkpoint；
- 30 分钟运行无 Worker task、subscriber、artifact 或显存泄漏。

### 8.2 XLeRobot VLA/VLN 与真机

- simulation end-to-end；
- 真机单步 action 和 action chunk；
- fresh-frame timeout 和 ModelService disconnect；
- emergency stop；
- 同一个 `LeRobotPolicyExecutor` 通过配置切换 checkpoint、observation mapping 和 action space。
- 使用 LeRobot 0.6 的 `fastwam` checkpoint 做启动、单步推理和 action contract smoke，确认 WAM 不引入
  新 executor 或新服务类型。

### 8.3 Classic 回归

- inspect/scan；
- move/turn/stop；
- arm/gripper；
- classic pick/place；
- oracle dock；
- 动作后重新观察和 completion evidence。

## 9. 完成条件

后续重构完成需要同时满足：

1. normal cancel 和 emergency stop 有阻塞场景测试；
2. Agent/Worker 重启能确定性 reconcile，且不重放物理动作；
3. nested Skill dependency 在启动期完整校验；
4. SkillClient 只有进程内生产实现；
5. 迁移兼容模块、旧配置字段和 transitional hook 全部删除；
6. environment done、Skill terminal 和 Agent terminal 状态一致；
7. full test、固定仿真 benchmark、长跑和至少一条真机 smoke 通过；
8. 文档、配置和实际唯一部署路径保持一致。

收口命令：

```bash
uv run --no-sync ruff check src tests evaluation
uv run --no-sync python -m pytest -q --no-cov
git diff --check
```
