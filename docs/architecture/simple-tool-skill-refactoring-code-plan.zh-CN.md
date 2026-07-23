# Hey Robot Tool 与 Skill 代码级重构计划

## 1. 文档目的

本文将 [Tool 与 Skill 简化重构方案](simple-tool-skill-refactoring.zh-CN.md) 落到当前仓库的
文件、类、协议、配置、数据库和测试上。

本文描述目标代码和渐进迁移顺序，不表示这些改动已经完成。实施时应按 PR 顺序推进；不做
一次性目录搬迁，不在同一个 PR 中同时改变 Agent Tool、异步语义、Skill 执行和 NATS 协议。

## 1.1 当前实现状态

截至当前分支，代码已经完成以下迁移主干：

- 新增 `hey_robot.skills` 执行内核：`models`、`context`、`registry`、`resources`、`runner`、
  `worker`、`client`、`transport/local`、`transport/nats` 和 legacy bridge；
- Agent Tool 已从 `request_skill` 聚合工具改为每个 Skill 一个 Tool，`inspect_scene` 也走同一
  Skill Tool 生成路径；
- `SkillCallProposal` 已成为 cognition 内部的机器人能力 proposal，legacy
  `ActionProposal` 只在兼容 adapter 中转换；
- `AgentTaskStore` 已支持 pending/running/terminal step、`run_id` 幂等索引和 event sequence
  去重；
- `TaskCoordinator` 已接入 `SkillClient`，Agent 服务通过 `skill_client.events()` 消费终态事件
  后恢复长程任务；
- `FileRunStore` 已作为第一版文件化 run 轨迹存储，保存 `events.jsonl` 与 terminal
  `result.json`；
- `RobotClient`、`LocalRobotClient`、`ModelRouter`、`RegistryModelRouter` 已形成 native Skill
  的机器人与模型边界；
- native builtin shell 已迁移 `inspect_scene`、基础移动、基础机械臂/夹爪、安全停止、
  `manipulate`、`pick/place` 的 classic/vla/hybrid 注册选择；
- `DeploymentRunner` 已支持 `skills.execution_mode: local`，单进程路径可直接装配
  `RobotService -> LocalRobotClient -> SkillRunner -> LocalSkillClient -> AutonomousAgentService`，
  不再需要旧 `SkillControllerService`；
- `skills.execution_mode: event_driven` 保留 NATS + legacy bridge 迁移路径，`legacy` 保持旧行为。

仍未完成或需要单独大 PR 的部分：

- `navigate_to`、`approach_object` 的完整 VLN 主循环尚未从旧 `skill_os` 迁移，当前 native
  handler 明确返回 `not_migrated`；
- `manipulate` 已有 bounded native VLA 接入，但还不是旧实现的完整多步 horizon/chunk/fresh
  observation/artifact 循环；
- dock manipulation 还未迁移到 native builtin；
- NATS transport 仍基于当前 `MessageBus` 抽象，专用 JetStream ack、receipt、redelivery 和
  reconcile 需要后续实现；
- 旧 `skill_os`、`RobotExecutionGateway`、`ShortOperationCommand`、旧 protocol
  `SkillIntent/SkillResult` 仍作为迁移兼容层存在，删除需要等 VLN/VLA/dock 和分布式 transport
  完成。

## 2. 重构约束

重构后仍必须满足：

- Hey Robot 面向长程任务，而不是一次性机器人函数调用；
- Agent/Cognition 是慢系统，Skill/VLA/Robot Runtime 是快系统；
- 单进程和分布式部署使用同一套领域协议；
- Agent 不直接构造 `RobotAction`，不依赖 Driver primitive；
- VLA、VLN 和 RoboCasa 的 gRPC 隔离继续存在；
- Robot Runtime 保留最终安全检查；
- 根任务完成由独立 verifier 根据 evidence 判断；
- 当前 classic、VLA 和 RoboCasa 行为必须有迁移基线。

## 3. 当前代码主链

### 3.1 Agent Tool

当前入口：

- `src/hey_robot/cognition/tools/robot.py`
- `src/hey_robot/cognition/runtime/agent_runner.py`
- `src/hey_robot/cognition/autonomous_agent_service.py`

模型只能调用四个固定 Tool：

```text
request_observation
request_skill
complete_task
control_task
```

`RequestSkillTool` 把：

```json
{
  "skill": "pick",
  "objective": "pick up the cup",
  "slots": {"object": "cup"}
}
```

转为 `ActionProposal`。每个 Skill 的真实 schema 不在 function Tool schema 中，而是由
`ToolRegistry.instructions` 作为 JSON 文本加入 system prompt。

### 3.2 同步等待桥接

`AutonomousAgentService._run_task_step()` 调用
`RobotExecutionGateway.execute()`。Gateway 发布 `short_operation.command`，再使用：

```python
self._waiters: dict[str, asyncio.Future[SkillResult]]
```

等待 `skill.result`。这会把异步消息链重新拼成一次内存 RPC；进程重启后 Future 和等待关系
不可恢复。

### 3.3 Skill Controller

`src/hey_robot/skill_os/controller.py` 当前同时负责：

- NATS subscribe/publish；
- `ShortOperationCommand -> SkillIntent` 转换；
- Skill contract 准入；
- 资源冲突；
- active run 状态；
- timeout、cancel、interrupt；
- plugin context 构造；
- RobotAction 发布和 RobotStatus 关联；
- ModelService 路由和取消；
- SkillEvent、SkillResult、evidence 发布。

目标不是删除这些行为，而是把 transport、runner、robot client、model client 和 event
publisher 分开，让领域执行只存在一份。

### 3.4 当前持久化

当前同时存在：

```text
conversations.sqlite3
sustained_tasks.sqlite3
skill_receipts.sqlite3
interaction_receipts.sqlite3
events/events.jsonl
skills/skill_events.jsonl
skills/skills.json
```

`AgentTaskStore.add_step()` 当前要求立即写入 `ToolOutcome`，无法表示已经提交但尚未完成的
异步 Skill run。

## 4. 目标代码目录

第一阶段在旧 `skill_os` 旁边新增 `skills`，迁移完成后再删除旧目录：

```text
src/hey_robot/
  skills/
    __init__.py
    models.py
    context.py
    registry.py
    runner.py
    resources.py
    clients.py
    worker.py
    transport/
      __init__.py
      local.py
      nats.py
      legacy.py
    builtins/
      __init__.py
      perception.py
      navigation.py
      manipulation.py
      safety.py

  cognition/
    tools/
      skill_tools.py
      task_tools.py
    runtime/
      task_coordinator.py
      harness_store.py

  persistence/
    run_store.py

  robot_runtime/
    clients.py

  foundation/
    clients/
      models.py
      manager.py
```

`skill_os` 和 `skills` 的并存只允许发生在迁移期。新模块不得反向导入
`SkillControllerService`。

### 4.1 当前符号到目标符号

| 当前符号 | 目标符号 | 处理 |
|---|---|---|
| `RequestSkillTool` | `SkillTool` | 每个 Skill 生成一个 Tool |
| `RequestObservationTool` | `SkillTool(inspect_scene)` | 删除特殊观察 dispatcher |
| `ActionProposal` | `SkillCallProposal` | 仅保留 cognition 内部 proposal |
| `ShortOperationCommand` | `SkillCommand` | 删除同步短操作包装 |
| `SkillIntent` | `SkillCommand` | transport 迁移期转换，最终不双存 |
| protocol `SkillEvent` | 新 `SkillEvent` | 增加 `run_id`、`sequence` 和 terminal result |
| protocol/base 两个 `SkillResult` | `skills.models.SkillResult` | wire 与内部统一后删除旧类型 |
| `SkillSpec` + `SkillContract` | `Skill` | 合并事实来源 |
| 多个 Catalog | `SkillRegistry` | enabled surface 留在配置 |
| `SkillRuntime` | `SkillRunner` | 唯一执行入口 |
| `SkillScheduler` | `ResourceManager` + worker task map | 分离资源与任务所有权 |
| `SkillControllerService` | `SkillWorker` + clients + Runner | 按职责拆分 |
| `RobotActionPort` | `RobotClient` | 保留 Robot Runtime 安全边界 |
| `ModelServicePort` | `ModelRouter` | 保留 gRPC ModelService |
| `_waiters[skill_id]` | pending DB step + event sequence | 支持重启恢复 |

`SkillIntent` 仍被 `RobotSkillAction.to_robot_action()` 使用的期间，由 `LegacyBusRobotClient` 在
最底层 adapter 内构造。Agent、TaskCoordinator 和新 Skill 不得构造它。

## 5. 目标核心接口

### 5.1 `skills/models.py`

```python
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from hey_robot.protocol import ArtifactRef, Envelope, ImageRef

SkillPhase = Literal[
    "accepted",
    "running",
    "progress",
    "completed",
    "failed",
    "cancelled",
]


@dataclass(frozen=True)
class SkillResult:
    success: bool
    summary: str
    status: Literal["completed", "failed", "cancelled"]
    data: dict[str, Any] = field(default_factory=dict)
    evidence_ids: tuple[str, ...] = ()
    observations: tuple[ImageRef, ...] = ()
    artifacts: tuple[ArtifactRef, ...] = ()
    failure_mode: str | None = None
    error: str | None = None


SkillHandler = Callable[
    ["SkillContext", dict[str, Any]],
    Awaitable[SkillResult],
]


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: SkillHandler
    resources: tuple[str, ...] = ()
    timeout_sec: float = 60.0
    supported_robots: tuple[str, ...] = ()
    required_actions: tuple[str, ...] = ()
    required_models: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkillCommand:
    envelope: Envelope
    run_id: str
    task_id: str
    robot_id: str
    name: str
    arguments: dict[str, Any]
    deadline_at: float | None = None


@dataclass(frozen=True)
class SkillEvent:
    envelope: Envelope
    run_id: str
    sequence: int
    name: str
    phase: SkillPhase
    timestamp: float
    progress: float | None = None
    frame_id: int | None = None
    result: SkillResult | None = None
```

约束：

- `run_id` 是端到端幂等键，替代当前混用的 operation ID 和 skill ID；
- `task_id` 标识长程根任务；
- `Envelope.trace_id` 只用于追踪，不用于幂等；
- `sequence` 在一个 run 内从 1 单调递增；
- terminal event 必须带 `SkillResult`；
- `completed` 不等价于 root task completed。

`supported_robots`、`required_actions` 和 `required_models` 是保留的最小部署要求，用于启动期
失败，而不是将错误推迟到真实动作执行。它们分别替代当前散落的 `supported_robots`、
`driver_primitives` 和 `required_model_service`。precondition、recovery hint、goal effect 等不
参与确定性准入的描述性字段不进入第一版核心对象。

迁移初期不立即删除 `protocol.messages.SkillResult`。增加显式转换函数：

```python
def from_legacy_result(value: protocol.SkillResult) -> skills.SkillResult: ...
def to_legacy_result(event: skills.SkillEvent) -> protocol.SkillResult: ...
```

转换函数只存在于 `skills/transport/legacy.py`，不能散落在 Agent 或 Skill 中。

### 5.2 `skills/context.py`

```python
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from hey_robot.protocol import RobotObservation
from hey_robot.skills.models import SkillResult


@dataclass
class SkillContext:
    run_id: str
    task_id: str
    robot_id: str
    robot: RobotClient
    models: ModelRouter
    _invoke: Callable[[str, dict[str, Any]], Awaitable[SkillResult]]
    _emit: Callable[..., Awaitable[None]]
    _cancelled: Callable[[], bool]

    async def observe(self) -> RobotObservation:
        return await self.robot.observe(self.robot_id)

    async def run(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> SkillResult:
        return await self._invoke(name, dict(arguments or {}))

    async def progress(
        self, value: float, *, summary: str | None = None
    ) -> None:
        await self._emit(progress=value, summary=summary)

    def raise_if_cancelled(self) -> None:
        if self._cancelled():
            raise SkillCancelled(self.run_id)
```

Skill 实现只能通过 Context 使用机器人、模型、嵌套 Skill 和事件。Context 不暴露 NATS、
protobuf stub、TaskStore 或 Agent 对象。

### 5.3 `robot_runtime/clients.py`

```python
class RobotClient(Protocol):
    async def capabilities(self, robot_id: str) -> RobotClientCapabilities: ...

    async def observe(self, robot_id: str) -> RobotObservation: ...

    async def execute(
        self,
        robot_id: str,
        action: str,
        arguments: dict[str, Any],
        *,
        run_id: str,
        expected_frame_id: int | None = None,
    ) -> RobotActionResult: ...

    async def stop(self, robot_id: str, *, reason: str) -> None: ...
```

实现顺序：

1. `LegacyBusRobotClient`：封装当前 `robot.action`/`robot.status`，用于迁移；
2. `LocalRobotClient`：单进程直接调用 `RobotRuntime`；
3. `GrpcRobotClient`：只有真实跨进程需求确定后再实现。

RobotClient 是 Skill 到 Robot Runtime 的边界，不替代 Robot Runtime 的 safety evaluator。

当前 `SkillContractRuntime` 同时混合了 Skill 部署要求和 Robot action 准入。删除它之前必须
把动作侧规则移动到 Robot Runtime 自己拥有的 `RobotActionSpec`：

```python
@dataclass(frozen=True)
class RobotActionSpec:
    name: str
    parameters: dict[str, Any]
    resources: tuple[str, ...]
    motion: bool = False
```

Robot Runtime/Driver 提供 `RobotClientCapabilities.actions`，并继续拥有参数范围、限位、速度、
碰撞、estop 和 frame freshness。Skill 的 `required_actions` 只检查动作是否存在，不复制动作
安全规则。

### 5.4 Model client

第一阶段继续使用现有 `ModelServiceRegistry` 和 `GrpcModelServiceClient`，只在 Context 前增加
薄的 `ModelRouter`：

```python
class ModelRouter(Protocol):
    async def infer(
        self,
        capability: str,
        request: dict[str, Any],
        *,
        run_id: str,
        robot_id: str,
        timeout_sec: float | None = None,
    ) -> ModelInferenceResult: ...

    async def cancel(self, run_id: str) -> None: ...
```

`GrpcModelServiceClient` 的 proto 和 server 在本轮重构中保持兼容。当前
`ServiceInvocationRequest` 对 `SkillIntent` 和 `SkillContract` 的依赖由 adapter 构造；等所有
调用迁移到 `ModelRouter` 后，再单独设计 ModelService v2，不与 Skill 重构绑定。

### 5.5 `skills/registry.py`

```python
class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        if skill.name in self._skills:
            raise ValueError(f"duplicate skill: {skill.name}")
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise UnknownSkill(name) from exc

    def list(self) -> tuple[Skill, ...]:
        return tuple(self._skills.values())

    def select(self, names: tuple[str, ...]) -> tuple[Skill, ...]:
        return tuple(self.get(name) for name in names)
```

Registry 不保存 enabled 状态，不创建 RobotAction，不拥有 contract runtime。Agent Tool
白名单来自部署配置；内部 Skill 可以注册但不出现在白名单中。

启动装配阶段额外调用纯函数：

```python
validate_skill_surface(
    registry.select(config.skills.tools),
    robot_capabilities,
    model_capabilities,
)
```

它只检查 robot family、required action 和 required model capability，不检查动态电量、busy、
观测新鲜度等运行时状态。

迁移 adapter：

```python
def adapt_legacy_skill(value: BaseSkill) -> Skill:
    return Skill(
        name=value.spec.name,
        description=value.spec.description,
        parameters=value.spec.input_schema,
        handler=value.execute,
        resources=value.spec.required_resources,
        timeout_sec=value.spec.timeout_sec,
        supported_robots=value.spec.supported_robots,
        required_actions=value.spec.driver_primitives,
        required_models=(value.spec.required_model_service,)
        if value.spec.required_model_service
        else (),
    )
```

### 5.6 `skills/runner.py`

```python
class SkillRunner:
    def __init__(
        self,
        registry: SkillRegistry,
        resources: ResourceManager,
        context_factory: SkillContextFactory,
        events: SkillEventPublisher,
        runs: RunStore,
    ) -> None: ...

    async def execute(self, command: SkillCommand) -> SkillResult:
        skill = self.registry.get(command.name)
        arguments = validate_and_apply_defaults(
            skill.parameters, command.arguments
        )
        timeout_sec = effective_timeout(skill, command.deadline_at)

        await self.events.accepted(command)
        try:
            async with self.resources.acquire(
                command.robot_id, skill.resources
            ):
                await self.events.running(command)
                return await asyncio.wait_for(
                    skill.handler(
                        self.context_factory.create(command), arguments
                    ),
                    timeout=timeout_sec,
                )
        except SkillCancelled:
            return SkillResult(False, "cancelled", "cancelled")
        except TimeoutError:
            return SkillResult(
                False,
                f"{skill.name} timed out",
                "failed",
                failure_mode="timeout",
            )
        except Exception as exc:
            logger.exception("skill failed", extra={"run_id": command.run_id})
            return SkillResult(
                False,
                f"{skill.name} failed",
                "failed",
                failure_mode="internal_error",
                error=str(exc),
            )
```

实际实现必须在 `finally` 中产生一个逻辑 terminal event，并写入 `result.json`；NATS
at-least-once 传输仍可能重复投递该事件，消费者必须按 sequence 去重。嵌套
`context.run()` 复用参数校验和异常归一化，但不产生新的顶层 `run_id`；事件中通过 step path
区分子 Skill，例如 `pick/approach_object`。

Runner 不做：

- NATS subscribe；
- Agent task 状态转换；
- 根任务完成判断；
- protobuf 编解码；
- Driver capability 实现；
- LLM recovery 决策。

### 5.7 `skills/resources.py`

资源锁按 `(robot_id, resource)` 建立，并以顶层 `run_id` 作为 owner：

```python
class ResourceManager:
    async def acquire(
        self,
        robot_id: str,
        resources: tuple[str, ...],
        *,
        owner: str,
    ) -> AsyncContextManager[None]: ...
```

具体约束：

- 所有资源名排序后获取，避免不同 Skill 以相反顺序死锁；
- 同一个 root `run_id` 调用嵌套 Skill 时可重入；
- child 只获取 parent 尚未持有的额外资源；
- child 返回时只释放自己新增的资源；
- 不同 robot 的同名资源不冲突；
- emergency stop 绕过普通资源等待。

`SkillRunner` 创建 root execution scope，`context.run()` 传递同一 scope，不能递归创建一个
不知道 parent owner 的全新 Runner 状态。

### 5.8 `skills/clients.py`

```python
class SkillClient(Protocol):
    async def submit(self, command: SkillCommand) -> str: ...
    async def cancel(self, run_id: str, reason: str) -> None: ...
    async def events(self) -> AsyncIterator[SkillEvent]: ...
    async def status(self, run_id: str) -> SkillEvent | None: ...
```

`submit()` 只保证请求已被 transport 接受，不等待 Skill 完成。terminal event 通过
`events()` 返回。

## 6. Agent Tool 代码改造

### 6.1 新增 `cognition/tools/skill_tools.py`

```python
@dataclass(frozen=True)
class SkillCallProposal:
    name: str
    arguments: dict[str, Any]


class SkillTool:
    def __init__(self, skill: Skill) -> None:
        self.name = skill.name
        self.schema = {
            "type": "function",
            "function": {
                "name": skill.name,
                "description": skill.description,
                "parameters": strict_object_schema(skill.parameters),
            },
        }

    def proposal(self, arguments: dict[str, Any]) -> SkillCallProposal:
        return SkillCallProposal(self.name, dict(arguments))
```

`ToolRegistry` 构造方式改为：

```python
skill_tools = {
    skill.name: SkillTool(skill)
    for skill in registry.select(config.skills.tools)
}
tools = ToolRegistry(
    {
        **skill_tools,
        "complete_task": CompleteTaskTool(),
        "control_task": ControlTaskTool(),
    }
)
```

结果：

- 删除 `RequestSkillTool`；
- `inspect_scene` 也由 Skill 自动生成，不再维护特殊 `RequestObservationTool`；
- 删除 `ToolRegistry.instructions` 中的 Skill catalog JSON；
- `_CONVERSATION_TOOLS` 改为实例级动态集合。

`AgentRunner` 的 provider 调用、单 Tool call 限制和纯 proposal 边界保持不变。

### 6.2 配置迁移

目标配置：

```yaml
skills:
  modules:
    - hey_robot.skills.builtins
  tools:
    - inspect_scene
    - navigate_to
    - pick
    - place
    - manipulate
  implementations:
    pick: hybrid
    place: classic
```

`config/model.py`：

```python
@dataclass(frozen=True)
class SkillSurfaceConfig:
    modules: tuple[str, ...] = ("hey_robot.skills.builtins",)
    tools: tuple[str, ...] = ()
    implementations: dict[str, str] = field(default_factory=dict)
```

兼容一个迁移周期：

```text
skills.enabled -> skills.tools
skills.mode    -> deprecated，validation warning
```

若 `tools` 和 `enabled` 同时出现，配置解析直接报错，避免不明确的优先级。

## 7. 长程异步任务改造

这是风险最高的阶段，必须在直接 Tool 化和新 Runner 有独立测试之后进行。

### 7.1 Task step 状态

把 `AgentTaskStep` 改为可以表示 pending run：

```python
StepStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


@dataclass(frozen=True)
class AgentTaskStep:
    step_id: str
    task_id: str
    sequence: int
    run_id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    status: StepStatus
    outcome: ToolOutcome | None
    started_at: float
    completed_at: float | None
    last_event_sequence: int
    evidence_ids: tuple[str, ...]
```

SQLite migration使用 `PRAGMA user_version`，不删除已有表：

```sql
ALTER TABLE task_steps ADD COLUMN run_id TEXT;
ALTER TABLE task_steps ADD COLUMN tool_call_id TEXT;
ALTER TABLE task_steps ADD COLUMN tool_name TEXT;
ALTER TABLE task_steps ADD COLUMN arguments_json TEXT;
ALTER TABLE task_steps ADD COLUMN status TEXT NOT NULL DEFAULT 'completed';
ALTER TABLE task_steps ADD COLUMN last_event_sequence INTEGER NOT NULL DEFAULT 0;

CREATE UNIQUE INDEX IF NOT EXISTS task_steps_run_id
ON task_steps(run_id)
WHERE run_id IS NOT NULL;
```

旧 `proposal_json` 和 `outcome_json` 在兼容期保留。完成回填和读取迁移后，再在新数据库 schema
中移除；SQLite 不为删除列做在线复杂迁移。

### 7.2 Store API

替换立即完成的 `add_step()`：

```python
def start_skill_step(
    self,
    task_id: str,
    *,
    run_id: str,
    tool_call_id: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> AgentTaskStep: ...

def apply_skill_event(
    self, task_id: str, event: SkillEvent
) -> AgentTaskStep: ...
```

`apply_skill_event()` 必须在一个 SQLite transaction 内：

1. 按 `run_id` 找到 step；
2. 若 `event.sequence <= last_event_sequence`，作为重复事件忽略；
3. 更新 status、last sequence 和 outcome；
4. terminal event 写 completed time 和 evidence；
5. commit 后才允许唤醒 Agent。

### 7.3 `TaskCoordinator`

从 `AutonomousAgentService` 提取：

```python
class TaskCoordinator:
    def __init__(
        self,
        store: HarnessStore,
        skill_client: SkillClient,
        agent_runner: AgentRunner,
        verifier: TaskCompletionVerifier,
    ) -> None: ...

    async def submit_skill(
        self,
        task: AgentTask,
        call: AgentToolCallRecord,
        proposal: SkillCallProposal,
        envelope: Envelope,
    ) -> str: ...

    async def on_skill_event(self, event: SkillEvent) -> None: ...
```

提交顺序：

```text
create pending step in SQLite
-> submit SkillCommand
-> return run_id
```

如果 submit 失败，将 step 标记为 failed，不删除记录。

terminal event 处理顺序：

```text
apply event transaction
-> rebuild bounded agent context
-> append assistant tool call + tool result
-> resume Agent turn
-> choose next Skill / verify / complete / block
```

提交 Tool 后当前渠道轮次返回一条简短 accepted 状态；terminal event 恢复 Agent 后，通过已有
`agent.reply`/channel delivery 路径发送后续结果。TaskStore 必须保存原始 channel、chat、user
和 interaction 信息，不能依赖最初 `_on_turn()` 的调用栈仍然存在。

Agent event loop 在 service `start()` 时启动一个长期任务消费 `skill_client.events()`。每个
terminal event 先持久化再调度 resume；服务停止时取消 consumer，但不把数据库中的 pending
step 标为完成。

同一 task 的 submit、event 和 user control 使用现有 session/task lock 串行化。锁只做进程内
调度，SQLite 状态和 sequence 才是重启后的事实来源。

### 7.4 删除 Future gateway

完成异步切换后删除：

```text
RobotExecutionGateway._waiters
RobotExecutionGateway.accept_result
short_operation.command
ShortOperationCommand
skill_result_timeout_sec 作为同步等待超时
```

Tool 调用不会持有一个等待整个物理动作的 Future。Gateway 只负责渠道消息，不负责机器人
执行关联。

## 8. Local 与 NATS transport

### 8.1 `LocalSkillClient`

`skills/transport/local.py` 用于单进程开发：

```python
class LocalSkillClient:
    def __init__(self, runner: SkillRunner) -> None:
        self._commands: asyncio.Queue[SkillCommand] = asyncio.Queue()
        self._events: asyncio.Queue[SkillEvent] = asyncio.Queue()
        self._worker: asyncio.Task | None = None

    async def submit(self, command: SkillCommand) -> str:
        await self._commands.put(command)
        return command.run_id
```

它仍然通过 queue 异步执行，不能退化成 `await runner.execute()`，否则本地模式无法覆盖真实的
异步恢复行为。

### 8.2 `NatsSkillClient` 和 `SkillWorker`

目标 subject：

```text
skill.command.<robot_id>
skill.control.<robot_id>
skill.event.<robot_id>
```

领域模块不直接使用这些字符串；subject 只存在于 `skills/transport/nats.py`。

不要直接把可靠命令建立在当前通用 `BusClient.subscribe()` 上。当前 JetStream wrapper 没有
完整表达 consumer ownership、显式 ack、redelivery 和每个 subject 的 stream 更新。为 Skill
command/event 建立专用 adapter，并保留现有 `MessageBus` 供瞬时事件使用。

生产语义：

```text
SkillCommand: JetStream, at-least-once
terminal SkillEvent: JetStream, at-least-once
progress/health: Core NATS 或 JetStream，允许按配置选择
dedupe: run_id / run_id:sequence
```

Worker 接收 command：

```text
decode and validate
-> receipt_store.receive(run_id, payload_hash)
-> duplicate: publish current/terminal state, ack
-> conflict: publish failed event, ack
-> new: persist accepted receipt, ack message
-> run SkillRunner in managed task
```

消息 ack 不等待物理 Skill 完成；accepted receipt 落盘后即可 ack。Worker 崩溃后 reconcile
active receipt，默认发布 failed/unknown 并停止机器人，不盲目重放物理动作。

### 8.3 Control

保留独立 `SkillControl` 语义：

```python
@dataclass(frozen=True)
class SkillControl:
    envelope: Envelope
    control_id: str
    action: Literal["cancel", "emergency_stop"]
    run_id: str | None
    robot_id: str
    reason: str
```

`emergency_stop` 可以绕过普通任务队列进入 RobotClient stop path，但仍记录 control event。
不要把 emergency stop 作为普通 Skill。

## 9. Skill Worker 持久化

慢系统与快系统分布式部署时不能共享一个 SQLite writer。最终落盘分为两个所有者：

```text
# Slow system host
runtime/<deployment>/harness.sqlite3
  tasks
  task_steps
  conversations
  submitted_commands
  consumed_event_sequences

# Skill worker host
runtime/<deployment>/skill_worker.sqlite3
  command_receipts
  active_runs
  terminal_results

runtime/<deployment>/runs/<run_id>/
  events.jsonl
  result.json
  observations/
  artifacts/
```

`harness.sqlite3` 是长程任务事实来源；`skill_worker.sqlite3` 只负责执行幂等和 worker 恢复。
这两个数据库不能通过网络文件系统由多个进程共同写入。

当前 `AgentTaskStore`、`ConversationStore` 和 `SkillCommandStore` 在前几个 PR 保持原文件，等
领域接口稳定后再合并。不要把存储迁移和 Agent 异步切换放在同一个 PR。

## 10. Builtin Skill 迁移

### 10.1 迁移顺序

```text
1. stop_motion / reset_posture
2. set_gripper / move_arm_joints / set_arm_pose
3. inspect_scene / detect_marker / look_around
4. navigate_to / approach_object / human_follow
5. classic pick / place
6. vla_act（当前 manipulate）
7. hybrid pick / place
8. dock manipulation
```

### 10.2 机械迁移模板

旧实现：

```python
class SetGripperSkill(BaseSkill):
    spec = spec(...)

    async def execute(self, ctx, arguments):
        await ctx.robot.set_gripper(**arguments)
        return SkillResult(success=True, summary="Gripper command completed.")
```

新实现：

```python
async def set_gripper(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    result = await ctx.robot.execute(
        ctx.robot_id,
        "set_gripper",
        arguments,
        run_id=ctx.run_id,
    )
    return SkillResult(
        success=result.success,
        summary=result.summary,
        status="completed" if result.success else "failed",
        failure_mode=result.failure_mode,
        data=result.data,
    )


SET_GRIPPER = Skill(
    name="set_gripper",
    description="Set the robot gripper opening.",
    parameters=...,
    handler=set_gripper,
    resources=("gripper",),
    timeout_sec=10.0,
    supported_robots=("xlerobot", "so101", "so101_mobile"),
    required_actions=("set_gripper",),
)
```

### 10.3 VLA 迁移

将当前 `ManipulateSkill` 主循环迁移为 `vla_act` handler，保持以下逻辑不变：

- 当前观测和多相机 payload；
- policy session ID；
- native action 与 action chunk adapter；
- action 后等待 fresh observation；
- bounded max steps；
- cancel ModelService；
- `requires_reobservation`；
- 原始模型结果进入 run artifact。

第一步保留 Agent Tool 名 `manipulate`，Registry 中将其 handler 指向 `vla_act`。完成配置和
评测迁移后，再决定是否在 Harness surface 暴露 `vla_act` 名称。不要仅为命名变化破坏现有
RoboCasa `provides: manipulate` 路由。

`FixedHorizonTerminationEvaluator` 第一阶段直接内联为循环条件。出现 grasp、placement 和
fixture 等三个共享 verifier 后再恢复独立 termination abstraction。

### 10.4 Classic 与 Hybrid

`pick`、`place` 保持稳定的 Agent Tool 名。启动注册时根据配置选择 handler：

```python
PICK_HANDLERS = {
    "classic": classic_pick,
    "vla": vla_pick,
    "hybrid": hybrid_pick,
}

implementation = config.skills.implementations["pick"]

registry.register(
    Skill(
        name="pick",
        description=PICK_DESCRIPTION,
        parameters=PICK_PARAMETERS,
        handler=PICK_HANDLERS[implementation],
        resources=("robot_control", "camera"),
        timeout_sec=180.0,
        required_actions=PICK_REQUIRED_ACTIONS[implementation],
        required_models=PICK_REQUIRED_MODELS[implementation],
    )
)
```

Hybrid 用普通 `ctx.run()`：

```python
async def hybrid_pick(ctx, arguments):
    staged = await ctx.run("approach_object", arguments)
    if not staged.success:
        return staged
    return await ctx.run(
        "manipulate",
        {"task_prompt": f"grasp {arguments['object']}"},
    )
```

不新增 DAG、OptionRegistry 或 HybridRuntime。

## 11. DeploymentRunner 改造

### 11.1 单进程

当前 `DeploymentRunner._build_services()` 分别构造 RobotService、SkillControllerService、
AutonomousAgentService 和 GatewayService，但它们只能通过 bus 获得彼此。

目标增加一个应用装配对象：

```python
@dataclass
class RuntimeComponents:
    registry: SkillRegistry
    robot_client: RobotClient
    model_router: ModelRouter
    run_store: RunStore
    skill_runner: SkillRunner
    skill_client: SkillClient
```

单进程构造：

```text
RobotManager -> RobotRuntime -> LocalRobotClient
ModelServiceRegistry -> ModelRouter
SkillRegistry + clients -> SkillRunner
SkillRunner -> LocalSkillClient
LocalSkillClient -> AutonomousAgentService
```

Gateway 的渠道输入输出可以继续使用现有 bus；不要求本轮同时重写 Web、Feishu 和 Voice。

### 11.2 分布式

分布式进程角色：

```text
hey-robot gateway
hey-robot agent
hey-robot skill-worker
hey-robot robot-runtime       # 可按部署需要合并进 worker
hey-robot model-service
```

Agent 进程构造 `NatsSkillClient`，Skill worker 构造 `NatsSkillCommandConsumer + SkillRunner`。
两者都不构造 `SkillControllerService`。

## 12. 分 PR 实施计划

### PR 0：行为基线

只增加测试和评测记录，不改生产代码。

新增：

```text
tests/characterization/test_agent_tool_surface.py
tests/characterization/test_skill_message_flow.py
tests/characterization/test_vla_manipulate_flow.py
tests/characterization/test_task_resume.py
```

固定：

- 当前 production/bringup Tool definitions；
- ShortOperation 到 SkillResult 的消息顺序；
- classic pick/place 代表路径；
- VLA payload 和 fresh frame 行为；
- emergency stop；
- RoboCasa 指定 checkpoint/seed 结果。

### PR 1：新 models、registry、runner

新增：

```text
src/hey_robot/skills/models.py
src/hey_robot/skills/context.py
src/hey_robot/skills/registry.py
src/hey_robot/skills/resources.py
src/hey_robot/skills/runner.py
src/hey_robot/skills/transport/legacy.py
```

旧执行链不切换。使用 legacy adapter 注册 2 个测试 Skill，验证嵌套执行、资源、timeout、
cancel、异常和 terminal event exactly once。

### PR 2：直接 Skill Tool

新增 `cognition/tools/skill_tools.py`，修改：

```text
cognition/tools/robot.py
cognition/runtime/agent_runner.py
cognition/autonomous_agent_service.py
config/model.py
config/validation.py
```

这一阶段 Tool 仍生成兼容 `ActionProposal`，仍走旧 Gateway 和 Controller。验收重点仅是模型
看到 `pick(...)` 等真实 schema，执行行为不变。

### PR 3：LocalSkillClient 和 SkillWorker shell

新增：

```text
skills/clients.py
skills/worker.py
skills/transport/local.py
persistence/run_store.py
```

用 Mock Skill 跑通：

```text
submit -> accepted -> running -> progress -> terminal
```

不接入 Agent，不迁移 Builtin。

### PR 4：TaskStore 支持 pending step

修改 `AgentTaskStore` 或新增 `HarnessStore` adapter，加入 SQLite schema migration、
`start_skill_step()` 和 `apply_skill_event()`。现有同步 `add_step()` 暂时保留。

### PR 5：Agent 异步切换

新增 `TaskCoordinator`，注入 `SkillClient`。将 Agent Tool call 改为：

```text
persist pending step -> submit -> release current turn
```

terminal event 再恢复 Agent。完成后删除 `RobotExecutionGateway._waiters`，但 legacy transport
仍可把新 command/event 映射到旧 Controller。

### PR 6：迁移 Builtin 和拆 Controller

按第 10 节顺序迁移。每迁移一组，就从 Controller 删除对应 plugin context 或执行分支。避免
先复制全部 Controller 再一次删除。

同时将 `SkillContractRuntime.validate_action()` 中真正属于机器人动作安全的规则迁移到
Robot Runtime 的 `RobotActionSpec` 和 safety tests。只有 Robot Runtime 不再导入
`SkillContractCatalog` 后，才允许删除 `contracts/skill_contracts.py`。

需要逐项修改：

```text
robot_runtime/base.py
robot_runtime/manager.py
robot_runtime/service.py
robot_runtime/mock.py
robot_runtime/xlerobot/driver.py
robot_runtime/simulation/xlerobot_sim_driver.py
foundation/catalog/loader.py
```

完成时 Controller 只剩 legacy NATS adapter，可以重命名为 `LegacySkillWorker`。

### PR 7：NATS Skill transport

实现专用 JetStream adapter、worker receipt、event sequence 和 reconcile。先用 Mock Robot 做
进程级集成测试，再切真机配置。

### PR 8：存储收敛

慢系统合并为 `harness.sqlite3`，worker 使用独立 `skill_worker.sqlite3`，轨迹统一进入 run
目录。删除重复 snapshot，但不删除历史数据文件。

### PR 9：删除旧 Skill OS

删除前要求 `rg` 不再发现生产代码引用：

```text
RequestSkillTool
ShortOperationCommand
SkillControllerService
SkillContractRuntime
SkillExecutionPlan
RobotActionPort
PerceptionPort
ModelServicePort
```

随后删除：

```text
skill_os/controller.py
skill_os/scheduler.py
skill_os/composition.py
skill_os/apis.py
skill_os/ports.py
skill_os/runtime/
contracts/skill_contracts.py
```

保留仍有独立价值的算法文件时，将其移动到 `skills/builtins`，不要通过兼容 import 永久保留
旧 package。

## 13. 测试计划

### 13.1 新单元测试

```text
tests/skills/test_models.py
tests/skills/test_registry.py
tests/skills/test_runner.py
tests/skills/test_resources.py
tests/skills/test_local_client.py
tests/skills/test_worker.py
tests/cognition/tools/test_skill_tools.py
tests/cognition/runtime/test_task_coordinator.py
tests/persistence/test_run_store.py
```

必须覆盖：

- Tool schema 与 Skill parameters 完全一致；
- unknown/disabled Tool 在调用前被拒绝；
- schema default、required、additionalProperties、enum、range；
- 同 robot 冲突资源互斥，不同 robot 不互斥；
- timeout/cancel/internal error 各自产生一个逻辑 terminal event；
- nested Skill 不生成第二个顶层 run；
- event sequence 去重；
- duplicate command 不重复执行 Robot action；
- restart 后 pending task 可以 reconcile；
- emergency stop 不受普通资源锁阻塞。

JSON Schema 校验使用完整 validator，不继续扩展当前手写 `_validate_slots()`。如果选择
`jsonschema`，应把它加入直接依赖，不能依赖当前 lock file 中的传递依赖。

### 13.2 集成测试

```text
tests/integration/test_local_skill_flow.py
tests/integration/test_nats_skill_flow.py
tests/integration/test_agent_async_resume.py
tests/integration/test_model_service_grpc_flow.py
tests/integration/test_robocasa365_contract.py
```

`test_nats_skill_flow.py` 必须真实启动 NATS/JetStream，不使用 InMemoryBus 冒充 durable
transport。

### 13.3 架构测试

更新 `tests/architecture`：

- cognition 不导入 `RobotAction` 或 Driver；
- Skill 不导入 cognition、NATS client 或 protobuf；
- Robot Runtime 不导入 cognition 或 skills；
- Foundation backend 不导入 cognition；
- NATS subject 只允许出现在 transport 和 operations 代码；
- Agent Tool definitions 必须由 configured Skill 生成；
- 生产代码不得构造 `RequestSkillTool` 或 `ShortOperationCommand`。

## 14. 每个阶段的验收命令

每个 PR 至少执行：

```bash
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
uv run --no-sync python -m pytest tests/skills tests/cognition tests/architecture
uv run --no-sync python -m pytest tests/robot_runtime tests/integration/test_model_service_grpc_flow.py
```

涉及 NATS 时额外执行：

```bash
uv run --no-sync python -m pytest tests/integration/test_nats_skill_flow.py
```

涉及 RoboCasa/VLA 时执行固定配置的 smoke/evaluation 命令，并记录 checkpoint、seed、成功
条件、step 数和失败原因；不能只以 pytest mock 通过作为迁移完成。

## 15. 回滚和兼容规则

- 每个迁移 PR 保持一种正式路径和最多一种 legacy adapter；
- legacy adapter 只能从新接口指向旧实现，旧实现不能导入新业务层；
- 新旧消息双写只允许用于观测对比，不能让两个 worker 同时执行；
- 配置字段兼容最多保留一个迁移周期；
- 数据库 migration 必须向前兼容已有 runtime 文件；
- 删除旧文件前先删除所有生产引用，再删除兼容测试，最后删除文件；
- 如果 VLA 或 classic 基线回退，回滚当前迁移组，不回滚已经稳定的 Tool schema 改造。

## 16. 完成定义

重构完成需要同时满足：

1. Agent 直接看到配置的独立 Skill Tool；
2. Skill 定义是 Tool schema 和执行 handler 的唯一事实来源；
3. 所有顶层和嵌套 Skill 经过一个 `SkillRunner`；
4. 慢系统通过持久化 pending step 和 terminal event 恢复，不依赖内存 Future；
5. Local 与 NATS 部署使用相同 `SkillCommand/SkillEvent`；
6. NATS 细节只存在于 transport adapter；
7. VLA/VLN/RoboCasa gRPC 行为不变；
8. Robot Runtime 保持最终安全边界；
9. classic、VLA、hybrid 是 handler 差异，不是 runtime 差异；
10. 旧 Controller、ContractRuntime、多 Catalog、API/Port 重复层被删除；
11. 长程任务、worker receipt 和 run artifacts 可以在进程重启后 reconcile；
12. 固定 RoboCasa 和真实机器人 smoke 基线没有行为回退。
