# Skill 扩展指南

本文说明当前 native local Skill架构的唯一扩展方式。普通Skill开发不需要修改Agent、
消息总线、Worker或具体硬件驱动；新增模型能力或硬件原语时，仍需扩展其所有权层。

## 1. 运行链

```text
skills.modules -> register(SkillRegistry)
skills.tools   -> ToolRegistry exposes selected Skills
model proposal -> PhysicalToolCall
TaskCoordinator -> SkillClient.submit(SkillCommand)
SkillWorker -> SkillRunner -> handler(SkillContext, arguments)
SkillContext.robot  -> RobotClient -> RobotRuntime
SkillContext.models -> ModelRouter -> gRPC ModelService
```

当前没有`LocalSkillClient`类，`SkillWorker`本身实现`SkillClient`接口。也没有嵌套
`ctx.run()`或Skill dependency graph；组合能力应写成一个有界handler，直接通过
RobotClient/ModelRouter调用其所需的稳定下层端口。

## 2. 最小 Skill

```python
from typing import Any

from hey_robot.skills import Skill, SkillContext, SkillResult


async def inspect_target(
    ctx: SkillContext, arguments: dict[str, Any]
) -> SkillResult:
    observation = await ctx.observe(timeout_sec=2.0)
    target = str(arguments["target"])
    return SkillResult(
        success=True,
        summary=f"inspected {target}",
        status="completed",
        data={"frame_id": observation.frame_id, "target": target},
        observations=tuple(observation.images),
        artifacts=tuple(observation.artifacts),
    )


INSPECT_TARGET = Skill(
    name="inspect_target",
    description="Inspect whether a named target is visible.",
    parameters={
        "type": "object",
        "properties": {"target": {"type": "string", "minLength": 1}},
        "required": ["target"],
        "additionalProperties": False,
    },
    handler=inspect_target,
    resources=("camera",),
    supported_robots=("xlerobot",),
    timeout_sec=6.0,
)
```

Handler必须返回`SkillResult`，不应抛出异常表达业务失败。未捕获异常会由SkillRunner归一化
为`internal_error`。

## 3. 当前 Skill contract

`hey_robot.skills.models.Skill`只有以下契约字段：

- `name`：registry内全局唯一；
- `description`：直接投影给Agent模型；
- `parameters`：JSON Schema；
- `handler`：异步执行函数；
- `resources`：按`(robot_id, resource)`互斥；
- `timeout_sec`：单次handler执行上限；
- `supported_robots`：允许的robot family；
- `required_actions`：Robot Runtime必须支持的动作；
- `required_models`：deployment必须提供的ModelService capability。

当前没有以下字段或机制：

- `agent_visible`；
- `dependencies`与`ctx.run()`；
- safety level或interruptibility contract；
- success criteria、failure modes、recovery hints或goal effects。

不要在扩展模块中假设这些历史设计字段存在。

## 4. SkillContext 边界

Handler可使用：

- `ctx.robot.execute(robot_id, action, arguments, run_id=...)`；
- `ctx.robot.observe(...)`，通常优先使用`ctx.observe()`；
- `ctx.models.infer(capability, request, run_id=..., robot_id=...)`；
- `ctx.progress(value, summary)`；
- `ctx.raise_if_cancelled()`。

使用前应处理`ctx.robot`或`ctx.models`为`None`的情况，并返回结构化失败。

禁止在普通Skill中：

- 发布bus消息或构造protocol payload；
- 导入Cognition或修改Agent任务状态；
- 直接访问串口、舵机SDK、MuJoCo actuator或remote environment；
- 自己创建全局资源锁、run store或生命周期事件；
- 绕过RobotClient直接调用driver。

## 5. 调用机器人动作

简单Skill可复用内置适配器：

```python
from hey_robot.skills.builtins.common import execute_robot_action


async def point_camera(ctx, arguments):
    return await execute_robot_action(ctx, "set_arm_pose", arguments)
```

随后在`Skill.required_actions`中声明`set_arm_pose`。部署校验会检查所选robot driver是否
支持该动作。

如果动作不存在，先在Robot Runtime/driver层实现和验证动作，再注册Skill。新增硬件能力
不应隐藏在Skill handler的私有串口调用中。

## 6. 调用 ModelService

模型驱动Skill应：

1. 在`required_models`声明capability；
2. 调用`ctx.models.infer()`时使用相同capability名称；
3. 确保deployment的`model_services.<id>.provides`包含该名称；
4. 为路由、timeout、取消、无服务和模型失败添加测试。

示意：

```python
if ctx.models is None:
    return SkillResult(
        False,
        "model router unavailable",
        "failed",
        failure_mode="model_service_unavailable",
    )

inference = await ctx.models.infer(
    "my_capability",
    {"observation": observation_payload, "prompt": prompt},
    run_id=ctx.run_id,
    robot_id=ctx.robot_id,
    timeout_sec=30.0,
)
```

对于多步VLA/VLN，应将observe-infer-act循环封装为有界option runner，并明确max steps、
fresh observation和termination语义。

## 7. 注册与配置

扩展模块暴露`register(registry)`：

```python
from hey_robot.skills import SkillRegistry


def register(registry: SkillRegistry) -> None:
    registry.register(INSPECT_TARGET)
```

部署配置：

```yaml
skills:
  mode: production
  execution_mode: local
  modules:
    - hey_robot.skills.builtins
    - my_robot_skills
  tools:
    - inspect_scene
    - inspect_target
```

含义：

- `modules`加载registry模块；当前部署校验默认只允许`hey_robot.skills.*`命名空间，若要
  支持外部包，需要先有意识地扩展该安全规则；
- `tools`是Agent可见Skill的唯一显式allowlist；
- `implementations`可为支持该参数的register函数选择具体实现；
- `execution_mode`当前只能是`local`；
- `mode`当前只校验`production/bringup`取值，不会自动过滤Skill。

因此将primitive加入`tools`会直接暴露给模型，无论mode名称是什么。

## 8. 组合能力

当前没有嵌套Skill API。组合既有动作时，在一个handler内按顺序调用RobotClient，并在
每一步检查`RobotActionResult`；长循环中调用`raise_if_cancelled()`并通过`progress()`
报告进度。

组合handler应满足：

- 执行有界；
- 每一步失败立即返回结构化`SkillResult`；
- 不声称未验证的成功；
- `resources`覆盖整个组合期间使用的资源；
- `required_actions`列出所有直接机器人动作；
- 需要fresh observation时使用`after_frame_id`和timeout。

如果组合逻辑需要复用，抽成普通Python helper或option runner，而不是构造第二套Skill
scheduler。

## 9. 测试要求

至少覆盖：

- JSON Schema缺失、额外字段和边界值；
- handler成功、业务失败、异常和timeout；
- cancellation与资源释放；
- resource conflict串行化；
- Registry加载和重名拒绝；
- `skills.tools`投影出的Agent schema；
- robot family、required action和required model部署校验；
- RobotClient/ModelRouter调用参数及失败传播；
- 多步能力的fresh observation、budget exhaustion和termination reason。

## 10. 完成标准

普通语义Skill提交不应修改Agent、Worker、Runner、协议或driver。若必须修改这些模块，先
判断新增的是系统级调度机制、远程执行adapter、Foundation Model capability还是新的
硬件原语，并在相应所有权层实现。
