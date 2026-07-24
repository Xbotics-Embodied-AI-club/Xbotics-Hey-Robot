# Skill 扩展指南

本文说明重构后的唯一 Skill 扩展方式。目标是让二次开发者在组合已有机器人能力时，只关注 Skill 层和部署配置，不需要修改 Agent、Controller、消息总线或具体硬件驱动。

在异步快慢双系统中，Skill 属于下层快系统：它接收慢系统给出的目标，在受约束的短时域
内调用 Foundation Model 或 Robot Runtime，并把进度和结果异步反馈给上层。

## 1. 先给结论

如果新能力只组合系统已经具备的能力，开发者只需：

1. 实现一个异步 handler，并创建一个 `Skill`；
2. 在 `Skill` 中声明输入、资源、依赖和运行时要求；
3. 通过 `register(registry)` 注册；
4. 在部署配置的 `skills.modules` 和 `skills.tools` 中启用；
5. 添加 Skill 单元测试和部署校验测试。

不需要修改：

- Agent prompt、Agent 主循环或 Agent tool；
- `LocalSkillClient`、`SkillWorker` 或 `SkillRunner`；
- Bus topic 和协议消息；
- Robot Driver。

但存在明确边界：

- 新增硬件原语时，需要扩展 Driver 和对应执行适配器；
- 新增传感器能力时，需要扩展感知或 Driver 适配器；
- 新增外部模型或服务时，需要实现 Foundation Model executor/ModelService，并在配置中启用；
- 只有底层能力已经存在时，Skill 才能只靠组合获得新语义能力。

## 2. 唯一运行链路

```text
deployment skills.modules
  -> register(SkillRegistry)
  -> deployment skills.tools
  -> Agent 读取可见 Skill 契约
  -> LocalSkillClient.submit(SkillCommand)
  -> SkillWorker 管理异步运行、取消和持久化
  -> SkillRunner 检查参数、依赖、资源和超时
  -> Skill handler
  -> SkillContext ports
  -> Robot Runtime / ModelService
```

系统没有兼容 Registry 或第二执行器。`Skill` 是契约的唯一事实源，
`SkillRunner.run()` 是顶层和嵌套 Skill 的唯一执行入口。

## 3. 最小 Skill

```python
from typing import Any

from hey_robot.skills import Skill, SkillContext, SkillResult


async def inspect_target(
    ctx: SkillContext, arguments: dict[str, Any]
) -> SkillResult:
    observation = await ctx.observe(timeout_sec=2.0)
    return SkillResult(
        success=True,
        summary=f"inspected {arguments['target']}",
        status="completed",
        data={"frame_id": observation.frame_id},
        observations=tuple(observation.images),
        artifacts=tuple(observation.artifacts),
    )


INSPECT_TARGET = Skill(
        name="inspect_target",
        description="Inspect whether a named target is visible.",
        parameters={
            "type": "object",
            "properties": {"target": {"type": "string"}},
            "required": ["target"],
            "additionalProperties": False,
        },
        handler=inspect_target,
        resources=("camera",),
        supported_robots=("xlerobot",),
        timeout_sec=6.0,
        required_actions=("inspect_scene",),
    )
```

Skill handler 只描述一次有界执行。实际执行轨迹由 Worker 根据真实事件记录，避免计划和执行形成两个事实源。

## 4. 组合已有 Skill

组合 Skill 使用 `ctx.run()`，并在父 Skill 中声明依赖：

```python
async def inspect_then_stop(ctx, arguments):
    inspection = await ctx.run("inspect_scene", dict(arguments))
    if not inspection.success:
        return inspection
    stopped = await ctx.run("stop_motion", {})
    if not stopped.success:
        return stopped
    return SkillResult(True, "Inspection completed and motion stopped.", "completed")


INSPECT_THEN_STOP = Skill(
        name="inspect_then_stop",
        description="Inspect the scene and then stop robot motion.",
        parameters={"type": "object"},
        handler=inspect_then_stop,
        dependencies=("inspect_scene", "stop_motion"),
        resources=("camera", "base"),
        supported_robots=("xlerobot",),
    )
```

必须同时在 `dependencies` 中声明所有子 Skill。部署校验会递归检查依赖是否存在，以及依赖的外部 ModelService 是否可用。

## 5. SkillContext 边界

Skill 只能通过以下端口访问系统能力：

```text
ctx.robot          已有机器人动作
ctx.models         已配置的 ModelService 路由
ctx.run            已声明依赖的其他 Skill
ctx.progress       当前 run 的进度事件
ctx.raise_if_cancelled  协作式取消检查
```

禁止在 Skill 中：

- 直接发布 Bus 消息；
- 构造 `RobotAction` 或协议 payload；
- 导入 Agent 运行时；
- 访问串口、舵机、MuJoCo actuator 等硬件细节；
- 自己实现调度、资源锁、超时或生命周期事件。

## 6. Skill 契约字段

`Skill` 是 Skill 契约的唯一来源，重点字段如下：

- `name`：全局唯一，重复注册会启动失败；
- `description`：准确描述真实能力，不允许语义夸大；
- `parameters`：JSON Schema 参数结构和必填参数；
- `resources`：如 `camera`、`base`、`arm`、`gripper`；
- `dependencies`：执行时调用的子 Skill；
- `required_actions`：该 Skill 直接需要 Robot Runtime 支持的动作；
- `required_models`：该 Skill 直接依赖的 ModelService 能力；
- `supported_robots`：支持的机器人族；
- `timeout_sec`：运行上限；

只有列入 `skills.tools` 的 Skill 才投影为 Agent tool。注册但未列入的底层 Skill 只能通过已声明的 `ctx.run()` 依赖调用。

## 7. 注册与配置

模块必须暴露统一注册函数：

```python
from typing import Any


def register(registry: Any) -> None:
    registry.register(INSPECT_TARGET)
```

部署配置：

```yaml
skills:
  mode: production
  modules:
    - hey_robot.skills.builtins
    - my_robot_skills
  tools:
    - inspect_scene
    - inspect_target
```

含义：

- `modules` 决定加载哪些注册模块；
- `tools` 是当前部署对 Agent 开放的显式能力面；
- 未注册、重名、不支持当前机器人或缺少外部 ModelService，部署校验会失败；
- `required_actions` 声明的动作不被当前 robot runtime 支持时，部署校验会失败；
- 未列入 `tools` 的内部依赖仍可由已启用 Skill 调用，但不会直接暴露给 Agent。

## 8. 三类扩展

### 8.1 纯语义组合

例：先观察，再转向，再复查。

只改 Skill 和配置，不改 Agent 与硬件层。

### 8.2 新外部能力

例：VLA、导航服务、IK 服务。

需要：

1. 实现 ModelService；
2. 在配置中声明该服务提供的能力名；
3. 创建模型驱动 Skill，设置 `required_models`；
4. 由语义 Skill 通过 `ctx.run()` 调用。

`required_models`、Skill 实现传给 `ctx.models.call(name, ...)` 的
`name`，以及 deployment 中 `model_services.<id>.provides` 必须一致。仅通过静态部署
校验还不足以发现实现调用名不一致，必须添加一次真实 `ModelServiceRegistry` 路由测试。

### 8.3 新硬件原语

例：新夹爪命令、新关节模式、新传感器。

需要：

1. 在 Driver 或感知层实现真实能力；
2. 在对应 port/adapter 暴露稳定接口；
3. 创建内部 Skill，并在 `required_actions` 中声明它需要的运行时动作；
4. 再创建面向 Agent 的语义 Skill。

这不是架构泄漏，而是能力所有权边界：Skill 定义“做什么”，Driver 定义“硬件怎样做”。

## 9. 测试要求

至少覆盖：

- 输入缺失时被 `SkillRunner` 拒绝；
- `execute()` 成功和失败结果；
- 嵌套 `ctx.run()` 的调用参数、依赖限制和失败传播；
- Registry 能加载模块且拒绝重名；
- 部署配置能启用 Skill；
- 机器人族不匹配或 ModelService 缺失时启动失败；
- 涉及资源的 Skill 具备冲突测试；
- 新硬件原语具备 Driver 或仿真集成测试。

## 10. 完成标准

一个普通语义 Skill 的提交不应修改 Agent、Worker、Runner、Robot Runtime、协议和 Driver。若必须修改这些模块，应先判断新增的是系统级机制、Foundation Model 服务，还是全新的硬件原语，而不是把它伪装成普通 Skill 扩展。
