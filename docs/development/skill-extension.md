# Skill 扩展指南

本文说明重构后的唯一 Skill 扩展方式。目标是让二次开发者在组合已有机器人能力时，只关注 Skill 层和部署配置，不需要修改 Agent、Controller、消息总线或具体硬件驱动。

## 1. 先给结论

如果新能力只组合系统已经具备的能力，开发者只需：

1. 实现一个 `BaseSkill` 子类；
2. 在 `SkillSpec` 中声明输入、资源、依赖和安全约束；
3. 通过 `register_skills(registry)` 注册；
4. 在部署配置的 `skills.modules` 和 `skills.enabled` 中启用；
5. 添加 Skill 单元测试和部署校验测试。

不需要修改：

- Agent prompt、Agent 主循环或 Agent tool；
- `SkillControllerService`；
- `SkillRuntime`、`SkillScheduler`；
- Bus topic 和协议消息；
- Robot Driver。

但存在明确边界：

- 新增硬件原语时，需要扩展 Driver 和对应执行适配器；
- 新增传感器能力时，需要扩展感知或 Driver 适配器；
- 新增外部模型或服务时，需要实现 capability service，并在配置中启用；
- 只有底层能力已经存在时，Skill 才能只靠组合获得新语义能力。

## 2. 唯一运行链路

```text
deployment skills.modules
  -> register_skills(SkillRegistry)
  -> deployment skills.enabled
  -> Agent 读取可见 Skill 契约
  -> SkillControllerService 接收 SkillIntent
  -> SkillScheduler 检查资源冲突、超时和中断
  -> SkillRuntime.validate / execute
  -> BaseSkill.execute
  -> SkillContext ports
  -> Driver / Perception / Capability Service
```

系统没有静态默认 Skill catalog、兼容 Registry 或第二执行器。`BaseSkill.spec` 是契约的唯一事实源，`SkillRuntime.execute()` 是顶层和嵌套 Skill 的唯一执行入口。

## 3. 最小 Skill

```python
from hey_robot.skills.base import BaseSkill, SkillResult, SkillSpec


class InspectTargetSkill(BaseSkill):
    spec = SkillSpec(
        name="inspect_target",
        description="Inspect whether a named target is visible.",
        category="perception",
        input_schema={
            "type": "object",
            "properties": {"target": {"type": "string"}},
            "required": ["target"],
        },
        required_resources=("camera",),
        supported_robots=("xlerobot",),
        safety_level="observe",
        timeout_sec=6.0,
        agent_visible=True,
        feedback_mode="vision",
    )

    async def execute(self, ctx, arguments):
        result = await ctx.perception.inspect_scene(
            question=f"find {arguments['target']}"
        )
        return SkillResult(
            success=bool(result.get("success", True)),
            summary=str(result.get("summary") or "inspection completed"),
            failure_mode=result.get("failure_mode"),
            error=result.get("error"),
            data=dict(result),
        )
```

Skill 只实现 `execute()`。系统不再提供 `plan()`；实际执行轨迹由运行时根据真实发生的动作记录，避免计划和执行形成两个事实源。

## 4. 组合已有 Skill

组合 Skill 使用 `ctx.invoke()`：

```python
class InspectThenStopSkill(BaseSkill):
    spec = SkillSpec(
        name="inspect_then_stop",
        description="Inspect the scene and then stop robot motion.",
        category="safety",
        dependencies=("inspect_scene", "stop_motion"),
        required_resources=("camera", "base"),
        supported_robots=("xlerobot",),
        safety_level="motion",
    )

    async def execute(self, ctx, arguments):
        inspection = await ctx.invoke("inspect_scene", dict(arguments))
        if not inspection.success:
            return inspection
        stopped = await ctx.invoke("stop_motion", {})
        if not stopped.success:
            return stopped
        return SkillResult(success=True, summary="Inspection completed and motion stopped.")
```

必须同时在 `dependencies` 中声明所有子 Skill。部署校验会递归检查依赖是否存在，以及依赖的外部 capability 是否可用。

## 5. SkillContext 边界

Skill 只能通过以下端口访问系统能力：

```text
ctx.robot          已有机器人动作
ctx.perception     已有感知能力
ctx.capabilities   已配置的外部能力服务
ctx.invoke         其他已注册 Skill
```

禁止在 Skill 中：

- 直接发布 Bus 消息；
- 构造 `RobotAction` 或协议 payload；
- 导入 Agent 运行时；
- 访问串口、舵机、MuJoCo actuator 等硬件细节；
- 自己实现调度、资源锁、超时或生命周期事件。

## 6. SkillSpec 必填思维

`SkillSpec` 是 Skill 契约的唯一来源，重点字段如下：

- `name`：全局唯一，重复注册会启动失败；
- `description`：准确描述真实能力，不允许语义夸大；
- `input_schema`：参数结构和必填参数；
- `required_resources`：如 `camera`、`base`、`arm`、`gripper`；
- `dependencies`：执行时调用的子 Skill；
- `driver_primitives`：该 Skill 直接需要当前 robot driver 支持的运行时原语；
- `external_capability`：该 Skill 直接依赖的外部服务能力；
- `supported_robots`：支持的机器人族；
- `safety_level`：`observe`、`normal`、`motion`、`stop` 等；
- `timeout_sec`：运行上限；
- `agent_visible`：是否作为语义能力暴露给 Agent；
- `failure_modes` 和 `recovery_hints`：预期失败及恢复建议。

生产配置只能启用 `agent_visible=True` 的语义 Skill。底层原语仍可注册，但只能由语义 Skill 通过 `ctx.invoke()` 使用。

## 7. 注册与配置

模块必须暴露统一注册函数：

```python
from hey_robot.skills.registry import SkillRegistry


def register_skills(registry: SkillRegistry) -> None:
    registry.register(InspectTargetSkill())
```

部署配置：

```yaml
skills:
  mode: production
  modules:
    - hey_robot.skills.builtin
    - my_robot_skills
  enabled:
    - inspect_scene
    - inspect_target
```

含义：

- `modules` 决定加载哪些注册模块；
- `enabled` 是当前部署对 Agent 开放的显式能力面；
- 未注册、重名、不支持当前机器人或缺少外部 capability，部署校验会失败；
- `driver_primitives` 声明的原语不被当前 robot driver 支持时，部署校验会失败；
- 未列入 `enabled` 的内部依赖仍可由已启用 Skill 调用，但不会直接暴露给 Agent。

## 8. 三类扩展

### 8.1 纯语义组合

例：先观察，再转向，再复查。

只改 Skill 和配置，不改 Agent 与硬件层。

### 8.2 新外部能力

例：VLA、导航服务、IK 服务。

需要：

1. 实现 capability service；
2. 在配置中声明该服务提供的能力名；
3. 创建隐藏的 capability Skill，设置 `external_capability`；
4. 由语义 Skill 通过 `ctx.invoke()` 调用。

### 8.3 新硬件原语

例：新夹爪命令、新关节模式、新传感器。

需要：

1. 在 Driver 或感知层实现真实能力；
2. 在对应 port/adapter 暴露稳定接口；
3. 创建隐藏原语 Skill，并在 `driver_primitives` 中声明它需要的运行时原语；
4. 再创建面向 Agent 的语义 Skill。

这不是架构泄漏，而是能力所有权边界：Skill 定义“做什么”，Driver 定义“硬件怎样做”。

## 9. 测试要求

至少覆盖：

- 输入缺失时被 `SkillRuntime.validate()` 拒绝；
- `execute()` 成功和失败结果；
- 嵌套 `ctx.invoke()` 的调用参数和失败传播；
- Registry 能加载模块且拒绝重名；
- 部署配置能启用 Skill；
- 机器人族不匹配或 capability 缺失时启动失败；
- 涉及资源的 Skill 具备冲突测试；
- 新硬件原语具备 Driver 或仿真集成测试。

## 10. 完成标准

一个普通语义 Skill 的提交不应修改 Agent、Controller、Runtime、协议和 Driver。若必须修改这些模块，应先判断新增的是系统级机制、外部 capability，还是全新的硬件原语，而不是把它伪装成普通 Skill 扩展。
