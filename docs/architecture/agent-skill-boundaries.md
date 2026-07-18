# Agent 与 Skill 边界

本文记录 Agent 层简化重构后的边界契约。

从快慢双系统视角看，Agent/Cognition 是上层慢系统，Skill OS、Foundation Model 和
Robot Runtime 构成下层快系统。本文件描述二者最关键的提交边界：慢系统只能表达
`SkillIntent`，不能绕过 Skill OS 直接产生机器人动作。

## 主链路

```text
AutonomousAgentService
  -> RobotAgentLoop
  -> RobotAgentCore
  -> AgentRuntime
  -> request_skill / request_perception
  -> SkillGateway
  -> SkillIntent
  -> SkillControllerService
  -> SkillRuntime
      -> RobotRuntime
      -> optional ModelServiceRegistry / gRPC ModelService
```

## 边界规则

- Agent 代码不直接提交 `RobotAction`。
- Agent 代码不依赖 driver primitive。
- LLM 自主动作必须通过 `request_skill`。
- direct action、busy-turn interrupt 也必须通过 `SkillGateway`。
- `SkillGateway` 是 Agent 层维护的唯一 `SkillIntent` 构造和提交边界。
- `AgentRuntime` 负责 message protocol、message window 和 response policy。
- Foundation Model 返回规划或动作结果，但不能绕过 Skill OS 直接访问 Robot Runtime。
- Robot Runtime 不依赖 cognition、Skill OS 或 Foundation backend。

## Deployment surface

`skills.mode` 决定 Agent 能看到的能力层级：

- `production` 只允许启用 `agent_visible=True` 的 semantic skill；
- `bringup` 允许显式启用 primitive，便于联调机器人。

当前仓库 real/sim 配置使用 `bringup`。即使 primitive 对 Agent 可见，它仍必须通过
`request_skill -> SkillGateway -> SkillControllerService`，不能变成直接 RobotAction。

## 当前模块拆分

- `RobotAgentCore`：机器人领域协调和 turn 级编排。
- `AgentRuntime`：模型调用、工具调用和主执行循环。
- `skill_gateway.py`：Skill 提交、安全检查、等待策略和 interrupt intent 构造。
- `service/skill_result_handler.py`：Skill 结果接入和任务侧归一化。
- `service/recovery_notifier.py`：recovery 发布和面向操作者的通知。

## 防回退要求

架构测试应继续阻止以下回退：

- 在 gateway 外直接构造 `SkillIntent(...)`
- Agent 模块直接依赖 `RobotAction`
- Agent 层依赖底层 driver primitive
- Foundation backend 导入 cognition 或 Skill OS
- Robot Runtime 导入上层系统模块
