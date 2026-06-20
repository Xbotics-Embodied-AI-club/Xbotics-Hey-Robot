# Agent 与 Skill 边界

本文记录 Agent 层简化重构后的边界契约。

## 主链路

```text
RobotAgentService
  -> RobotAgentLoop
  -> RobotAgentCore
  -> AgentRuntime
  -> request_capability / request_perception
  -> SkillGateway
  -> SkillIntent
  -> SkillControllerService
  -> SkillRuntime / CapabilityRuntime
  -> RobotRuntime / Capability Services
```

## 边界规则

- Agent 代码不直接提交 `RobotAction`。
- Agent 代码不依赖 driver primitive。
- LLM 自主动作必须通过 `request_capability`。
- direct action、busy-turn interrupt 也必须通过 `SkillGateway`。
- `SkillGateway` 是 Agent 层维护的唯一 `SkillIntent` 构造和提交边界。
- `AgentRuntime` 负责 message protocol、message window 和 response policy。

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
