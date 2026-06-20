# 运行时形态

Hey Robot 是面向真实机器人的具身 Agent 运行时。当前主线目标是 `XLeRobot`。

## 主执行链路

```text
User Channel
  -> GatewayService
  -> bus topic: user_turn
  -> RobotAgentService
      -> RobotAgentLoop
      -> RobotAgentCore
      -> AgentRuntime
      -> request_capability / request_perception
      -> SkillGateway
  -> bus topic: skill_intent
  -> SkillControllerService
      -> SkillContractRuntime
      -> CapabilityRuntime
  -> bus topic: robot_action
  -> RobotService / RobotRuntime
      -> XLeRobotDriver / SO101Driver / LeKiwiDriver / MockRobotDriver
  -> bus topics: robot_status / robot_observation / skill_event / skill_result
```

## 边界摘要

- Agent 层只做任务理解、上下文组织和 Skill 级决策。
- Skill 层负责能力契约、资源门禁、就绪检查和执行生命周期。
- Robot 层负责真实硬件或仿真执行，并通过 `RobotRuntime / PerceptionService` 产出 observation 和 status。
- VLA 等外部模型能力通过 capability service 暴露，并保持在 Skill 边界之后。
