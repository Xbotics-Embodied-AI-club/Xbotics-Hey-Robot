"""新面向服务运行时的 Topic 名称。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Topics:
    # Compatibility names retained for the public 1.x protocol surface. The current
    # runtime does not subscribe to the retired distributed Skill execution topics.
    user_turn: str = "user.turn"
    conversation_turn: str = "conversation.turn"
    conversation_result: str = "conversation.result"
    agent_control: str = "agent.control"
    short_operation_command: str = "short_operation.command"
    skill_command: str = "skill.command"
    skill_cancel: str = "skill.cancel"
    skill_run_event: str = "skill.run.event"
    agent_reply: str = "agent.reply"
    skill_intent: str = "skill.intent"
    skill_event: str = "skill.event"
    skill_result: str = "skill.result"
    robot_observation: str = "robot.observation"
    robot_status: str = "robot.status"
    robot_action: str = "robot.action"
    runtime_event: str = "runtime.event"
    skill_control: str = "skill.control"
    skill_control_result: str = "skill.control.result"
    camera_frame: str = "robot.camera.frame"
    base_velocity_stream: str = "robot.base.velocity_stream"
    human_follow_command: str = "human_follow.command"
    human_follow_status: str = "human_follow.status"

    def for_robot(self, base: str, robot_id: str | None) -> str:
        if not robot_id:
            return base
        return f"{base}.{robot_id}"
