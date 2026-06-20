"""Topic names for the new service-oriented runtime."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Topics:
    user_turn: str = "user.turn"
    agent_reply: str = "agent.reply"
    skill_intent: str = "skill.intent"
    skill_event: str = "skill.event"
    skill_result: str = "skill.result"
    robot_observation: str = "robot.observation"
    camera_observation: str = "camera.observation"
    robot_status: str = "robot.status"
    robot_action: str = "robot.action"
    runtime_event: str = "runtime.event"
    camera_frame: str = "robot.camera.frame"
    base_velocity_stream: str = "robot.base.velocity_stream"
    human_follow_command: str = "human_follow.command"
    human_follow_status: str = "human_follow.status"

    def for_robot(self, base: str, robot_id: str | None) -> str:
        if not robot_id:
            return base
        return f"{base}.{robot_id}"
