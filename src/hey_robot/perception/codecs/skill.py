from __future__ import annotations

from typing import Any

from hey_robot.protocol import RobotAction, RobotObservation, SkillIntent
from hey_robot.skills import RobotSkillAction


class RobotSkillActionCodec:
    name = "skill"

    def observation_to_policy_input(
        self, observation: RobotObservation, intent: SkillIntent
    ) -> dict[str, Any]:
        return {
            "frame_id": observation.frame_id,
            "task": intent.objective,
            "skill": intent.name,
            "arguments": intent.arguments,
            "image_count": len(observation.images),
            "artifact_count": len(observation.artifacts),
            "proprioception": observation.proprioception,
            "raw": observation.raw,
        }

    def policy_output_to_action(
        self, output: Any, _observation: RobotObservation, intent: SkillIntent
    ) -> RobotAction:
        if isinstance(output, RobotSkillAction):
            skill = output
        elif isinstance(output, dict):
            skill = RobotSkillAction.from_dict(output)
        else:
            raise TypeError("skill policy output must be RobotSkillAction or dict")
        return skill.to_robot_action(intent)
