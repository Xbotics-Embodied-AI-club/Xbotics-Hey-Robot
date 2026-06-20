from __future__ import annotations

import time
from typing import Any

from hey_robot.protocol import RobotAction, RobotObservation, SkillIntent


class SimpleVectorCodec:
    name = "simple"

    def observation_to_policy_input(
        self, observation: RobotObservation, intent: SkillIntent
    ) -> dict[str, Any]:
        return {
            "frame_id": observation.frame_id,
            "proprioception": observation.proprioception,
            "task": intent.objective,
            "raw": observation.raw,
        }

    def policy_output_to_action(
        self, output: Any, _observation: RobotObservation, intent: SkillIntent
    ) -> RobotAction:
        values = output if isinstance(output, list) else [float(output)]
        return RobotAction(
            values=[float(value) for value in values],
            envelope=intent.envelope,
            skill_id=intent.skill_id,
            timestamp=time.time(),
        )
