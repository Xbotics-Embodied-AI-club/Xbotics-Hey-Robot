from __future__ import annotations

from typing import Any, Protocol

from hey_robot.protocol import RobotAction, RobotObservation, SkillIntent


class ObservationActionCodec(Protocol):
    name: str

    def observation_to_policy_input(
        self, observation: RobotObservation, intent: SkillIntent
    ) -> Any: ...

    def policy_output_to_action(
        self, output: Any, observation: RobotObservation, intent: SkillIntent
    ) -> RobotAction: ...
