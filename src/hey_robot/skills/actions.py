from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from hey_robot.protocol import RobotAction, SkillIntent


@dataclass(frozen=True)
class RobotSkillAction:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    safety_level: str = "normal"
    expected_duration_sec: float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("skill action name must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": dict(self.arguments),
            "safety_level": self.safety_level,
            "expected_duration_sec": self.expected_duration_sec,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RobotSkillAction:
        return cls(
            name=str(payload.get("name") or payload.get("skill") or ""),
            arguments=dict(payload.get("arguments") or payload.get("args") or {}),
            safety_level=str(payload.get("safety_level") or "normal"),
            expected_duration_sec=(
                float(payload["expected_duration_sec"])
                if payload.get("expected_duration_sec") is not None
                else None
            ),
        )

    def to_robot_action(self, intent: SkillIntent) -> RobotAction:
        return RobotAction(
            envelope=intent.envelope,
            skill_id=intent.skill_id,
            values=[],
            timestamp=time.time(),
            metadata={
                "action_type": "skill",
                "skill": self.to_dict(),
            },
        )

    @classmethod
    def from_robot_action(cls, action: RobotAction) -> RobotSkillAction:
        if action.metadata.get("action_type") != "skill":
            raise ValueError("robot action is not a skill action")
        skill = action.metadata.get("skill")
        if not isinstance(skill, dict):
            raise ValueError("skill action metadata is missing metadata.skill")
        return cls.from_dict(skill)


@dataclass(frozen=True)
class RobotSkillResult:
    success: bool
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            **dict(self.data),
        }

    @classmethod
    def from_response(
        cls, response: dict[str, Any], *, default_message: str = ""
    ) -> RobotSkillResult:
        return cls(
            success=bool(response.get("success", False)),
            message=str(response.get("message") or default_message),
            data=dict(response),
        )
