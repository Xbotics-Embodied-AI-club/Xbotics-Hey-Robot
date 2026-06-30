from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCapability:
    name: str
    source: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    safety_level: str = "normal"
    read_only: bool = False
    destructive: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "description": self.description,
            "input_schema": self.input_schema,
            "safety_level": self.safety_level,
            "read_only": self.read_only,
            "destructive": self.destructive,
        }


@dataclass(frozen=True)
class RobotSkillCapability:
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    safety_level: str = "normal"
    required_resources: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    success_criteria: tuple[str, ...] = ()
    failure_modes: tuple[str, ...] = ()
    recovery_hints: tuple[str, ...] = ()
    timeout_sec: float = 10.0
    interruptible: bool = True
    feedback_mode: str = "status"
    refresh_observation: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "safety_level": self.safety_level,
            "required_resources": list(self.required_resources),
            "preconditions": list(self.preconditions),
            "success_criteria": list(self.success_criteria),
            "failure_modes": list(self.failure_modes),
            "recovery_hints": list(self.recovery_hints),
            "timeout_sec": self.timeout_sec,
            "interruptible": self.interruptible,
            "feedback_mode": self.feedback_mode,
            "refresh_observation": self.refresh_observation,
        }


@dataclass(frozen=True)
class CapabilityManifest:
    tools: tuple[ToolCapability, ...] = ()
    robot_skill_actions: tuple[RobotSkillCapability, ...] = ()
    robot_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tools": [item.to_dict() for item in self.tools],
            "robot_skill_actions": [
                item.to_dict() for item in self.robot_skill_actions
            ],
        }
