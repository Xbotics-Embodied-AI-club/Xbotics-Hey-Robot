from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

FeedbackMode = str


@dataclass(frozen=True)
class RobotSkillSpec:
    """Runtime contract derived from the canonical plugin SkillSpec."""

    name: str
    description: str
    level: str = "primitive"
    agent_visible: bool = True
    category: str = "general"
    input_schema: dict[str, Any] = field(default_factory=dict)
    safety_level: str = "normal"
    supported_robots: tuple[str, ...] = ()
    external_capability: str | None = None
    driver_primitives: tuple[str, ...] = ()
    required_resources: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    success_criteria: tuple[str, ...] = ()
    failure_modes: tuple[str, ...] = ()
    recovery_hints: tuple[str, ...] = ()
    timeout_sec: float = 10.0
    interruptible: bool = True
    feedback_mode: FeedbackMode = "status"
    refresh_observation: bool = True
    capability_type: str | None = None
    goal_effects: tuple[str, ...] = ()
    evidence_outputs: tuple[str, ...] = ()
    cannot_satisfy: tuple[str, ...] = ()

    def supports(self, robot_type: str | None) -> bool:
        return (
            robot_type is None
            or not self.supported_robots
            or robot_type in self.supported_robots
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in (
            "supported_robots",
            "driver_primitives",
            "required_resources",
            "preconditions",
            "success_criteria",
            "failure_modes",
            "recovery_hints",
            "goal_effects",
            "evidence_outputs",
            "cannot_satisfy",
        ):
            data[key] = list(data[key])
        return data


class RobotSkillCatalog:
    """Read-only runtime view derived from a SkillRegistry."""

    def __init__(
        self, specs: list[RobotSkillSpec] | tuple[RobotSkillSpec, ...]
    ) -> None:
        self._specs = {spec.name: spec for spec in specs}

    def get(self, name: str) -> RobotSkillSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"unknown robot skill action: {name}") from exc

    def list(self, *, robot_type: str | None = None) -> tuple[RobotSkillSpec, ...]:
        return tuple(spec for spec in self._specs.values() if spec.supports(robot_type))

    def list_agent_visible(
        self, *, robot_type: str | None = None
    ) -> tuple[RobotSkillSpec, ...]:
        return tuple(
            spec for spec in self.list(robot_type=robot_type) if spec.agent_visible
        )

    def names(self, *, robot_type: str | None = None) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.list(robot_type=robot_type))

    def agent_visible_names(self, *, robot_type: str | None = None) -> tuple[str, ...]:
        return tuple(
            spec.name for spec in self.list_agent_visible(robot_type=robot_type)
        )

    def resolve(
        self, name: str | None, *, robot_type: str | None = None
    ) -> RobotSkillSpec:
        if not name:
            raise KeyError("robot skill action name is required")
        spec = self.get(name)
        if not spec.supports(robot_type):
            raise KeyError(
                f"robot skill action {name!r} does not support robot type {robot_type!r}"
            )
        return spec
