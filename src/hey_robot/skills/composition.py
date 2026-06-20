from __future__ import annotations

from dataclasses import dataclass

from hey_robot.skills.actions import RobotSkillAction


@dataclass(frozen=True)
class SkillExecutionPlan:
    actions: tuple[RobotSkillAction, ...]
    strategy: str = "runtime_trace"
    notes: tuple[str, ...] = ()

    @property
    def is_composite(self) -> bool:
        return len(self.actions) > 1
