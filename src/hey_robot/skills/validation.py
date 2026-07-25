"""Startup validation for configured skill surfaces."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from hey_robot.robot_api import RobotClientCapabilities
from hey_robot.skills.models import Skill


@dataclass(frozen=True)
class SkillSurfaceIssue:
    skill: str
    message: str


def validate_skill_surface(
    skills: Iterable[Skill],
    *,
    robot_family: str,
    robot_capabilities: RobotClientCapabilities,
    model_capabilities: Iterable[str] = (),
) -> tuple[SkillSurfaceIssue, ...]:
    """Validate deterministic deployment requirements for visible skills."""

    action_names = {action.name for action in robot_capabilities.actions}
    model_names = set(model_capabilities)
    issues: list[SkillSurfaceIssue] = []
    for skill in skills:
        if skill.supported_robots and robot_family not in skill.supported_robots:
            issues.append(
                SkillSurfaceIssue(
                    skill.name,
                    f"robot family {robot_family!r} is not supported",
                )
            )
        missing_actions = sorted(set(skill.required_actions) - action_names)
        if missing_actions:
            issues.append(
                SkillSurfaceIssue(
                    skill.name,
                    f"missing robot actions: {', '.join(missing_actions)}",
                )
            )
        missing_models = sorted(set(skill.required_models) - model_names)
        if missing_models:
            issues.append(
                SkillSurfaceIssue(
                    skill.name,
                    f"missing model capabilities: {', '.join(missing_models)}",
                )
            )
    return tuple(issues)
