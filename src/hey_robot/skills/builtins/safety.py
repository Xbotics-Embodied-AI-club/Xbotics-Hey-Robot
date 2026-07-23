"""Native safety skills for the new skill runner."""

from __future__ import annotations

from typing import Any

from hey_robot.skills.builtins.common import execute_robot_action
from hey_robot.skills.context import SkillContext
from hey_robot.skills.models import Skill, SkillResult
from hey_robot.skills.registry import SkillRegistry

_SUPPORTED_ROBOTS = ("xlerobot", "so101", "so101_mobile")


async def stop_motion(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    return await execute_robot_action(ctx, "stop_motion", arguments)


async def reset_posture(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    return await execute_robot_action(ctx, "reset_posture", arguments)


STOP_MOTION = Skill(
    name="stop_motion",
    description="Stop robot motion. Set emergency=true for emergency stop.",
    parameters={
        "type": "object",
        "properties": {"emergency": {"type": "boolean", "default": False}},
        "additionalProperties": False,
    },
    handler=stop_motion,
    resources=("arm",),
    timeout_sec=8.0,
    supported_robots=_SUPPORTED_ROBOTS,
    required_actions=("stop_motion",),
)

RESET_POSTURE = Skill(
    name="reset_posture",
    description="Stop motion and return the robot to a safe default posture.",
    parameters={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    handler=reset_posture,
    resources=("arm", "gripper"),
    timeout_sec=15.0,
    supported_robots=_SUPPORTED_ROBOTS,
    required_actions=("reset_posture",),
)


def register(registry: SkillRegistry) -> None:
    registry.register(STOP_MOTION)
    registry.register(RESET_POSTURE)
