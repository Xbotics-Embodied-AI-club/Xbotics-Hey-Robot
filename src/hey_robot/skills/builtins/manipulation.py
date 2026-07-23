"""Native arm and gripper primitive skills."""

from __future__ import annotations

from typing import Any

from hey_robot.skills.builtins.common import execute_robot_action
from hey_robot.skills.context import SkillContext
from hey_robot.skills.models import Skill, SkillResult
from hey_robot.skills.registry import SkillRegistry

_SUPPORTED_ROBOTS = ("xlerobot", "so101", "so101_mobile")


async def set_arm_pose(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    return await execute_robot_action(ctx, "set_arm_pose", arguments)


async def move_arm_joints(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    return await execute_robot_action(ctx, "move_arm_joints", arguments)


async def set_gripper(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    return await execute_robot_action(ctx, "set_gripper", arguments)


SET_ARM_POSE = Skill(
    name="set_arm_pose",
    description="Move the arm to a named verified pose.",
    parameters={
        "type": "object",
        "properties": {"pose_name": {"type": "string"}},
        "required": ["pose_name"],
        "additionalProperties": False,
    },
    handler=set_arm_pose,
    resources=("arm",),
    timeout_sec=12.0,
    supported_robots=_SUPPORTED_ROBOTS,
    required_actions=("set_arm_pose",),
)

MOVE_ARM_JOINTS = Skill(
    name="move_arm_joints",
    description="Set multiple arm joints. Use mode=delta for relative movement.",
    parameters={
        "type": "object",
        "properties": {
            "joints": {"type": "object"},
            "mode": {"type": "string"},
        },
        "required": ["joints"],
        "additionalProperties": False,
    },
    handler=move_arm_joints,
    resources=("arm",),
    timeout_sec=10.0,
    supported_robots=_SUPPORTED_ROBOTS,
    required_actions=("move_arm_joints",),
)

SET_GRIPPER = Skill(
    name="set_gripper",
    description="Set gripper opening. Use opening_pct or action=open/close.",
    parameters={
        "type": "object",
        "properties": {
            "opening_pct": {"type": "number"},
            "action": {"type": "string"},
        },
        "additionalProperties": False,
    },
    handler=set_gripper,
    resources=("gripper",),
    timeout_sec=10.0,
    supported_robots=_SUPPORTED_ROBOTS,
    required_actions=("set_gripper",),
)


def register(registry: SkillRegistry) -> None:
    registry.register(SET_ARM_POSE)
    registry.register(MOVE_ARM_JOINTS)
    registry.register(SET_GRIPPER)
