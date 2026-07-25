"""Native perception skill definitions."""

from __future__ import annotations

from typing import Any

from hey_robot.skills.builtins.common import execute_robot_action
from hey_robot.skills.context import SkillContext
from hey_robot.skills.models import Skill, SkillResult
from hey_robot.skills.registry import SkillRegistry


async def inspect_scene(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    return await execute_robot_action(ctx, "inspect_scene", arguments)


async def look_around(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    return await execute_robot_action(ctx, "look_around", arguments)


async def detect_marker(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    return await execute_robot_action(ctx, "detect_marker", arguments)


INSPECT_SCENE = Skill(
    name="inspect_scene",
    description="Inspect the current scene and return grounded visual evidence.",
    parameters={
        "type": "object",
        "properties": {"question": {"type": "string"}},
        "additionalProperties": False,
    },
    handler=inspect_scene,
    resources=("camera",),
    timeout_sec=45.0,
    supported_robots=("xlerobot", "so101", "so101_mobile", "robocasa"),
    required_actions=("inspect_scene",),
)

LOOK_AROUND = Skill(
    name="look_around",
    description="Collect visual evidence from multiple short viewing directions.",
    parameters={
        "type": "object",
        "properties": {"question": {"type": "string"}},
        "additionalProperties": False,
    },
    handler=look_around,
    resources=("camera", "base"),
    timeout_sec=30.0,
    required_actions=("look_around", "turn_base"),
)

DETECT_MARKER = Skill(
    name="detect_marker",
    description="Detect a workspace marker in the current camera frame.",
    parameters={
        "type": "object",
        "properties": {"marker_id": {"type": "integer"}},
        "additionalProperties": False,
    },
    handler=detect_marker,
    resources=("camera",),
    timeout_sec=6.0,
    required_actions=("detect_marker",),
)


def register(registry: SkillRegistry) -> None:
    registry.register(INSPECT_SCENE)
    registry.register(LOOK_AROUND)
    registry.register(DETECT_MARKER)
