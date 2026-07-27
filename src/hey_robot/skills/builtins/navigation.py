"""Native base movement skill definitions."""

from __future__ import annotations

from typing import Any

from hey_robot.skills.builtins.common import execute_robot_action
from hey_robot.skills.context import SkillContext
from hey_robot.skills.models import Skill, SkillResult
from hey_robot.skills.registry import SkillRegistry
from hey_robot.skills.vln import VLNOptionRequest, VLNOptionRunner


async def move_base(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    return await execute_robot_action(ctx, "move_base", arguments)


async def turn_base(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    return await execute_robot_action(ctx, "turn_base", arguments)


async def base_velocity_step(
    ctx: SkillContext, arguments: dict[str, Any]
) -> SkillResult:
    return await execute_robot_action(ctx, "base_velocity_step", arguments)


async def navigate_to(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    result = await VLNOptionRunner().run(
        ctx, VLNOptionRequest("navigate_to", arguments)
    )
    return result.to_skill_result()


async def approach_object(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    result = await VLNOptionRunner().run(
        ctx, VLNOptionRequest("approach_object", arguments)
    )
    return result.to_skill_result()


MOVE_BASE = Skill(
    name="move_base",
    description="Move the base forward, backward, left, or right in the robot body frame by a short distance in centimeters.",
    parameters={
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["forward", "backward", "left", "right"],
            },
            "distance_cm": {
                "type": "number",
                "default": 20.0,
                "minimum": 5.0,
                "maximum": 50.0,
            },
        },
        "required": ["direction"],
        "additionalProperties": False,
    },
    handler=move_base,
    resources=("base",),
    timeout_sec=20.0,
    required_actions=("move_base",),
)

TURN_BASE = Skill(
    name="turn_base",
    description="Turn the base left or right from the robot perspective by a bounded angle in degrees.",
    parameters={
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": ["left", "right"]},
            "angle_deg": {"type": "number"},
        },
        "required": ["direction", "angle_deg"],
        "additionalProperties": False,
    },
    handler=turn_base,
    resources=("base",),
    timeout_sec=20.0,
    required_actions=("turn_base",),
)

BASE_VELOCITY_STEP = Skill(
    name="base_velocity_step",
    description="Apply a short bounded base velocity command for supervised following.",
    parameters={
        "type": "object",
        "properties": {
            "vx": {"type": "number"},
            "vy": {"type": "number"},
            "wz": {"type": "number"},
            "duration_ms": {"type": "integer"},
        },
        "required": ["vx", "vy", "wz", "duration_ms"],
        "additionalProperties": False,
    },
    handler=base_velocity_step,
    resources=("base",),
    timeout_sec=3.0,
    required_actions=("base_velocity_step",),
)

NAVIGATE_TO = Skill(
    name="navigate_to",
    description="Navigate toward a semantic target using a foundation VLN planner.",
    parameters={
        "type": "object",
        "properties": {
            "target": {"type": "string"},
            "instruction": {"type": "string"},
            "camera": {"type": "string"},
            "image_path": {"type": "string"},
            "execute_primitives": {"type": "boolean"},
            "max_steps": {"type": "integer"},
            "model_timeout_sec": {"type": "number"},
            "fresh_observation_timeout_sec": {"type": "number"},
        },
        "required": ["target"],
        "additionalProperties": False,
    },
    handler=navigate_to,
    resources=("camera", "base"),
    timeout_sec=180.0,
    required_actions=(
        "move_base",
        "turn_base",
        "base_velocity_step",
        "stop_motion",
    ),
    required_models=("navigate_to",),
)

APPROACH_OBJECT = Skill(
    name="approach_object",
    description="Approach a visible or named object using a foundation VLN planner.",
    parameters=NAVIGATE_TO.parameters,
    handler=approach_object,
    resources=("camera", "base"),
    timeout_sec=180.0,
    required_actions=(
        "move_base",
        "turn_base",
        "base_velocity_step",
        "stop_motion",
    ),
    required_models=("approach_object",),
)


def register(registry: SkillRegistry) -> None:
    registry.register(MOVE_BASE)
    registry.register(TURN_BASE)
    registry.register(BASE_VELOCITY_STEP)
    registry.register(NAVIGATE_TO)
    registry.register(APPROACH_OBJECT)
