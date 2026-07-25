"""Native tabletop task skills with classic, VLA and hybrid handlers."""

from __future__ import annotations

from typing import Any

from hey_robot.skills.builtins.common import execute_robot_action
from hey_robot.skills.builtins.navigation import approach_object
from hey_robot.skills.builtins.vla import manipulate
from hey_robot.skills.context import SkillContext
from hey_robot.skills.models import Skill, SkillResult
from hey_robot.skills.registry import SkillRegistry

PICK_PARAMETERS = {
    "type": "object",
    "properties": {
        "object": {"type": "string"},
        "target": {"type": "string"},
        "task_prompt": {"type": "string"},
        "max_attempts": {
            "type": "integer",
            "minimum": 1,
            "maximum": 8,
        },
    },
    "additionalProperties": False,
}

PLACE_PARAMETERS = {
    "type": "object",
    "properties": {
        "object": {"type": "string"},
        "location": {"type": "string"},
        "target": {"type": "string"},
        "task_prompt": {"type": "string"},
        "placement_hint": {"type": "string"},
        "max_attempts": {
            "type": "integer",
            "minimum": 1,
            "maximum": 8,
        },
    },
    "additionalProperties": False,
}


async def classic_pick(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    return await execute_robot_action(ctx, "pick", arguments)


async def classic_place(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    return await execute_robot_action(ctx, "place", arguments)


async def vla_pick(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    target = arguments.get("object") or arguments.get("target") or "object"
    task_prompt = arguments.get("task_prompt") or f"grasp {target}"
    return await manipulate(
        ctx,
        {
            "task_prompt": task_prompt,
            "max_steps": int(arguments.get("max_attempts", 1)),
        },
    )


async def vla_place(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    location = arguments.get("location") or arguments.get("target") or "target"
    task_prompt = arguments.get("task_prompt") or f"place at {location}"
    return await manipulate(
        ctx,
        {
            "task_prompt": task_prompt,
            "max_steps": int(arguments.get("max_attempts", 1)),
        },
    )


async def hybrid_pick(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    staged = await approach_object(ctx, dict(arguments))
    if not staged.success:
        return staged
    return await vla_pick(ctx, arguments)


async def hybrid_place(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    return await vla_place(ctx, arguments)


def pick_skill(*, implementation: str = "classic") -> Skill:
    handler = {
        "classic": classic_pick,
        "vla": vla_pick,
        "hybrid": hybrid_pick,
    }.get(implementation)
    if handler is None:
        raise ValueError(f"unsupported pick implementation: {implementation}")
    return Skill(
        name="pick",
        description="Pick up an object from the tabletop.",
        parameters=PICK_PARAMETERS,
        handler=handler,
        resources=("robot_control", "camera"),
        timeout_sec=180.0,
        required_actions=() if implementation in {"vla", "hybrid"} else ("pick",),
        required_models=("manipulate",) if implementation in {"vla", "hybrid"} else (),
    )


def place_skill(*, implementation: str = "classic") -> Skill:
    handler = {
        "classic": classic_place,
        "vla": vla_place,
        "hybrid": hybrid_place,
    }.get(implementation)
    if handler is None:
        raise ValueError(f"unsupported place implementation: {implementation}")
    return Skill(
        name="place",
        description="Place a held object at a target location.",
        parameters=PLACE_PARAMETERS,
        handler=handler,
        resources=("robot_control", "camera"),
        timeout_sec=180.0,
        required_actions=() if implementation in {"vla", "hybrid"} else ("place",),
        required_models=("manipulate",) if implementation in {"vla", "hybrid"} else (),
    )


def register(
    registry: SkillRegistry, *, implementations: dict[str, str] | None = None
) -> None:
    implementations = implementations or {}
    registry.register(pick_skill(implementation=implementations.get("pick", "classic")))
    registry.register(
        place_skill(implementation=implementations.get("place", "classic"))
    )
