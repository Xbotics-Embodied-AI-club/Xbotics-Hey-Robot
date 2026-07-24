"""Native base movement skill definitions."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from hey_robot.skills.builtins.common import execute_robot_action
from hey_robot.skills.context import SkillContext
from hey_robot.skills.models import Skill, SkillResult
from hey_robot.skills.registry import SkillRegistry


async def move_base(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    return await execute_robot_action(ctx, "move_base", arguments)


async def turn_base(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    return await execute_robot_action(ctx, "turn_base", arguments)


async def base_velocity_step(
    ctx: SkillContext, arguments: dict[str, Any]
) -> SkillResult:
    return await execute_robot_action(ctx, "base_velocity_step", arguments)


async def navigate_to(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    return await _run_vln(ctx, arguments, capability="navigate_to")


async def approach_object(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    return await _run_vln(ctx, arguments, capability="approach_object")


async def _run_vln(
    ctx: SkillContext, arguments: dict[str, Any], *, capability: str
) -> SkillResult:
    """执行 bounded VLN loop：observe → infer → base action → re-observe。"""
    if ctx.models is None:
        return SkillResult(
            False,
            "VLN model router 不可用。",
            "failed",
            failure_mode="model_service_unavailable",
            error="model router is unavailable",
        )
    execute_primitives = bool(arguments.get("execute_primitives", True))
    if execute_primitives and ctx.robot is None:
        return SkillResult(
            False,
            "执行 VLN primitive 需要 RobotClient。",
            "failed",
            failure_mode="robot_client_unavailable",
            error="robot client is unavailable",
        )

    max_steps = max(1, int(arguments.get("max_steps", 30)))
    observation = await ctx.observe()
    steps: list[dict[str, Any]] = []
    planner_history: list[dict[str, Any]] = []
    look_down_requested = False

    for step_index in range(max_steps):
        ctx.raise_if_cancelled()
        payload = _vln_payload(
            arguments,
            observation,
            reset_policy=step_index == 0,
            policy_session_id=ctx.run_id,
            look_down=look_down_requested,
        )
        result = await ctx.models.infer(
            capability,
            payload,
            run_id=ctx.run_id,
            robot_id=ctx.robot_id,
            timeout_sec=arguments.get("model_timeout_sec"),
        )
        planner = _planner_data(result.data)
        planner_history.append(planner)
        if not result.success:
            return _vln_result(
                False,
                result.summary,
                planner_history,
                steps,
                "planner_failed",
                failure_mode=result.failure_mode or "vln_planner_failed",
                error=result.error,
            )

        if _requires_secondary_observation(planner):
            if step_index + 1 >= max_steps:
                return _vln_result(
                    False,
                    "VLN 请求 secondary observation，但没有剩余 planning step。",
                    planner_history,
                    steps,
                    "secondary_observation_required",
                    failure_mode="vln_secondary_observation_required",
                )
            look_down_requested = True
            await ctx.progress(
                (step_index + 1) / max_steps,
                "VLN 请求 secondary observation，准备重新观察。",
            )
            observation = await ctx.observe()
            continue

        try:
            command = _planner_to_action(planner)
        except ValueError as exc:
            return _vln_result(
                False,
                "VLN planner 未返回可执行 primitive。",
                planner_history,
                steps,
                "no_valid_goal",
                failure_mode="vln_no_valid_goal",
                error=str(exc),
            )

        if not execute_primitives:
            return _vln_result(
                True,
                result.summary,
                planner_history,
                steps,
                "plan_only",
                command=command,
            )

        action_result = await ctx.robot.execute(
            ctx.robot_id,
            command["name"],
            command["arguments"],
            run_id=ctx.run_id,
            expected_frame_id=observation.frame_id,
        )
        steps.append(
            {
                "step_index": step_index,
                "primitive": command["name"],
                "arguments": command["arguments"],
                "reason": command["reason"],
                "success": action_result.success,
                "summary": action_result.summary,
                "frame_id": action_result.frame_id,
                "data": dict(action_result.data),
            }
        )
        if not action_result.success:
            return _vln_result(
                False,
                action_result.summary,
                planner_history,
                steps,
                "primitive_execution_failed",
                failure_mode=action_result.failure_mode or "primitive_execution_failed",
                error=action_result.error,
            )
        await ctx.progress(
            (step_index + 1) / max_steps,
            f"VLN 已执行 {command['name']} ({step_index + 1}/{max_steps})。",
        )
        if command["name"] == "stop_motion":
            return _vln_result(
                True,
                "VLN planner 已确认到达目标。",
                planner_history,
                steps,
                "model_stop",
            )
        if step_index + 1 < max_steps:
            observation = await ctx.observe()

    return _vln_result(
        False,
        f"VLN 已达到 max_steps={max_steps}，尚未确认到达目标。",
        planner_history,
        steps,
        "max_steps",
        failure_mode="max_steps",
    )


def _vln_payload(
    arguments: dict[str, Any],
    observation: Any,
    *,
    reset_policy: bool,
    policy_session_id: str,
    look_down: bool,
) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in arguments.items()
        if key
        not in {
            "execute_primitives",
            "max_steps",
            "model_timeout_sec",
            "fresh_observation_timeout_sec",
        }
    }
    payload["observation"] = {
        "frame_id": observation.frame_id,
        "timestamp": observation.envelope.timestamp,
        "images": [asdict(image) for image in observation.images],
        "proprioception": list(observation.proprioception),
        "raw": dict(observation.raw),
    }
    payload["policy_session_id"] = policy_session_id
    payload["reset_policy"] = reset_policy
    if look_down:
        payload["look_down"] = True
    return payload


def _planner_data(data: dict[str, Any]) -> dict[str, Any]:
    planner = data.get("vln")
    if isinstance(planner, dict):
        return dict(planner)
    planner = data.get("planner")
    if isinstance(planner, dict):
        return dict(planner)
    return dict(data)


def _requires_secondary_observation(planner: dict[str, Any]) -> bool:
    return bool(planner.get("requires_secondary_observation")) or (
        planner.get("mode") == "look_down_required"
    )


def _planner_to_action(planner: dict[str, Any]) -> dict[str, Any]:
    if bool(planner.get("stop")) or planner.get("mode") == "stop":
        return {"name": "stop_motion", "arguments": {}, "reason": "planner_stop"}
    heading = planner.get("heading_deg")
    if isinstance(heading, int | float):
        if abs(heading) < 10.0:
            return {
                "name": "move_base",
                "arguments": {"direction": "forward", "distance_cm": 15.0},
                "reason": "heading_centered",
            }
        return {
            "name": "turn_base",
            "arguments": {
                "direction": "right" if heading > 0 else "left",
                "angle_deg": min(abs(heading), 30.0),
            },
            "reason": "planner_heading",
        }
    pixel_goal = planner.get("pixel_goal")
    if isinstance(pixel_goal, list | tuple) and len(pixel_goal) >= 2:
        offset = float(pixel_goal[1]) - 320.0
        if abs(offset) <= 80.0:
            return {
                "name": "move_base",
                "arguments": {"direction": "forward", "distance_cm": 15.0},
                "reason": "pixel_centered",
            }
        angle = 10.0 + 20.0 * min(abs(offset) / 320.0, 1.0)
        return {
            "name": "turn_base",
            "arguments": {
                "direction": "right" if offset > 0 else "left",
                "angle_deg": angle,
            },
            "reason": "pixel_off_center",
        }
    raise ValueError("planner output lacks stop, heading_deg, or pixel_goal")


def _vln_result(
    success: bool,
    summary: str,
    planner_history: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    termination_reason: str,
    *,
    command: dict[str, Any] | None = None,
    failure_mode: str | None = None,
    error: str | None = None,
) -> SkillResult:
    return SkillResult(
        success,
        summary,
        "completed" if success else "failed",
        data={
            "vln": planner_history[-1] if planner_history else {},
            "vln_history": planner_history,
            "steps": steps,
            "termination_reason": termination_reason,
            "command": command,
            "requires_reobservation": bool(steps),
        },
        failure_mode=failure_mode,
        error=error,
    )


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
    timeout_sec=8.0,
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
    timeout_sec=8.0,
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
        },
        "required": ["target"],
        "additionalProperties": False,
    },
    handler=navigate_to,
    resources=("camera", "base"),
    timeout_sec=180.0,
    required_actions=("move_base", "turn_base", "stop_motion"),
    required_models=("navigate_to",),
)

APPROACH_OBJECT = Skill(
    name="approach_object",
    description="Approach a visible or named object using a foundation VLN planner.",
    parameters=NAVIGATE_TO.parameters,
    handler=approach_object,
    resources=("camera", "base"),
    timeout_sec=180.0,
    required_actions=("move_base", "turn_base", "stop_motion"),
    required_models=("approach_object",),
)


def register(registry: SkillRegistry) -> None:
    registry.register(MOVE_BASE)
    registry.register(TURN_BASE)
    registry.register(BASE_VELOCITY_STEP)
    registry.register(NAVIGATE_TO)
    registry.register(APPROACH_OBJECT)
