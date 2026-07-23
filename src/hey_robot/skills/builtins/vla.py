"""Native VLA bounded-option skills."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from hey_robot.skills.context import SkillContext
from hey_robot.skills.models import Skill, SkillResult
from hey_robot.skills.registry import SkillRegistry


async def manipulate(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    if ctx.models is None:
        return SkillResult(
            False,
            "VLA model router is unavailable.",
            "failed",
            failure_mode="model_service_unavailable",
            error="model router is unavailable",
        )
    observation = await ctx.observe()
    task_prompt = str(
        arguments.get("task_prompt") or arguments.get("objective") or "manipulate"
    )
    result = await ctx.models.infer(
        "manipulate",
        {
            **dict(arguments),
            "task_prompt": task_prompt,
            "observation": {
                "frame_id": observation.frame_id,
                "timestamp": observation.envelope.timestamp,
                "images": [asdict(image) for image in observation.images],
                "proprioception": list(observation.proprioception),
                "raw": dict(observation.raw),
            },
            "policy_session_id": ctx.run_id,
        },
        run_id=ctx.run_id,
        robot_id=ctx.robot_id,
        timeout_sec=arguments.get("model_timeout_sec"),
    )
    if not result.success:
        return SkillResult(
            False,
            result.summary,
            "failed",
            data=dict(result.data),
            failure_mode=result.failure_mode or "vla_inference_failed",
            error=result.error,
        )
    action = _action_from_model_data(result.data)
    if action is None:
        return SkillResult(
            True,
            result.summary,
            "completed",
            data={**dict(result.data), "requires_reobservation": True},
        )
    if ctx.robot is None:
        return SkillResult(
            False,
            "robot client is unavailable for VLA action execution",
            "failed",
            data=dict(result.data),
            failure_mode="robot_client_unavailable",
            error="robot client is unavailable",
        )
    action_result = await ctx.robot.execute(
        ctx.robot_id,
        action["name"],
        action["arguments"],
        run_id=ctx.run_id,
        expected_frame_id=observation.frame_id,
    )
    success = bool(action_result.success)
    return SkillResult(
        success,
        action_result.summary,
        "completed" if success else "failed",
        data={
            **dict(result.data),
            "action": action,
            "action_result": dict(action_result.data),
            "requires_reobservation": True,
        },
        failure_mode=action_result.failure_mode,
        error=action_result.error,
    )


MANIPULATE = Skill(
    name="manipulate",
    description="Execute one bounded VLA manipulation option.",
    parameters={
        "type": "object",
        "properties": {
            "task_prompt": {"type": "string"},
            "objective": {"type": "string"},
            "max_steps": {"type": "integer", "default": 1},
            "model_timeout_sec": {"type": "number"},
        },
        "additionalProperties": True,
    },
    handler=manipulate,
    resources=("robot_control", "camera"),
    timeout_sec=180.0,
    supported_robots=("xlerobot", "so101", "so101_mobile", "robocasa"),
    required_actions=("embodiment_native_action",),
    required_models=("manipulate",),
)


def register(registry: SkillRegistry) -> None:
    registry.register(MANIPULATE)


def _action_from_model_data(data: dict[str, Any]) -> dict[str, Any] | None:
    primitive = data.get("primitive")
    if isinstance(primitive, dict):
        name = primitive.get("name") or primitive.get("action")
        arguments = primitive.get("arguments", {})
        if isinstance(name, str) and isinstance(arguments, dict):
            return {"name": name, "arguments": dict(arguments)}
    action = data.get("action") or data.get("native_action")
    if isinstance(action, dict):
        name = action.get("name") or action.get("action") or "embodiment_native_action"
        arguments = action.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {key: value for key, value in action.items() if key != "name"}
        if isinstance(name, str):
            return {"name": name, "arguments": dict(arguments)}
    values = data.get("values")
    if isinstance(values, list):
        return {"name": "embodiment_native_action", "arguments": {"values": values}}
    return None
