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
    task_prompt = str(
        arguments.get("task_prompt") or arguments.get("objective") or "manipulate"
    )
    if ctx.robot is None:
        return SkillResult(
            False,
            "robot client is unavailable for VLA action execution",
            "failed",
            data={},
            failure_mode="robot_client_unavailable",
            error="robot client is unavailable",
        )

    max_steps = max(1, int(arguments.get("max_steps", 1)))
    fresh_timeout = float(arguments.get("fresh_observation_timeout_sec", 2.0))
    observation = await ctx.observe(timeout_sec=fresh_timeout)
    before_frame_id = observation.frame_id
    after_frame_id: int | None = None
    executed_actions: list[dict[str, Any]] = []
    model_outputs: list[dict[str, Any]] = []

    for step_index in range(max_steps):
        ctx.raise_if_cancelled()
        result = await ctx.models.infer(
            "manipulate",
            {
                **dict(arguments),
                "task_prompt": task_prompt,
                "observation": _observation_payload(observation),
                "policy_session_id": ctx.run_id,
                "step_index": step_index,
                "max_steps": max_steps,
            },
            run_id=ctx.run_id,
            robot_id=ctx.robot_id,
            timeout_sec=arguments.get("model_timeout_sec"),
        )
        ctx.raise_if_cancelled()
        if not result.success:
            return SkillResult(
                False,
                result.summary,
                "failed",
                data={
                    "vla": dict(result.data),
                    "steps": executed_actions,
                    "before_frame_id": before_frame_id,
                    "after_frame_id": after_frame_id,
                },
                failure_mode=result.failure_mode or "model_failed",
                error=result.error,
            )

        model_data = dict(result.data)
        model_outputs.append(model_data)
        if _environment_done(model_data):
            return _vla_result(
                success=True,
                summary=result.summary,
                model_outputs=model_outputs,
                executed_actions=executed_actions,
                before_frame_id=before_frame_id,
                after_frame_id=after_frame_id,
                termination_reason="environment_done",
            )
        actions = _actions_from_model_data(model_data)
        if not actions:
            return _vla_result(
                success=True,
                summary=result.summary,
                model_outputs=model_outputs,
                executed_actions=executed_actions,
                before_frame_id=before_frame_id,
                after_frame_id=after_frame_id,
                termination_reason="model_done"
                if _vla_task_done(model_data)
                else "no_action",
            )

        for action in actions:
            ctx.raise_if_cancelled()
            action_result = await ctx.robot.execute(
                ctx.robot_id,
                action["name"],
                action["arguments"],
                run_id=ctx.run_id,
                expected_frame_id=observation.frame_id,
            )
            executed_actions.append(
                {
                    "step_index": step_index,
                    "action": action,
                    "success": action_result.success,
                    "summary": action_result.summary,
                    "data": dict(action_result.data),
                    "frame_id": action_result.frame_id,
                }
            )
            if not action_result.success:
                return _vla_result(
                    success=False,
                    summary=action_result.summary,
                    model_outputs=model_outputs,
                    executed_actions=executed_actions,
                    before_frame_id=before_frame_id,
                    after_frame_id=after_frame_id,
                    termination_reason="action_failed",
                    failure_mode=action_result.failure_mode,
                    error=action_result.error,
                )
            after_frame_id = action_result.frame_id
            if _environment_done(action_result.data):
                return _vla_result(
                    success=True,
                    summary=action_result.summary,
                    model_outputs=model_outputs,
                    executed_actions=executed_actions,
                    before_frame_id=before_frame_id,
                    after_frame_id=after_frame_id,
                    termination_reason="environment_done",
                )

        await ctx.progress(
            (step_index + 1) / max_steps,
            f"VLA 已完成 bounded step {step_index + 1}/{max_steps}",
        )
        if _vla_task_done(model_data):
            return _vla_result(
                success=True,
                summary=result.summary,
                model_outputs=model_outputs,
                executed_actions=executed_actions,
                before_frame_id=before_frame_id,
                after_frame_id=after_frame_id,
                termination_reason="model_done",
            )

        if step_index + 1 < max_steps:
            try:
                observation = await ctx.observe(
                    after_frame_id=observation.frame_id,
                    timeout_sec=fresh_timeout,
                )
            except TimeoutError:
                return _vla_result(
                    success=False,
                    summary="VLA action 后未获得 fresh observation。",
                    model_outputs=model_outputs,
                    executed_actions=executed_actions,
                    before_frame_id=before_frame_id,
                    after_frame_id=after_frame_id,
                    termination_reason="observation_stale",
                    failure_mode="observation_stale",
                )

    return SkillResult(
        True,
        f"VLA reached bounded limit ({max_steps} steps).",
        "completed",
        data={
            "vla": model_outputs[-1] if model_outputs else {},
            "vla_history": model_outputs,
            "steps": executed_actions,
            "termination_reason": "max_steps",
            "before_frame_id": before_frame_id,
            "after_frame_id": after_frame_id,
            "requires_reobservation": bool(executed_actions),
        },
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
            "fresh_observation_timeout_sec": {"type": "number", "default": 2.0},
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


def _actions_from_model_data(data: dict[str, Any]) -> list[dict[str, Any]]:
    chunk = data.get("action_chunk")
    if isinstance(chunk, dict):
        actions = chunk.get("actions")
        if isinstance(actions, list):
            normalized = [_normalize_action(action) for action in actions]
            return [action for action in normalized if action is not None]

    action = _normalize_action(data.get("primitive"))
    if action is not None:
        return [action]
    action = _normalize_action(data.get("action") or data.get("native_action"))
    if action is not None:
        return [action]
    values = data.get("values")
    if isinstance(values, list):
        return [{"name": "embodiment_native_action", "arguments": {"values": values}}]
    return []


def _normalize_action(candidate: Any) -> dict[str, Any] | None:
    if not isinstance(candidate, dict):
        return None
    name = candidate.get("name") or candidate.get("action")
    arguments = candidate.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {
            key: value
            for key, value in candidate.items()
            if key not in {"name", "action", "done"}
        }
    if isinstance(name, str):
        return {"name": name, "arguments": dict(arguments)}
    return None


def _vla_task_done(data: dict[str, Any]) -> bool:
    if bool(data.get("done")):
        return True
    for key in ("policy_result", "action_chunk"):
        value = data.get(key)
        if isinstance(value, dict) and bool(value.get("done")):
            return True
    return False


def _environment_done(data: dict[str, Any]) -> bool:
    if bool(data.get("environment_done")):
        return True
    for key in ("policy_result", "action_chunk", "environment"):
        value = data.get(key)
        if isinstance(value, dict) and bool(
            value.get("environment_done") or value.get("done_by_environment")
        ):
            return True
    return False


def _observation_payload(observation: Any) -> dict[str, Any]:
    return {
        "frame_id": observation.frame_id,
        "timestamp": observation.envelope.timestamp,
        "images": [asdict(image) for image in observation.images],
        "proprioception": list(observation.proprioception),
        "raw": dict(observation.raw),
    }


def _vla_result(
    *,
    success: bool,
    summary: str,
    model_outputs: list[dict[str, Any]],
    executed_actions: list[dict[str, Any]],
    before_frame_id: int,
    after_frame_id: int | None,
    termination_reason: str,
    failure_mode: str | None = None,
    error: str | None = None,
) -> SkillResult:
    return SkillResult(
        success,
        summary,
        "completed" if success else "failed",
        data={
            "vla": model_outputs[-1] if model_outputs else {},
            "vla_history": model_outputs,
            "steps": executed_actions,
            "termination_reason": termination_reason,
            "before_frame_id": before_frame_id,
            "after_frame_id": after_frame_id,
            "requires_reobservation": bool(executed_actions),
        },
        failure_mode=failure_mode,
        error=error,
    )
