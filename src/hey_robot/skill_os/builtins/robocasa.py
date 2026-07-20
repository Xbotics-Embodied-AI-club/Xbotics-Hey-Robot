from __future__ import annotations

from typing import Any

from hey_robot.skill_os.base import BaseSkill, SkillResult
from hey_robot.skill_os.builtins.common import spec


class RoboCasaRolloutSkill(BaseSkill):
    """Run one complete RoboCasa benchmark episode in an external worker.

    The worker owns the PandaOmron environment, VLA checkpoint and 12D control
    loop.  This skill deliberately has no local robot primitives: RoboCasa is a
    benchmark embodiment, not the configured Hey Robot driver.
    """

    spec = spec(
        "robocasa_rollout",
        "Run one RoboCasa365 VLA benchmark episode in the external worker.",
        category="benchmark",
        input_schema={
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "n_episodes": {
                    "type": "integer",
                    "const": 1,
                    "default": 1,
                },
                "seed": {"type": "integer", "default": 1000},
                "policy_path": {"type": "string"},
                "obj_registries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["lightwheel"],
                },
                "record_video": {"type": "boolean", "default": True},
            },
            "required": ["task"],
        },
        required_model_service="robocasa_rollout",
        supported_robots=(),
        safety_level="normal",
        timeout_sec=1800.0,
        feedback_mode="status",
        refresh_observation=False,
        goal_effects=("runs_external_robocasa_benchmark",),
        evidence_outputs=("robocasa_rollout_result",),
        failure_modes=(
            "invalid_task",
            "checkpoint_unavailable",
            "asset_unavailable",
            "environment_reset_failed",
            "policy_load_failed",
            "cuda_out_of_memory",
            "rollout_timeout",
            "rollout_cancelled",
            "task_unsuccessful",
            "result_parse_failed",
        ),
    )

    async def execute(self, ctx: Any, arguments: dict[str, Any]) -> SkillResult:
        if ctx.model_services is None:
            return SkillResult(
                success=False,
                summary="robocasa_rollout requires a deployed model service",
                status="failed",
                failure_mode="model_service_unavailable",
                error="model service port is unavailable",
            )

        result = await ctx.model_services.call(
            self.spec.required_model_service or self.spec.name,
            arguments,
        )
        return SkillResult(
            success=bool(getattr(result, "success", False)),
            summary=str(getattr(result, "summary", "") or "RoboCasa rollout ended"),
            status=str(
                getattr(result, "status", "")
                or ("completed" if getattr(result, "success", False) else "failed")
            ),
            failure_mode=getattr(result, "failure_mode", None),
            error=getattr(result, "error", None),
            data={"metrics": dict(getattr(result, "metrics", {}) or {})},
        )


class RoboCasaOptionSkill(BaseSkill):
    """Run one bounded RoboCasa option with the live VLA policy."""

    spec = spec(
        "robocasa_option",
        "Run one bounded RoboCasa option with the external VLA policy.",
        category="benchmark",
        input_schema={
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "objective": {"type": "string"},
                "option_command": {"type": "string"},
                "session_id": {"type": "string"},
                "policy_path": {"type": "string"},
                "seed": {"type": "integer", "default": 1000},
                "max_steps": {"type": "integer", "default": 30},
                "reset_episode": {"type": "boolean", "default": False},
                "close_episode": {"type": "boolean", "default": False},
                "device": {"type": "string", "default": "cuda"},
            },
            "required": ["task"],
        },
        required_model_service="robocasa_option",
        supported_robots=(),
        safety_level="normal",
        timeout_sec=1800.0,
        feedback_mode="status",
        refresh_observation=False,
        goal_effects=("runs_external_robocasa_option",),
        evidence_outputs=("robocasa_option_result",),
        failure_modes=(
            "invalid_task",
            "checkpoint_unavailable",
            "asset_unavailable",
            "environment_reset_failed",
            "policy_load_failed",
            "action_schema_mismatch",
            "observation_schema_mismatch",
            "session_task_mismatch",
            "episode_already_done",
            "option_timeout",
            "execution_failed",
        ),
    )

    async def execute(self, ctx: Any, arguments: dict[str, Any]) -> SkillResult:
        if ctx.model_services is None:
            return SkillResult(
                success=False,
                summary="robocasa_option requires a deployed model service",
                status="failed",
                failure_mode="model_service_unavailable",
                error="model service port is unavailable",
            )

        payload = dict(arguments)
        payload.setdefault("session_id", getattr(ctx, "skill_id", None))
        payload.setdefault(
            "objective", payload.get("option_command") or payload["task"]
        )
        payload.setdefault(
            "option_command", payload.get("objective") or payload["task"]
        )
        result = await ctx.model_services.call(
            self.spec.required_model_service or self.spec.name,
            payload,
        )
        return SkillResult(
            success=bool(getattr(result, "success", False)),
            summary=str(getattr(result, "summary", "") or "RoboCasa option ended"),
            status=str(
                getattr(result, "status", "")
                or ("completed" if getattr(result, "success", False) else "failed")
            ),
            failure_mode=getattr(result, "failure_mode", None),
            error=getattr(result, "error", None),
            data={"metrics": dict(getattr(result, "metrics", {}) or {})},
        )
