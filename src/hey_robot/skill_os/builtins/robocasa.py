from __future__ import annotations

from typing import Any

from hey_robot.skill_os.base import BaseSkill, SkillResult
from hey_robot.skill_os.builtins.common import spec


class RoboCasaOptionSkill(BaseSkill):
    """Run one bounded RoboCasa option with the live VLA policy."""

    spec = spec(
        "robocasa_option",
        (
            "Run one bounded chunk of the frozen RoboCasa root task with the "
            "external VLA policy. option_command is an Agent orchestration note; "
            "the worker preserves the checkpoint's native root-task prompt contract."
        ),
        category="benchmark",
        input_schema={
            "type": "object",
            "properties": {
                "option_command": {
                    "type": "string",
                    "description": "A plain natural-language reason for continuing the root task.",
                },
            },
            "required": ["option_command"],
            "additionalProperties": False,
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
            "trial_unavailable",
            "episode_terminal",
            "stale_action",
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

        payload = {"option_command": str(arguments["option_command"])}
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
