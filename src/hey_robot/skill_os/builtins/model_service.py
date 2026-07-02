from __future__ import annotations

from hey_robot.skill_os.base import BaseSkill, SkillResult
from hey_robot.skill_os.builtins.common import spec


class VLAManipulationSkill(BaseSkill):
    spec = spec(
        "vla_manipulation",
        "Run the deployed VLA policy for a natural-language arm manipulation task.",
        category="manipulation",
        input_schema={
            "type": "object",
            "properties": {
                "task_prompt": {"type": "string"},
                "arm": {"type": "string"},
                "execution_time": {"type": "number"},
            },
            "required": ["task_prompt"],
        },
        required_resources=("arm", "gripper", "camera"),
        required_model_service="vla_manipulation",
        safety_level="motion",
        timeout_sec=30.0,
        feedback_mode="vision",
    )

    async def execute(self, ctx, arguments):
        if ctx.model_services is None:
            return SkillResult(
                success=False,
                summary="vla_manipulation requires a VLA model service",
                status="failed",
                failure_mode="model_service_unavailable",
                error="model service port is unavailable",
            )
        result = await ctx.model_services.call(self.spec.name, dict(arguments))
        return SkillResult(
            success=bool(result.success),
            summary=str(result.summary),
            status=result.status,
            failure_mode=result.failure_mode,
            error=result.error,
            data=dict(result.metrics),
        )
