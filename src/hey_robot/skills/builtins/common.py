"""Shared helpers for native builtin skills."""

from __future__ import annotations

from typing import Any

from hey_robot.robot_api import RobotActionResult
from hey_robot.skills.context import SkillContext
from hey_robot.skills.models import SkillResult


async def execute_robot_action(
    ctx: SkillContext, action: str, arguments: dict[str, Any]
) -> SkillResult:
    if ctx.robot is None:
        raise RuntimeError("robot client is unavailable")
    result = await ctx.robot.execute(
        ctx.robot_id,
        action,
        arguments,
        run_id=ctx.run_id,
    )
    return result_to_skill(result)


def result_to_skill(result: RobotActionResult) -> SkillResult:
    return SkillResult(
        success=result.success,
        summary=result.summary,
        status="completed" if result.success else "failed",
        failure_mode=result.failure_mode,
        error=result.error,
        observation_error=result.observation_error,
        data={**dict(result.data), "frame_id": result.frame_id},
        observations=(
            tuple(result.observation.images) if result.observation is not None else ()
        ),
        artifacts=(
            tuple(result.observation.artifacts)
            if result.observation is not None
            else ()
        ),
    )
