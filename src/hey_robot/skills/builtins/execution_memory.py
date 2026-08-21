"""Agent-facing retrieval of verified policy execution attempts."""

from __future__ import annotations

from typing import Any

from hey_robot.skills.context import SkillContext
from hey_robot.skills.models import Skill, SkillResult
from hey_robot.skills.registry import SkillRegistry
from hey_robot.skills.vla.execution_memory import retrieve_successful_attempts


async def read_execution_memory(
    ctx: SkillContext, arguments: dict[str, Any]
) -> SkillResult:
    # The agent often supplies the natural-language goal here.  Memory is
    # keyed by the stable RoboCasa task id instead, so reference runs survive
    # paraphrases and retain the same seed-to-held-out binding.
    observation = await ctx.observe(timeout_sec=10.0)
    raw = dict(observation.raw or {})
    diagnostics = dict(raw.get("execution_diagnostics") or {})
    task = str(
        diagnostics.get("task") or observation.task or arguments.get("task") or ""
    ).strip()
    if not task:
        return SkillResult(
            False, "No active RoboCasa task for memory lookup.", "failed"
        )
    attempts = retrieve_successful_attempts(task)
    return SkillResult(
        True,
        f"Retrieved {len(attempts)} verified policy attempts for {task}.",
        "completed",
        data={"task": task, "verified_attempts": attempts},
    )


READ_EXECUTION_MEMORY = Skill(
    name="read_execution_memory",
    description=(
        "Retrieve verified symbolic reference attempts for the active RoboCasa task. "
        "Use their primitive ordering and prompt style as a prior, then re-ground "
        "all spatial arguments from the live observation."
    ),
    parameters={
        "type": "object",
        "properties": {"task": {"type": "string"}},
        "additionalProperties": False,
    },
    handler=read_execution_memory,
    resources=("camera",),
    timeout_sec=15.0,
    supported_robots=("robocasa",),
)


def register(registry: SkillRegistry) -> None:
    registry.register(READ_EXECUTION_MEMORY)
