"""Native VLA Skill adapter over the reusable bounded-option runner."""

from __future__ import annotations

from typing import Any

from hey_robot.skills.builtins.common import execute_robot_action
from hey_robot.skills.context import SkillContext
from hey_robot.skills.models import Skill, SkillResult
from hey_robot.skills.registry import SkillRegistry
from hey_robot.skills.vla import VLAOptionRequest, VLAOptionRunner

# Large remote policies can delay simulation observations, so observation
# freshness must tolerate inference scheduling jitter.
_DEFAULT_FRESH_OBSERVATION_TIMEOUT_SEC = 10.0
_DEFAULT_OPTION_STEPS = 128
_MAX_PUBLIC_STEPS = 600
# Long-horizon RoboCasa365 options can take more than half an hour.
_MAX_MANIPULATE_TIMEOUT_SEC = 3600.0

MANIPULATE_PARAMETERS = {
    "type": "object",
    "properties": {
        "task_prompt": {"type": "string", "minLength": 1},
        "max_steps": {
            "type": "integer",
            "minimum": 1,
            "maximum": _MAX_PUBLIC_STEPS,
            "default": _DEFAULT_OPTION_STEPS,
            "description": (
                "Bounded action-step budget for this semantic option. Use a larger "
                "budget when the single observable state change requires a longer "
                "continuous manipulation."
            ),
        },
    },
    "required": ["task_prompt"],
    "additionalProperties": False,
}


async def manipulate(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    task_prompt = str(arguments["task_prompt"])
    request = VLAOptionRequest(
        task_prompt=task_prompt,
        max_steps=int(arguments.get("max_steps", _DEFAULT_OPTION_STEPS)),
        fresh_observation_timeout_sec=_DEFAULT_FRESH_OBSERVATION_TIMEOUT_SEC,
    )
    execution = (await VLAOptionRunner().run(ctx, request)).to_skill_result()
    if not execution.success or execution.data.get("subgoal_succeeded") is True:
        return execution

    verification = await execute_robot_action(
        ctx,
        "inspect_scene",
        {
            "question": (
                "Is the complete end state requested by this subgoal visibly true now? "
                f"Subgoal: {task_prompt}"
            )
        },
    )
    verdict = str(verification.data.get("verification") or "unknown")
    if verdict not in {"yes", "no"}:
        verdict = "unknown"
    subgoal_succeeded = True if verdict == "yes" else False if verdict == "no" else None
    subgoal_status = (
        "achieved"
        if subgoal_succeeded is True
        else "not_achieved"
        if subgoal_succeeded is False
        else "unknown"
    )
    evidence = verification.data.get("visual_evidence")
    summary = f"{execution.summary} Postcondition verification={verdict}" + (
        f": {evidence}" if evidence else "."
    )
    return SkillResult(
        execution.success,
        summary,
        execution.status,
        data={
            **execution.data,
            "subgoal_succeeded": subgoal_succeeded,
            "subgoal_status": subgoal_status,
            "verification": verdict,
            "verification_target": task_prompt,
            "visual_evidence": evidence,
            "verification_frame_id": verification.data.get("frame_id"),
            "decision_state": {
                "execution_success": execution.data.get("execution_success"),
                "termination_reason": execution.data.get("termination_reason"),
                "subgoal_status": subgoal_status,
                "subgoal_succeeded": subgoal_succeeded,
                "verification": verdict,
                "verification_target": task_prompt,
                "visual_evidence": evidence,
            },
        },
        evidence_ids=execution.evidence_ids,
        observations=verification.observations,
        artifacts=execution.artifacts,
        failure_mode=execution.failure_mode,
        error=execution.error,
        observation_error=verification.observation_error,
    )


MANIPULATE = Skill(
    name="manipulate",
    description=(
        "Execute one bounded VLA subgoal and return visual verification of its complete "
        "postcondition."
    ),
    parameters=MANIPULATE_PARAMETERS,
    handler=manipulate,
    resources=("robot_control", "camera"),
    timeout_sec=_MAX_MANIPULATE_TIMEOUT_SEC,
    supported_robots=("xlerobot", "so101", "so101_mobile", "robocasa"),
    required_actions=("embodiment_native_action", "inspect_scene"),
    required_models=("manipulate",),
)


def register(registry: SkillRegistry) -> None:
    registry.register(MANIPULATE)
