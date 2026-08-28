"""Native VLA Skill adapter over the reusable bounded-option runner."""

from __future__ import annotations

import logging
from typing import Any

from hey_robot.skills.builtins.common import execute_robot_action
from hey_robot.skills.builtins.robocasa_session import (
    consume_vla_desync,
)
from hey_robot.skills.context import SkillContext
from hey_robot.skills.models import Skill, SkillResult
from hey_robot.skills.registry import SkillRegistry
from hey_robot.skills.vla import VLAOptionRequest, VLAOptionRunner
from hey_robot.skills.vla.execution_memory import record_attempt

LOGGER = logging.getLogger(__name__)

# Large remote policies can delay simulation observations, so observation
# freshness must tolerate inference scheduling jitter.
_DEFAULT_FRESH_OBSERVATION_TIMEOUT_SEC = 10.0
_DEFAULT_OPTION_STEPS = 400
# A full contact-control window spans 70 chunks x 8 actions = 560 simulator
# steps and only stops on environment_done. Give the VLA the full budget to
# finish an object subgoal (grasp + transport + place) without cutting it
# short mid-manipulation.
_MAX_PUBLIC_STEPS = 560
# Long-horizon RoboCasa365 options can take more than half an hour.
_MAX_MANIPULATE_TIMEOUT_SEC = 3600.0

MANIPULATE_PARAMETERS = {
    "type": "object",
    "properties": {
        "task_prompt": {
            "type": "string",
            "minLength": 1,
            "description": (
                "One concise semantic physical outcome in English. Include known "
                "concrete object identity and source/target spatial relations from "
                "the latest observation so the instruction is grounded, but omit "
                "motion details and visual-verification requirements."
            ),
        },
        "max_steps": {
            "type": "integer",
            "minimum": 1,
            "maximum": _MAX_PUBLIC_STEPS,
            "default": _DEFAULT_OPTION_STEPS,
            "description": (
                "Bounded action-step budget for this semantic option (default "
                "400, max 560). One manipulate call should cover ONE object "
                "subgoal end to end (grasp + transport + place); the VLA stops "
                "only on environment_done. Verify with read_progress between "
                "calls."
            ),
        },
    },
    "required": ["task_prompt"],
    "additionalProperties": False,
}


async def manipulate(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    task_prompt = str(arguments["task_prompt"])
    requested_max_steps = int(arguments.get("max_steps", _DEFAULT_OPTION_STEPS))
    max_steps = await _effective_max_steps(ctx, task_prompt, requested_max_steps)
    uses_local_option = await _uses_local_foundation_option(ctx)
    if uses_local_option:
        # RPent force-resets its RLDX memory/history after any analytic action,
        # while consecutive VLA calls remain continuous.  Hey Robot's option
        # runner already resets on prompt/session changes; this supplies the
        # missing manual-action desynchronization signal.
        reset_session = consume_vla_desync(ctx.robot_id, ctx.task_id)
        execution = await execute_robot_action(
            ctx,
            "run_policy_option",
            {
                "session_id": ctx.task_id,
                "instruction": task_prompt,
                "max_actions": max_steps,
                "reset_session": reset_session,
            },
        )
        execution.data["requested_max_steps"] = requested_max_steps
        execution.data["effective_max_steps"] = max_steps
        option = dict(execution.data.get("option") or {})
        execution.data.update(
            {
                "execution_success": execution.success,
                "termination_reason": option.get("status"),
                "environment_done": bool(option.get("environment_done", False)),
                "vla_history": [option],
            }
        )
    else:
        execution = (
            await VLAOptionRunner().run(
                ctx,
                VLAOptionRequest(
                    task_prompt=task_prompt,
                    max_steps=max_steps,
                    fresh_observation_timeout_sec=_DEFAULT_FRESH_OBSERVATION_TIMEOUT_SEC,
                ),
            )
        ).to_skill_result()
    execution.data["requested_max_steps"] = requested_max_steps
    execution.data["effective_max_steps"] = max_steps
    attempt_diagnostics = _attempt_diagnostics(execution.data)
    execution.data["attempt_diagnostics"] = attempt_diagnostics
    after_diagnostics = dict(attempt_diagnostics.get("after") or {})
    await _record_harness_attempt(ctx, task_prompt, execution.data)
    if not execution.success or execution.data.get("subgoal_succeeded") is True:
        return execution

    # RoboCasa exposes the actual task predicates in the observation.  They
    # are strictly stronger than a VLM's image-only verdict and, crucially,
    # remain available when the captioner times out.  Let the planner recover
    # from this structured evidence rather than spending turns re-asking the
    # same visual question.
    progress = await _task_progress(ctx)
    if progress is not None:
        return SkillResult(
            execution.success,
            f"{execution.summary} task_progress={progress}",
            execution.status,
            data={
                **execution.data,
                "task_progress": progress,
                "verification": "structured_progress",
                "decision_state": {
                    "execution_success": execution.data.get("execution_success"),
                    "termination_reason": execution.data.get("termination_reason"),
                    "subgoal_status": execution.data.get("subgoal_status"),
                    "subgoal_succeeded": execution.data.get("subgoal_succeeded"),
                    "task_progress": progress,
                    "attempt_diagnostics": attempt_diagnostics,
                    "task_prompt": task_prompt,
                    "retry_prompt_must_match_exactly": bool(
                        execution.data.get("termination_reason") == "budget"
                    ),
                    # This signal is evidence, not a retry gate.  With the
                    # environment root task, XR-1 must retain its own
                    # closed-loop retries just as it does in the official
                    # evaluator.
                    "reposition_required": bool(
                        after_diagnostics.get("reposition_required")
                    ),
                    "required_next_action": None,
                },
            },
            evidence_ids=execution.evidence_ids,
            artifacts=execution.artifacts,
            failure_mode=execution.failure_mode,
            error=execution.error,
        )

    verification = await execute_robot_action(
        ctx,
        "inspect_scene",
        {
            "question": (
                "Is the primary current-state physical outcome of this subgoal visibly "
                "true now? Ignore historical preconditions, motion details, and "
                "nonessential fine-grained placement qualifiers that cannot be judged "
                "from the final image. "
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


async def _uses_local_foundation_option(ctx: SkillContext) -> bool:
    if ctx.robot is None:
        return False
    try:
        capabilities = await ctx.robot.capabilities(ctx.robot_id)
    except (AttributeError, NotImplementedError):
        return False
    return "run_policy_option" in {action.name for action in capabilities.actions}


async def _effective_max_steps(
    ctx: SkillContext, task_prompt: str, requested_max_steps: int
) -> int:
    """Preserve the direct-policy horizon for an unchanged root instruction.

    A full RLDX rollout uses 70 action chunks (560 simulator steps) for
    the environment's original task language.  The agent can otherwise choose
    a smaller horizon for genuine subgoals.  Comparing canonical strings at
    the skill boundary makes the root-task rule reliable instead of depending
    on the planner to follow a textual budget instruction.
    """
    requested = max(1, min(requested_max_steps, _MAX_PUBLIC_STEPS))
    try:
        observation = await ctx.observe(timeout_sec=10.0)
    except Exception:
        return requested
    policy_task = str((observation.raw or {}).get("policy_task") or "").strip()
    if policy_task and _canonical_task(task_prompt) == _canonical_task(policy_task):
        return _MAX_PUBLIC_STEPS
    return requested


def _canonical_task(task: str) -> str:
    return " ".join(
        "".join(char.lower() if char.isalnum() else " " for char in task).split()
    )


def _attempt_diagnostics(data: dict[str, Any]) -> dict[str, Any]:
    history = data.get("vla_history")
    if not isinstance(history, list):
        return {}
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        diagnostics = item.get("harness_diagnostics")
        if isinstance(diagnostics, dict):
            return dict(diagnostics)
        # The co-located RoboCasa option loop reports the same physical
        # evidence directly.  Normalize it to the before/after shape used by
        # the generic VLA runner so the planner sees one stable contract.
        diagnostics = item.get("diagnostics")
        if isinstance(diagnostics, dict):
            return {"after": dict(diagnostics)}
    return {}


async def _task_progress(ctx: SkillContext) -> dict[str, Any] | None:
    try:
        observation = await ctx.observe(timeout_sec=10.0)
    except Exception:
        return None
    progress = dict((observation.raw or {}).get("task_progress", {}) or {})
    return progress or None


async def _record_harness_attempt(
    ctx: SkillContext, task_prompt: str, data: dict[str, Any]
) -> None:
    """Best-effort persistence; telemetry must never fail physical execution."""
    try:
        observation = await ctx.observe(timeout_sec=10.0)
        diagnostics = dict(
            (observation.raw or {}).get("execution_diagnostics", {}) or {}
        )
        task = str(
            diagnostics.get("task")
            or observation.task
            or data.get("task")
            or (data.get("attempt_diagnostics", {}).get("before", {}) or {}).get("task")
            or "robocasa"
        )
        record_attempt(task=task, run_id=ctx.run_id, prompt=task_prompt, result=data)
    except Exception:
        LOGGER.warning("Could not persist VLA execution telemetry", exc_info=True)


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
