"""One reusable observe-infer-act bounded option."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from hey_robot.skills.context import SkillContext
from hey_robot.skills.models import SkillResult
from hey_robot.skills.vla.termination import (
    CompositeTermination,
    TerminationDecision,
    TerminationPolicy,
    VLAOptionState,
)


@dataclass(frozen=True)
class VLAOptionRequest:
    task_prompt: str
    max_steps: int
    fresh_observation_timeout_sec: float = 2.0
    model_timeout_sec: float | None = None


@dataclass(frozen=True)
class VLAOptionResult:
    success: bool
    summary: str
    option_completed: bool
    subgoal_succeeded: bool | None
    termination_reason: str
    before_frame_id: int | None
    after_frame_id: int | None
    model_outputs: tuple[dict[str, Any], ...]
    executed_actions: tuple[dict[str, Any], ...]
    evidence_ids: tuple[str, ...] = ()
    failure_mode: str | None = None
    error: str | None = None

    def to_skill_result(self) -> SkillResult:
        subgoal_status = (
            "achieved"
            if self.subgoal_succeeded is True
            else "not_achieved"
            if self.subgoal_succeeded is False
            else "unknown"
        )
        return SkillResult(
            self.success,
            self.summary,
            "completed" if self.success else "failed",
            data={
                "vla": dict(self.model_outputs[-1]) if self.model_outputs else {},
                "vla_history": [dict(item) for item in self.model_outputs],
                "steps": [dict(item) for item in self.executed_actions],
                "termination_reason": self.termination_reason,
                "execution_success": self.success,
                "option_completed": self.option_completed,
                "subgoal_succeeded": self.subgoal_succeeded,
                "subgoal_status": subgoal_status,
                "before_frame_id": self.before_frame_id,
                "after_frame_id": self.after_frame_id,
                "steps_used": len(self.executed_actions),
                "decision_state": {
                    "execution_success": self.success,
                    "termination_reason": self.termination_reason,
                    "subgoal_status": subgoal_status,
                    "subgoal_succeeded": self.subgoal_succeeded,
                },
            },
            evidence_ids=self.evidence_ids,
            failure_mode=self.failure_mode,
            error=self.error,
        )


class VLAOptionRunner:
    def __init__(self, termination: TerminationPolicy | None = None) -> None:
        self._termination = termination or CompositeTermination()
        # Settle (idle) detection stops a policy call when the
        # eef+gripper stop changing for `settle_patience` chunks — the policy is
        # finished and burning more budget only wastes time. Track per-run
        # proprioception deltas here (fresh per run() call).
        self._settle_eps = 0.012
        self._settle_grip_eps = 0.003
        # A large default patience (~never settles) lets the
        # VLA keeps executing until environment_done or the max_steps budget;
        # an aggressive settle window cut the VLA off mid-grasp (observed:
        # "robot holds broccoli but placement not finished"). Keep the
        # detection available but effectively disabled by default.
        self._settle_patience = 999  # consecutive near-zero-delta steps
        self._settle_streak = 0
        self._prev_eef: np.ndarray | None = None
        self._prev_base: np.ndarray | None = None
        self._prev_grip: float | None = None

    def _settle_signal(self, observation: Any) -> bool:
        """Return True when the robot has idled (eef+base+gripper frozen)."""
        p = np.asarray(
            getattr(observation, "proprioception", None) or [], dtype=np.float64
        )
        if p.shape != (16,):
            return False
        eef = p[0:3]  # eef position relative to base
        base = p[7:10]  # base world position
        grip = float(p[14])
        if (
            self._prev_eef is not None
            and self._prev_base is not None
            and self._prev_grip is not None
        ):
            moved = float(np.linalg.norm(eef - self._prev_eef)) + float(
                np.linalg.norm(base - self._prev_base)
            )
            grip_delta = abs(grip - self._prev_grip)
            if moved < self._settle_eps and grip_delta < self._settle_grip_eps:
                self._settle_streak += 1
            else:
                self._settle_streak = 0
        self._prev_eef = eef
        self._prev_base = base
        self._prev_grip = grip
        return self._settle_streak >= self._settle_patience

    async def run(
        self, context: SkillContext, request: VLAOptionRequest
    ) -> VLAOptionResult:
        if context.models is None:
            return self._terminal(
                success=False,
                summary="VLA model router is unavailable.",
                request=request,
                termination_reason="model_unavailable",
                failure_mode="model_service_unavailable",
                error="model router is unavailable",
            )
        if context.robot is None:
            return self._terminal(
                success=False,
                summary="robot client is unavailable for VLA action execution",
                request=request,
                termination_reason="robot_unavailable",
                failure_mode="robot_client_unavailable",
                error="robot client is unavailable",
            )
        try:
            observation = await context.observe(
                timeout_sec=request.fresh_observation_timeout_sec
            )
        except TimeoutError:
            return self._terminal(
                success=False,
                summary="VLA 执行前未获得 fresh observation。",
                request=request,
                termination_reason="observation_stale",
                failure_mode="observation_stale",
            )

        before_frame_id = observation.frame_id
        before_diagnostics = _execution_diagnostics(observation)
        after_frame_id: int | None = None
        executed_actions: list[dict[str, Any]] = []
        model_outputs: list[dict[str, Any]] = []
        # RLDX conditions its next action chunk on frames from every simulator
        # action, not solely the last frame of the preceding chunk.
        pending_video_history: list[dict[str, Any]] = []

        steps_used = 0
        while steps_used < request.max_steps:
            # A model response may contain a full RLDX action chunk.  The
            # budget remains simulator actions, not model invocations.
            step_index = steps_used
            context.raise_if_cancelled()
            result = await context.models.infer(
                "manipulate",
                {
                    "task_prompt": request.task_prompt,
                    "observation": _observation_payload(observation),
                    "observation_history": list(pending_video_history),
                    "policy_session_id": context.run_id,
                    "step_index": step_index,
                    "max_steps": request.max_steps,
                },
                run_id=context.run_id,
                robot_id=context.robot_id,
                timeout_sec=request.model_timeout_sec,
            )
            context.raise_if_cancelled()
            if not result.success:
                return self._terminal(
                    success=False,
                    summary=result.summary,
                    request=request,
                    termination_reason="model_failed",
                    before_frame_id=before_frame_id,
                    after_frame_id=after_frame_id,
                    model_outputs=model_outputs,
                    executed_actions=executed_actions,
                    failure_mode=result.failure_mode or "model_failed",
                    error=result.error,
                )

            model_data = dict(result.data)
            model_outputs.append(model_data)
            pending_video_history.clear()
            actions = _actions_from_model_data(model_data)[
                : request.max_steps - steps_used
            ]
            decision = self._termination.evaluate(
                VLAOptionState(
                    "after_model",
                    step_index,
                    request.max_steps,
                    model_data,
                    {},
                    len(actions),
                )
            )
            if decision.terminate:
                return self._decision_result(
                    decision,
                    result.summary,
                    request,
                    before_frame_id,
                    after_frame_id,
                    model_outputs,
                    executed_actions,
                    before_diagnostics=before_diagnostics,
                    after_diagnostics=_execution_diagnostics(observation),
                )

            environment_data: dict[str, Any] = {}
            action_batches = _coalesce_native_action_chunk(actions)
            for action, applied_count in action_batches:
                context.raise_if_cancelled()
                action_result = await context.robot.execute(
                    context.robot_id,
                    action["name"],
                    action["arguments"],
                    run_id=context.run_id,
                    expected_frame_id=observation.frame_id,
                )
                environment_data = dict(action_result.data)
                executed_actions.append(
                    {
                        "step_index": step_index,
                        "action": action,
                        "success": action_result.success,
                        "summary": action_result.summary,
                        "data": environment_data,
                        "frame_id": action_result.frame_id,
                    }
                )
                if not action_result.success:
                    return self._terminal(
                        success=False,
                        summary=action_result.summary,
                        request=request,
                        termination_reason="action_failed",
                        before_frame_id=before_frame_id,
                        after_frame_id=after_frame_id,
                        model_outputs=model_outputs,
                        executed_actions=executed_actions,
                        failure_mode=action_result.failure_mode,
                        error=action_result.error,
                        subgoal_succeeded=False,
                    )
                after_frame_id = action_result.frame_id
                steps_used += applied_count
                if action_result.observation is not None:
                    # Subsequent actions in this chunk must use the frame
                    # produced by the preceding action, not the stale chunk
                    # input frame.
                    observation = action_result.observation
                    pending_video_history.append(
                        _observation_payload(action_result.observation)
                    )
                decision = self._termination.evaluate(
                    VLAOptionState(
                        "after_actions",
                        steps_used - 1,
                        request.max_steps,
                        model_data,
                        environment_data,
                        len(actions),
                    )
                )
                if decision.terminate and decision.reason == "environment_done":
                    return self._decision_result(
                        decision,
                        action_result.summary,
                        request,
                        before_frame_id,
                        after_frame_id,
                        model_outputs,
                        executed_actions,
                        before_diagnostics=before_diagnostics,
                        after_diagnostics=_execution_diagnostics(
                            action_result.observation
                        ),
                    )

            await context.progress(
                steps_used / request.max_steps,
                f"VLA 已完成 bounded step {steps_used}/{request.max_steps}",
            )
            decision = self._termination.evaluate(
                VLAOptionState(
                    "after_actions",
                    steps_used - 1,
                    request.max_steps,
                    model_data,
                    environment_data,
                    len(actions),
                )
            )
            if decision.terminate:
                return self._decision_result(
                    decision,
                    result.summary,
                    request,
                    before_frame_id,
                    after_frame_id,
                    model_outputs,
                    executed_actions,
                    before_diagnostics=before_diagnostics,
                    after_diagnostics=_execution_diagnostics(observation),
                )
            decision = self._termination.evaluate(
                VLAOptionState(
                    "budget",
                    steps_used - 1,
                    request.max_steps,
                    model_data,
                    environment_data,
                    len(actions),
                )
            )
            if decision.terminate:
                return self._decision_result(
                    decision,
                    (
                        "VLA execution window ended after "
                        f"{request.max_steps} steps; subgoal completion is unverified."
                    ),
                    request,
                    before_frame_id,
                    after_frame_id,
                    model_outputs,
                    executed_actions,
                    before_diagnostics=before_diagnostics,
                    after_diagnostics=_execution_diagnostics(observation),
                )

            # RoboCasa Step returns the resulting observation with each action,
            # so the next model call can begin immediately.  Keep the generic
            # freshness fallback for robots that do not return one.
            if after_frame_id is None or observation.frame_id != after_frame_id:
                try:
                    observation = await context.observe(
                        after_frame_id=observation.frame_id,
                        timeout_sec=request.fresh_observation_timeout_sec,
                    )
                except TimeoutError:
                    return self._terminal(
                        success=False,
                        summary="VLA action 后未获得 fresh observation。",
                        request=request,
                        termination_reason="observation_stale",
                        before_frame_id=before_frame_id,
                        after_frame_id=after_frame_id,
                        model_outputs=model_outputs,
                        executed_actions=executed_actions,
                        failure_mode="observation_stale",
                    )

            # If the robot has idled for
            # `settle_patience` consecutive steps, the VLA is done moving —
            # return control to the agent instead of burning the budget.
            if self._settle_signal(observation):
                return self._terminal(
                    success=True,
                    summary=(
                        f"VLA execution idled after {step_index + 1} steps "
                        "(eef/base/gripper frozen); subgoal completion is unverified."
                    ),
                    request=request,
                    termination_reason="settled",
                    before_frame_id=before_frame_id,
                    after_frame_id=after_frame_id,
                    model_outputs=model_outputs,
                    executed_actions=executed_actions,
                    subgoal_succeeded=None,
                )

        raise RuntimeError("VLA option loop exhausted without a termination decision")

    def _decision_result(
        self,
        decision: TerminationDecision,
        summary: str,
        request: VLAOptionRequest,
        before_frame_id: int,
        after_frame_id: int | None,
        model_outputs: list[dict[str, Any]],
        executed_actions: list[dict[str, Any]],
        *,
        before_diagnostics: dict[str, Any] | None = None,
        after_diagnostics: dict[str, Any] | None = None,
    ) -> VLAOptionResult:
        result = self._terminal(
            success=True,
            summary=summary,
            request=request,
            termination_reason=decision.reason or "unknown",
            before_frame_id=before_frame_id,
            after_frame_id=after_frame_id,
            model_outputs=model_outputs,
            executed_actions=executed_actions,
            subgoal_succeeded=decision.subgoal_succeeded,
        )
        return _with_diagnostics(result, before_diagnostics, after_diagnostics)

    @staticmethod
    def _terminal(
        *,
        success: bool,
        summary: str,
        request: VLAOptionRequest,
        termination_reason: str,
        before_frame_id: int | None = None,
        after_frame_id: int | None = None,
        model_outputs: list[dict[str, Any]] | None = None,
        executed_actions: list[dict[str, Any]] | None = None,
        failure_mode: str | None = None,
        error: str | None = None,
        subgoal_succeeded: bool | None = None,
    ) -> VLAOptionResult:
        del request
        return VLAOptionResult(
            success=success,
            summary=summary,
            option_completed=True,
            subgoal_succeeded=subgoal_succeeded,
            termination_reason=termination_reason,
            before_frame_id=before_frame_id,
            after_frame_id=after_frame_id,
            model_outputs=tuple(model_outputs or ()),
            executed_actions=tuple(executed_actions or ()),
            failure_mode=failure_mode,
            error=error,
        )


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


def _coalesce_native_action_chunk(
    actions: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], int]]:
    """Send one RLDX native action chunk through the existing Step RPC.

    Other robot primitives retain their one-action semantics.  The remote
    RoboCasa driver recognizes nested ``values`` and advances the block under
    a single runtime lock/RPC.
    """
    if len(actions) <= 1 or any(
        action.get("name") != "embodiment_native_action" for action in actions
    ):
        return [(action, 1) for action in actions]
    arguments = [dict(action.get("arguments") or {}) for action in actions]
    values = [item.get("values") for item in arguments]
    raw_values = [item.get("raw_values") or item.get("values") for item in arguments]
    if not all(isinstance(item, list) and len(item) == 12 for item in values):
        return [(action, 1) for action in actions]
    return [
        (
            {
                "name": "embodiment_native_action",
                "arguments": {
                    **arguments[0],
                    "values": values,
                    "raw_values": raw_values,
                },
            },
            len(actions),
        )
    ]


def _observation_payload(observation: Any) -> dict[str, Any]:
    return {
        "frame_id": observation.frame_id,
        "timestamp": observation.envelope.timestamp,
        "images": [asdict(image) for image in observation.images],
        "proprioception": list(observation.proprioception),
        "raw": dict(observation.raw),
    }


def _execution_diagnostics(observation: Any | None) -> dict[str, Any]:
    """Extract backend-provided physical diagnostics without guessing values."""
    if observation is None:
        return {}
    raw = getattr(observation, "raw", None)
    if not isinstance(raw, dict):
        return {}
    diagnostics = raw.get("execution_diagnostics")
    return dict(diagnostics) if isinstance(diagnostics, dict) else {}


def _with_diagnostics(
    result: VLAOptionResult,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> VLAOptionResult:
    """Attach attempt evidence while preserving the immutable result contract."""
    if not before and not after:
        return result
    outputs = list(result.model_outputs)
    outputs.append(
        {"harness_diagnostics": {"before": before or {}, "after": after or {}}}
    )
    return VLAOptionResult(**{**result.__dict__, "model_outputs": tuple(outputs)})
