"""One reusable observe-infer-act bounded option."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

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
        return SkillResult(
            self.success,
            self.summary,
            "completed" if self.success else "failed",
            data={
                "vla": dict(self.model_outputs[-1]) if self.model_outputs else {},
                "vla_history": [dict(item) for item in self.model_outputs],
                "steps": [dict(item) for item in self.executed_actions],
                "termination_reason": self.termination_reason,
                "option_completed": self.option_completed,
                "subgoal_succeeded": self.subgoal_succeeded,
                "before_frame_id": self.before_frame_id,
                "after_frame_id": self.after_frame_id,
                "steps_used": len(self.executed_actions),
            },
            evidence_ids=self.evidence_ids,
            failure_mode=self.failure_mode,
            error=self.error,
        )


class VLAOptionRunner:
    def __init__(self, termination: TerminationPolicy | None = None) -> None:
        self._termination = termination or CompositeTermination()

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
        after_frame_id: int | None = None
        executed_actions: list[dict[str, Any]] = []
        model_outputs: list[dict[str, Any]] = []

        for step_index in range(request.max_steps):
            context.raise_if_cancelled()
            result = await context.models.infer(
                "manipulate",
                {
                    "task_prompt": request.task_prompt,
                    "observation": _observation_payload(observation),
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
            actions = _actions_from_model_data(model_data)
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
                )

            environment_data: dict[str, Any] = {}
            for action in actions:
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
                decision = self._termination.evaluate(
                    VLAOptionState(
                        "after_actions",
                        step_index,
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
                    )

            await context.progress(
                (step_index + 1) / request.max_steps,
                f"VLA 已完成 bounded step {step_index + 1}/{request.max_steps}",
            )
            decision = self._termination.evaluate(
                VLAOptionState(
                    "after_actions",
                    step_index,
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
                )
            decision = self._termination.evaluate(
                VLAOptionState(
                    "budget",
                    step_index,
                    request.max_steps,
                    model_data,
                    environment_data,
                    len(actions),
                )
            )
            if decision.terminate:
                return self._decision_result(
                    decision,
                    f"VLA reached bounded limit ({request.max_steps} steps).",
                    request,
                    before_frame_id,
                    after_frame_id,
                    model_outputs,
                    executed_actions,
                )

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
    ) -> VLAOptionResult:
        return self._terminal(
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


def _observation_payload(observation: Any) -> dict[str, Any]:
    return {
        "frame_id": observation.frame_id,
        "timestamp": observation.envelope.timestamp,
        "images": [asdict(image) for image in observation.images],
        "proprioception": list(observation.proprioception),
        "raw": dict(observation.raw),
    }
