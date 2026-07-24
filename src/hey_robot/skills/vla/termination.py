"""Explicit termination policies for one bounded VLA option."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

TerminationStage = Literal["after_model", "after_actions", "budget"]


@dataclass(frozen=True)
class VLAOptionState:
    stage: TerminationStage
    step_index: int
    max_steps: int
    model_data: dict[str, Any]
    environment_data: dict[str, Any]
    action_count: int


@dataclass(frozen=True)
class TerminationDecision:
    terminate: bool
    reason: str | None = None
    subgoal_succeeded: bool | None = None


class TerminationPolicy(Protocol):
    def evaluate(self, state: VLAOptionState) -> TerminationDecision: ...


class EnvironmentDoneTermination:
    def evaluate(self, state: VLAOptionState) -> TerminationDecision:
        done = _environment_done(state.model_data) or _environment_done(
            state.environment_data
        )
        return TerminationDecision(done, "environment_done" if done else None, True)


class ModelDoneTermination:
    def evaluate(self, state: VLAOptionState) -> TerminationDecision:
        done = _model_done(state.model_data) and (
            state.stage == "after_actions"
            or (state.stage == "after_model" and state.action_count == 0)
        )
        return TerminationDecision(done, "model_done" if done else None, True)


class NoActionTermination:
    def evaluate(self, state: VLAOptionState) -> TerminationDecision:
        done = state.stage == "after_model" and state.action_count == 0
        return TerminationDecision(done, "no_action" if done else None, None)


class BudgetTermination:
    def evaluate(self, state: VLAOptionState) -> TerminationDecision:
        done = state.stage == "budget" and state.step_index + 1 >= state.max_steps
        return TerminationDecision(done, "max_steps" if done else None, None)


class CompositeTermination:
    def __init__(self, policies: tuple[TerminationPolicy, ...] | None = None) -> None:
        self._policies = policies or (
            EnvironmentDoneTermination(),
            ModelDoneTermination(),
            NoActionTermination(),
            BudgetTermination(),
        )

    def evaluate(self, state: VLAOptionState) -> TerminationDecision:
        for policy in self._policies:
            decision = policy.evaluate(state)
            if decision.terminate:
                return decision
        return TerminationDecision(False)


def _model_done(data: dict[str, Any]) -> bool:
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
