from __future__ import annotations

from typing import Any

from hey_robot.skill_os.termination.base import (
    TerminationDecision,
    TerminationState,
)


class FixedHorizonTerminationEvaluator:
    """Bound an option without claiming that the semantic task succeeded."""

    def __init__(self, horizon: int) -> None:
        if horizon < 1:
            raise ValueError("termination horizon must be positive")
        self.horizon = int(horizon)

    def evaluate(
        self,
        *,
        steps_executed: int,
        policy_result: dict[str, Any],
        action_result: dict[str, Any] | None,
    ) -> TerminationDecision:
        if not bool(policy_result.get("valid", True)):
            return TerminationDecision(
                TerminationState.FAILED,
                str(policy_result.get("failure_mode") or "invalid_policy_result"),
            )
        if bool(policy_result.get("done", False)):
            return TerminationDecision(
                TerminationState.SUCCESS,
                "policy_done",
                {"source": "public_policy_result"},
            )
        if action_result is not None and not bool(action_result.get("success", True)):
            return TerminationDecision(
                TerminationState.FAILED,
                str(action_result.get("error") or "action_failed"),
            )
        if action_result is not None and bool(action_result.get("done", False)):
            return TerminationDecision(
                TerminationState.UNKNOWN,
                "episode_terminal",
                {"source": "public_runtime_status"},
            )
        if steps_executed >= self.horizon:
            return TerminationDecision(
                TerminationState.UNKNOWN,
                "fixed_horizon_reached",
                {"horizon": self.horizon},
            )
        return TerminationDecision(TerminationState.CONTINUE, "within_horizon")
