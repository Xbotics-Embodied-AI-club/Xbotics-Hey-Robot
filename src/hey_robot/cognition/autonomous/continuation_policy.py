"""Pure continuation policy; it never publishes or retries an action."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from hey_robot.cognition.autonomous.progress import ProgressAssessment

ContinuationDecision = Literal[
    "schedule_deliberation", "wait", "needs_review", "fail", "block", "stop"
]


@dataclass(frozen=True)
class ContinuationAssessment:
    decision: ContinuationDecision
    reason: str


def decide_continuation(
    *,
    goal_status: str,
    progress: ProgressAssessment,
    budget_allowed: bool,
    gate_ready: bool,
    has_wake_trigger: bool,
) -> ContinuationAssessment:
    """Return an explainable next-state recommendation from durable inputs."""
    if goal_status in {"completed", "failed", "cancelled"}:
        return ContinuationAssessment("stop", "goal is terminal")
    if not gate_ready:
        return ContinuationAssessment("block", "robot execution gate is not ready")
    if not budget_allowed:
        return ContinuationAssessment("fail", "goal budget is exhausted")
    if goal_status in {"waiting", "waiting_condition"}:
        return ContinuationAssessment("wait", f"goal is {goal_status}")
    if progress.state == "no_progress":
        return ContinuationAssessment("needs_review", progress.reason)
    if has_wake_trigger and goal_status in {"pending", "active"}:
        return ContinuationAssessment(
            "schedule_deliberation", "trusted wake trigger received"
        )
    return ContinuationAssessment("wait", "no trusted wake trigger")
