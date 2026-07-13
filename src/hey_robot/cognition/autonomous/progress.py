"""Pure, conservative progress assessment for sustained autonomous goals."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

ProgressState = Literal["progressing", "waiting", "blocked", "no_progress"]


@dataclass(frozen=True)
class ProgressAssessment:
    state: ProgressState
    reason: str
    repeated_action_count: int = 0
    evidence_count: int = 0


def assess_progress(
    *,
    goal_status: str,
    actions: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    repeat_limit: int = 2,
) -> ProgressAssessment:
    """Assess durable facts without making a scheduling or retry decision."""
    if goal_status == "blocked":
        return ProgressAssessment("blocked", "robot execution state is blocked")
    if goal_status in {"waiting", "waiting_condition"}:
        return ProgressAssessment("waiting", f"goal is {goal_status}")
    if not actions:
        return ProgressAssessment("progressing", "no action history yet")

    signatures = [_action_signature(item) for item in actions]
    latest = signatures[-1]
    repeated = 0
    for signature in reversed(signatures):
        if signature != latest:
            break
        repeated += 1
    if repeated >= max(2, repeat_limit) and not evidence:
        return ProgressAssessment(
            "no_progress",
            "repeated equivalent actions produced no trusted evidence",
            repeated_action_count=repeated,
            evidence_count=0,
        )
    return ProgressAssessment(
        "progressing",
        "recent action history has not crossed the no-progress threshold",
        repeated_action_count=repeated,
        evidence_count=len(evidence),
    )


def _action_signature(action: dict[str, Any]) -> str:
    raw_payload = action.get("payload")
    payload: dict[str, Any] = raw_payload if isinstance(raw_payload, dict) else {}
    # An inspect_scene question is prompt context, not a distinct physical
    # operation.  Treat reworded scene observations as the same action so an
    # automatic re-observe cannot evade the no-progress circuit breaker.
    arguments = payload.get("arguments")
    if (
        payload.get("intent_kind") == "observation"
        and payload.get("name") == "inspect_scene"
    ):
        arguments = {}
    value = {
        "intent_kind": payload.get("intent_kind"),
        "name": payload.get("name"),
        "arguments": arguments,
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
