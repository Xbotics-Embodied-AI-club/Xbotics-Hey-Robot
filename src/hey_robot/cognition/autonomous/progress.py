"""用于持续自主 Goal 的纯函数、保守进度评估。"""

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
    """评估持久化事实，但不作调度或重试决定。"""
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
    # inspect_scene 的 question 属于 prompt 上下文，不是不同的物理操作。
    # 将改写后的场景观测视为同一动作，避免自动重新观测绕过无进展断路器。
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
