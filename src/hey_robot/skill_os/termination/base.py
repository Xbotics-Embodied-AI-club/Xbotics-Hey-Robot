from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class TerminationState(StrEnum):
    SUCCESS = "success"
    CONTINUE = "continue"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TerminationDecision:
    state: TerminationState
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)


class TerminationEvaluator(Protocol):
    def evaluate(
        self,
        *,
        steps_executed: int,
        policy_result: dict[str, Any],
        action_result: dict[str, Any] | None,
    ) -> TerminationDecision: ...
