"""Results of one bounded autonomous reasoning slice."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from hey_robot.protocol import ActionProposal, FailurePayload


@dataclass(frozen=True)
class AgentRunRequest:
    messages: tuple[object, ...]
    allowed_tools: frozenset[str]
    deadline: float
    run_id: str
    deliberation_id: str


@dataclass(frozen=True)
class ToolCallRecord:
    tool_call_id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class AgentRunResult:
    status: Literal["returned", "action_proposed", "failed"]
    final_text: str | None
    stop_reason: str
    tool_calls: tuple[ToolCallRecord, ...] = ()
    proposal: ActionProposal | None = None
    failure: FailurePayload | None = None
    usage: dict[str, int] = field(default_factory=dict)
