"""Common option contract for embodied foundation-model workers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class OptionStatus(StrEnum):
    SUCCESS = "success"
    BUDGET = "budget"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class OptionRequest:
    session_id: str
    instruction: str
    max_actions: int
    reset_session: bool = False


@dataclass(frozen=True)
class OptionResult:
    status: OptionStatus
    actions_executed: int
    chunks_executed: int
    environment_done: bool
    progress: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
