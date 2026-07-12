from __future__ import annotations

from typing import Literal

GoalStatus = Literal[
    "pending", "active", "waiting", "blocked", "completed", "failed", "cancelled"
]
ActionStatus = Literal[
    "persisted",
    "publishing",
    "published",
    "accepted",
    "running",
    "completed",
    "failed",
    "interrupted",
    "cancelled",
    "unknown",
    "reconciled_idle",
]

TERMINAL_GOAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
TERMINAL_ACTION_STATUSES = frozenset(
    {"completed", "failed", "interrupted", "cancelled", "unknown", "reconciled_idle"}
)
