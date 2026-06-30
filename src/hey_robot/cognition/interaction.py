from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from hey_robot.interaction.intent import (
    UserInteractionIntent,
    classify_user_interaction,
)


@dataclass(frozen=True)
class PendingConfirmationDecision:
    action: str


def interpret_pending_confirmation_reply(
    payload: str | dict[str, Any] | None,
) -> PendingConfirmationDecision:
    if isinstance(payload, dict):
        raw_action = payload.get("action")
    else:
        text = str(payload or "").strip()
        if not text:
            return PendingConfirmationDecision(action="ignore")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return PendingConfirmationDecision(action="ignore")
        if not isinstance(parsed, dict):
            return PendingConfirmationDecision(action="ignore")
        raw_action = parsed.get("action")
    action = str(raw_action or "").strip().lower()
    if action in {"confirm", "decline", "new_task", "ignore"}:
        return PendingConfirmationDecision(action=action)
    return PendingConfirmationDecision(action="ignore")


__all__ = [
    "PendingConfirmationDecision",
    "UserInteractionIntent",
    "classify_user_interaction",
    "interpret_pending_confirmation_reply",
]
