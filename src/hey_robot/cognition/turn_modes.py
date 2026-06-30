from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TurnModeDecision:
    kind: str

    @property
    def is_direct(self) -> bool:
        return self.kind == "direct"


def decide_turn_mode(spec: Any, turn: Any) -> TurnModeDecision:
    del turn
    if str(getattr(spec, "settings", {}).get("mode", "agent")).lower() == "direct":
        return TurnModeDecision(kind="direct")
    return TurnModeDecision(kind="agent")
