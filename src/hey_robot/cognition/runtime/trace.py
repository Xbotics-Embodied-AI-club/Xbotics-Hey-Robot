"""Best-effort append-only trace for autonomous control decisions."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class RunTraceWriter:
    """Trace failures never change authoritative goal or action state."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        event: str,
        *,
        goal_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> bool:
        record = {
            "timestamp": time.time(),
            "event": event,
            "goal_id": goal_id,
            "details": dict(details or {}),
        }
        try:
            with self._path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                )
            return True
        except OSError:
            return False
