"""自主控制决策的尽力而为、仅追加追踪记录。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class RunTraceWriter:
    """追踪失败绝不改变权威 Goal 或动作状态。"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        event: str,
        *,
        task_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> bool:
        record = {
            "timestamp": time.time(),
            "event": event,
            "task_id": task_id,
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
