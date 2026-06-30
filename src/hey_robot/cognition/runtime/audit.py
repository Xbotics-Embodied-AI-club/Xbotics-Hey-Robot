from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ToolAuditRecord:
    tool_call_id: str
    tool: str
    arguments: dict[str, Any]
    status: str
    started_at: float
    ended_at: float
    duration_sec: float
    permission_behavior: str
    permission_reason: str
    capability_behavior: str = "allow"
    capability_reason: str = "not evaluated"
    capability_rule: str = "none"
    capability_source: str = ""
    capability_safety_level: str = ""
    result_preview: str = ""
    error: str | None = None
    task: str | None = None
    task_step: str | None = None


class ToolAuditLogger:
    def __init__(
        self, log_dir: str | Path = "logs", agent_run_id: str | None = None
    ) -> None:
        self.log_dir = Path(log_dir)
        self.agent_run_id = agent_run_id or str(int(time.time()))
        self.path = self.log_dir / "agent_runs" / self.agent_run_id / "tool_calls.jsonl"

    def write(self, record: ToolAuditRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
