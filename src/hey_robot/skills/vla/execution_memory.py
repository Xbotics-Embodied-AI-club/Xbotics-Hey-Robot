"""Append-only, auditable execution memory for policy attempts.

The store intentionally saves symbolic prompts and physical outcomes, never
camera pixels or absolute object coordinates.  A successful reference rollout
can therefore be re-grounded on a held-out seed instead of being replayed
open-loop.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any


def record_attempt(
    *,
    task: str,
    run_id: str,
    prompt: str,
    result: dict[str, Any],
) -> Path:
    root = Path(
        os.environ.get(
            "HEY_ROBOT_EXECUTION_MEMORY_DIR",
            "runtime/robocasa365.rldx/execution_memory",
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    filename = _safe_name(task) + ".jsonl"
    path = root / filename
    record = {
        "schema": "policy-attempt/v1",
        "timestamp": time.time(),
        "task": task,
        "run_id": run_id,
        "primitive": "vla_act",
        "prompt": prompt,
        "termination_reason": result.get("termination_reason"),
        "subgoal_succeeded": result.get("subgoal_succeeded"),
        "steps_used": result.get("steps_used"),
        "attempt_diagnostics": result.get("attempt_diagnostics", {}),
    }
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def retrieve_successful_attempts(task: str, *, limit: int = 12) -> list[dict[str, Any]]:
    """Load verified reference attempts for the current task, newest first."""
    root = Path(
        os.environ.get(
            "HEY_ROBOT_EXECUTION_MEMORY_DIR",
            "runtime/robocasa365.rldx/execution_memory",
        )
    )
    path = root / (_safe_name(task) + ".jsonl")
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if isinstance(row, dict) and row.get("subgoal_succeeded") is True:
                records.append(row)
    except (OSError, json.JSONDecodeError):
        return []
    return records[-max(1, limit) :][::-1]


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned[:120] or "unknown_task"
