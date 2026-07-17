"""Skill 与控制消息的持久化幂等回执。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def canonical_payload_hash(payload: dict[str, Any]) -> str:
    copy = _plain(asdict(payload) if is_dataclass(payload) else dict(payload))
    envelope = dict(copy.get("envelope") or {})
    envelope.pop("timestamp", None)
    copy["envelope"] = envelope
    return hashlib.sha256(
        json.dumps(copy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return _plain(asdict(value))  # type: ignore[arg-type]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


class SkillCommandStore:
    def __init__(self, path: str | Path) -> None:
        self._db = sqlite3.connect(str(path))
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS receipts (command_id TEXT PRIMARY KEY, payload_hash TEXT NOT NULL, state TEXT NOT NULL, terminal_result TEXT)"
        )
        self._db.commit()

    def receive(self, command_id: str, payload_hash: str) -> str:
        row = self._db.execute(
            "SELECT payload_hash, state FROM receipts WHERE command_id=?", (command_id,)
        ).fetchone()
        if row is None:
            self._db.execute(
                "INSERT INTO receipts VALUES (?, ?, 'active', NULL)",
                (command_id, payload_hash),
            )
            self._db.commit()
            return "new"
        return "replay" if row[0] == payload_hash else "conflict"

    def terminal(self, command_id: str, result: dict[str, Any]) -> None:
        self._db.execute(
            "UPDATE receipts SET state='terminal', terminal_result=? WHERE command_id=?",
            (json.dumps(result, sort_keys=True), command_id),
        )
        self._db.commit()

    def result(self, command_id: str) -> dict[str, Any] | None:
        row = self._db.execute(
            "SELECT terminal_result FROM receipts WHERE command_id=?", (command_id,)
        ).fetchone()
        return None if row is None or row[0] is None else json.loads(row[0])

    def is_active(self, command_id: str) -> bool:
        row = self._db.execute(
            "SELECT state FROM receipts WHERE command_id=?", (command_id,)
        ).fetchone()
        return row is not None and row[0] == "active"
