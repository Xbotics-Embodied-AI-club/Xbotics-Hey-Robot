"""持久化回执存储；中断的模型工作绝不恢复。"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from hey_robot.protocol.messages import (
    DeliberationRequest,
    DeliberationResult,
    from_payload,
    to_payload,
)


class DeliberationStore:
    def __init__(self, path: str | Path) -> None:
        self._db = sqlite3.connect(str(path))
        self._db.execute("""CREATE TABLE IF NOT EXISTS deliberations (
          deliberation_id TEXT PRIMARY KEY, request_hash TEXT NOT NULL, goal_id TEXT NOT NULL,
          task_id TEXT NOT NULL, phase TEXT NOT NULL, terminal_result TEXT, terminal_hash TEXT,
          updated_at REAL NOT NULL, request_payload TEXT)""")
        columns = {
            row[1] for row in self._db.execute("PRAGMA table_info(deliberations)")
        }
        if "request_payload" not in columns:
            self._db.execute(
                "ALTER TABLE deliberations ADD COLUMN request_payload TEXT"
            )
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def schedule(
        self,
        *,
        deliberation_id: str,
        request_hash: str,
        goal_id: str,
        task_id: str,
        request: DeliberationRequest,
    ) -> bool:
        try:
            self._db.execute(
                "INSERT INTO deliberations (deliberation_id, request_hash, goal_id, task_id, phase, terminal_result, terminal_hash, updated_at, request_payload) VALUES (?, ?, ?, ?, 'SCHEDULED', NULL, NULL, ?, ?)",
                (
                    deliberation_id,
                    request_hash,
                    goal_id,
                    task_id,
                    time.time(),
                    json.dumps(
                        to_payload(request), sort_keys=True, separators=(",", ":")
                    ),
                ),
            )
            self._db.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def transition(self, deliberation_id: str, expected: str, phase: str) -> bool:
        cursor = self._db.execute(
            "UPDATE deliberations SET phase=?, updated_at=? WHERE deliberation_id=? AND phase=?",
            (phase, time.time(), deliberation_id, expected),
        )
        self._db.commit()
        return cursor.rowcount == 1

    def terminal(self, result: DeliberationResult, result_hash: str) -> bool:
        encoded = json.dumps(to_payload(result), sort_keys=True, separators=(",", ":"))
        cursor = self._db.execute(
            "UPDATE deliberations SET phase='TERMINAL', terminal_result=?, terminal_hash=?, updated_at=? WHERE deliberation_id=? AND phase != 'TERMINAL'",
            (encoded, result_hash, time.time(), result.deliberation_id),
        )
        self._db.commit()
        return cursor.rowcount == 1

    def result(self, deliberation_id: str) -> DeliberationResult | None:
        row = self._db.execute(
            "SELECT terminal_result FROM deliberations WHERE deliberation_id=?",
            (deliberation_id,),
        ).fetchone()
        return (
            None
            if row is None or row[0] is None
            else from_payload(DeliberationResult, json.loads(row[0]))
        )

    def interrupt_incomplete(self) -> list[DeliberationRequest]:
        rows = self._db.execute(
            "SELECT request_payload FROM deliberations WHERE phase != 'TERMINAL'"
        ).fetchall()
        return [
            from_payload(DeliberationRequest, json.loads(row[0]))
            for row in rows
            if row[0]
        ]
