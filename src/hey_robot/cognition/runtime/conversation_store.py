"""可持久化的 Conversation 专用记忆；不拥有物理状态权威。"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from hey_robot.model import ModelMessage


class ConversationStore:
    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS messages (session_key TEXT NOT NULL, position INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, created_at REAL NOT NULL, PRIMARY KEY(session_key, position))"
        )
        self._db.commit()

    def recent(self, session_key: str, limit: int = 16) -> list[ModelMessage]:
        rows = self._db.execute(
            "SELECT role, content FROM messages WHERE session_key=? ORDER BY position DESC LIMIT ?",
            (session_key, limit),
        ).fetchall()
        return [
            ModelMessage(role=role, content=content) for role, content in reversed(rows)
        ]

    def append(self, session_key: str, role: str, content: str) -> None:
        next_position = self._db.execute(
            "SELECT COALESCE(MAX(position), 0) + 1 FROM messages WHERE session_key=?",
            (session_key,),
        ).fetchone()[0]
        self._db.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
            (session_key, next_position, role, content, time.time()),
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()
