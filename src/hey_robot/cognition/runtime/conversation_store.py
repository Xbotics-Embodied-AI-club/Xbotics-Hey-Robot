"""可持久化的 Conversation 专用记忆；不拥有物理状态权威。"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from hey_robot.providers import ReasoningMessage


class ConversationStore:
    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS messages (session_key TEXT NOT NULL, position INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, created_at REAL NOT NULL, PRIMARY KEY(session_key, position))"
        )
        self._db.commit()
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS goal_links (goal_id TEXT PRIMARY KEY, session_key TEXT NOT NULL, channel TEXT, chat_id TEXT, sender_id TEXT, user_id TEXT, agent_id TEXT, robot_id TEXT, episode_id TEXT, objective TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'pending', updated_at REAL NOT NULL DEFAULT 0)"
        )
        self._ensure_goal_link_columns()
        self._db.commit()

    def _ensure_goal_link_columns(self) -> None:
        """让已有运行时数据库兼容 Goal 投影字段。"""
        columns = {
            row[1]
            for row in self._db.execute("PRAGMA table_info(goal_links)").fetchall()
        }
        additions = {
            "objective": "TEXT NOT NULL DEFAULT ''",
            "status": "TEXT NOT NULL DEFAULT 'pending'",
            "updated_at": "REAL NOT NULL DEFAULT 0",
        }
        for name, definition in additions.items():
            if name not in columns:
                self._db.execute(
                    f"ALTER TABLE goal_links ADD COLUMN {name} {definition}"
                )

    def recent(self, session_key: str, limit: int = 16) -> list[ReasoningMessage]:
        rows = self._db.execute(
            "SELECT role, content FROM messages WHERE session_key=? ORDER BY position DESC LIMIT ?",
            (session_key, limit),
        ).fetchall()
        return [
            ReasoningMessage(role=role, content=content)
            for role, content in reversed(rows)
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

    def link_goal(
        self,
        goal_id: str,
        session_key: str,
        envelope: object,
        *,
        objective: str,
    ) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO goal_links (goal_id, session_key, channel, chat_id, sender_id, user_id, agent_id, robot_id, episode_id, objective, status, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                goal_id,
                session_key,
                getattr(envelope, "channel", None),
                getattr(envelope, "chat_id", None),
                getattr(envelope, "sender_id", None),
                getattr(envelope, "user_id", None),
                getattr(envelope, "agent_id", None),
                getattr(envelope, "robot_id", None),
                getattr(envelope, "episode_id", None),
                objective,
                "pending",
                time.time(),
            ),
        )
        self._db.commit()

    def update_goal_status(self, goal_id: str, status: str) -> None:
        self._db.execute(
            "UPDATE goal_links SET status=?, updated_at=? WHERE goal_id=?",
            (status, time.time(), goal_id),
        )
        self._db.commit()

    def active_goal(self, session_key: str) -> dict[str, str] | None:
        row = self._db.execute(
            "SELECT goal_id, objective, status FROM goal_links WHERE session_key=? AND status IN ('pending', 'active', 'waiting', 'waiting_condition', 'blocked', 'needs_review') ORDER BY updated_at DESC LIMIT 1",
            (session_key,),
        ).fetchone()
        if row is None:
            return None
        return {"goal_id": row[0], "objective": row[1], "status": row[2]}

    def goal_link(self, goal_id: str) -> tuple[str, dict[str, str | None]] | None:
        row = self._db.execute(
            "SELECT session_key, channel, chat_id, sender_id, user_id, agent_id, robot_id, episode_id FROM goal_links WHERE goal_id=?",
            (goal_id,),
        ).fetchone()
        if row is None:
            return None
        return row[0], dict(
            zip(
                (
                    "channel",
                    "chat_id",
                    "sender_id",
                    "user_id",
                    "agent_id",
                    "robot_id",
                    "episode_id",
                ),
                row[1:],
                strict=True,
            )
        )
