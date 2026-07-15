"""Durable conversation-only memory; never an authority for physical state."""

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
            "CREATE TABLE IF NOT EXISTS goal_links (goal_id TEXT PRIMARY KEY, session_key TEXT NOT NULL, channel TEXT, chat_id TEXT, sender_id TEXT, user_id TEXT, agent_id TEXT, robot_id TEXT, episode_id TEXT)"
        )
        self._db.commit()

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

    def link_goal(self, goal_id: str, session_key: str, envelope: object) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO goal_links VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            ),
        )
        self._db.commit()

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
