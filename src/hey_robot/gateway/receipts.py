"""用于入站交互幂等处理的 Gateway 持久化回执。"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path


class InteractionReceiptStore:
    """在路由前仅领取一次每个渠道交互。

    此存储刻意按上游交互标识去重，而不是按消息文本去重：重复说出或输入相同
    请求，仍可能是一次合法的新命令。
    """

    def __init__(self, path: str | Path) -> None:
        location = Path(path)
        location.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(location))
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS interaction_receipts (
                interaction_id TEXT PRIMARY KEY,
                payload_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                result_kind TEXT,
                created_at REAL NOT NULL,
                completed_at REAL
            )"""
        )
        self._db.commit()

    def claim(self, interaction_id: str, payload_hash: str) -> bool:
        """在产生副作用前持久化回执；重放请求返回 ``False``。"""
        try:
            with self._db:
                self._db.execute(
                    "INSERT INTO interaction_receipts VALUES (?, ?, 'processing', NULL, ?, NULL)",
                    (interaction_id, payload_hash, time.time()),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def complete(self, interaction_id: str, result_kind: str) -> None:
        with self._db:
            self._db.execute(
                "UPDATE interaction_receipts SET status='completed', result_kind=?, completed_at=? "
                "WHERE interaction_id=?",
                (result_kind, time.time(), interaction_id),
            )

    def close(self) -> None:
        self._db.close()


class GoalNotificationReceiptStore:
    """由 Gateway 管理的面向用户 Goal 通知投递回执。"""

    def __init__(self, path: str | Path) -> None:
        location = Path(path)
        location.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(location))
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS goal_notification_receipts (
                goal_id TEXT NOT NULL,
                goal_version INTEGER NOT NULL,
                status TEXT NOT NULL,
                channel TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY(goal_id, goal_version, status, channel)
            )"""
        )
        self._db.commit()

    def claim(
        self, *, goal_id: str, goal_version: int, status: str, channel: str
    ) -> bool:
        try:
            with self._db:
                self._db.execute(
                    "INSERT INTO goal_notification_receipts VALUES (?, ?, ?, ?, ?)",
                    (goal_id, goal_version, status, channel, time.time()),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def close(self) -> None:
        self._db.close()
