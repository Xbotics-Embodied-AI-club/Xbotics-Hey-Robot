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

    def status(self, interaction_id: str) -> str | None:
        row = self._db.execute(
            "SELECT status FROM interaction_receipts WHERE interaction_id=?",
            (interaction_id,),
        ).fetchone()
        return str(row[0]) if row is not None else None

    def close(self) -> None:
        self._db.close()
