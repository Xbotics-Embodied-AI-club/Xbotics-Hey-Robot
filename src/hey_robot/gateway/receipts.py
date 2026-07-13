"""Durable gateway receipts for idempotent inbound interaction handling."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path


class InteractionReceiptStore:
    """Claim each channel interaction exactly once before routing it.

    This store deliberately deduplicates by an upstream interaction identifier,
    not by message text: repeating the same spoken or written request can be a
    legitimate new command.
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
        """Persist a receipt before side effects; return false for replays."""
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
