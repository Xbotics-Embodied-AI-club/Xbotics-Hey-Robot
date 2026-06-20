from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from hey_robot.events.event import RuntimeEvent


class RuntimeEventStore:
    def __init__(
        self,
        root: str | Path = "runtime/events",
        *,
        max_items: int = 1000,
        trim_every: int = 100,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "events.jsonl"
        self.max_items = max(1, int(max_items))
        self.trim_every = max(1, int(trim_every))
        self._writes_since_trim = 0
        self._trim()

    def append(self, event: RuntimeEvent) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
        self._writes_since_trim += 1
        if self._writes_since_trim >= self.trim_every:
            self._trim()
            self._writes_since_trim = 0

    def recent(self, limit: int = 100) -> list[dict]:
        if not self.path.exists():
            return []
        lines = _tail_lines(self.path, max(1, min(int(limit), self.max_items)))
        items: list[dict] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                items.append(payload)
        return items

    def count(self) -> int:
        if not self.path.exists():
            return 0
        return sum(
            1
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    def _trim(self) -> None:
        if not self.path.exists():
            return
        lines = self.path.read_text(encoding="utf-8").splitlines()
        if len(lines) <= self.max_items:
            return
        self.path.write_text(
            "\n".join(lines[-self.max_items :]) + "\n", encoding="utf-8"
        )


def _tail_lines(path: Path, limit: int, *, block_size: int = 64 * 1024) -> list[str]:
    """Read only enough data from the end of a JSONL file for ``limit`` lines."""
    if limit <= 0:
        return []
    chunks: list[bytes] = []
    newline_count = 0
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        while position > 0 and newline_count <= limit:
            size = min(block_size, position)
            position -= size
            handle.seek(position)
            chunk = handle.read(size)
            chunks.append(chunk)
            newline_count += chunk.count(b"\n")
    text = b"".join(reversed(chunks)).decode("utf-8", errors="replace")
    return text.splitlines()[-limit:]
