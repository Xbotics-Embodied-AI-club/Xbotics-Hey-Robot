from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RobotTaskEvent:
    event_id: str
    episode_id: str
    task_id: str | None
    kind: str
    summary: str = ""
    skill_id: str | None = None
    frame_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobotTaskEvent:
        return cls(
            event_id=str(data.get("event_id") or f"event_{uuid.uuid4().hex[:16]}"),
            episode_id=str(data["episode_id"]),
            task_id=str(data["task_id"]) if data.get("task_id") is not None else None,
            kind=str(data.get("kind") or "event"),
            summary=str(data.get("summary") or ""),
            skill_id=str(data["skill_id"])
            if data.get("skill_id") is not None
            else None,
            frame_id=int(data["frame_id"])
            if data.get("frame_id") is not None
            else None,
            metadata=dict(data.get("metadata") or {}),
            timestamp=float(data.get("timestamp") or time.time()),
        )


class RobotTaskEventLog:
    """Append-only physical task event log.

    This is the durable source of truth for task-facing facts. TaskRun is a
    compact execution-state projection of these events, not the event log.
    """

    def __init__(self, root: str | Path, *, max_items_per_episode: int = 1000) -> None:
        self.root = Path(root) / "agent_task_events"
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_items_per_episode = max(1, int(max_items_per_episode))

    def append(
        self,
        *,
        episode_id: str,
        task_id: str | None,
        kind: str,
        summary: str = "",
        skill_id: str | None = None,
        frame_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RobotTaskEvent:
        event = RobotTaskEvent(
            event_id=f"event_{uuid.uuid4().hex[:16]}",
            episode_id=episode_id,
            task_id=task_id,
            kind=kind,
            summary=summary,
            skill_id=skill_id,
            frame_id=frame_id,
            metadata=dict(metadata or {}),
        )
        path = self._path(episode_id)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        self._trim(path)
        return event

    def recent(
        self, episode_id: str, *, limit: int = 50, kind: str | None = None
    ) -> list[RobotTaskEvent]:
        events = [
            event
            for event in self._read(self._path(episode_id))
            if kind is None or event.kind == kind
        ]
        return events[-max(1, int(limit)) :]

    def prompt_context(self, episode_id: str, *, limit: int = 8) -> str | None:
        events = self.recent(episode_id, limit=limit)
        if not events:
            return None
        lines = ["Recent task events:"]
        for event in events:
            details = []
            if event.skill_id:
                details.append(f"skill={event.skill_id}")
            if event.frame_id is not None:
                details.append(f"frame={event.frame_id}")
            detail_text = f" ({', '.join(details)})" if details else ""
            lines.append(f"- {event.kind}{detail_text}: {event.summary}")
        return "\n".join(lines)

    def _read(self, path: Path) -> list[RobotTaskEvent]:
        if not path.exists():
            return []
        events: list[RobotTaskEvent] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                    if isinstance(data, dict):
                        events.append(RobotTaskEvent.from_dict(data))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
        return events

    def _trim(self, path: Path) -> None:
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) <= self.max_items_per_episode:
            return
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(
            "\n".join(lines[-self.max_items_per_episode :]) + "\n", encoding="utf-8"
        )
        tmp.replace(path)

    def _path(self, episode_id: str) -> Path:
        return self.root / f"{_sanitize(episode_id)}.events.jsonl"


def _sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value)
