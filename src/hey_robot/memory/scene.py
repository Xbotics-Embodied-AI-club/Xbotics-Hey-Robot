from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hey_robot.perception.scene import SceneUnderstanding
from hey_robot.protocol import RobotObservation, RobotStatus


@dataclass(frozen=True)
class SceneMemoryRecord:
    record_id: str
    episode_id: str | None
    robot_id: str | None
    frame_id: int | None
    summary: str
    task: str | None = None
    image_count: int = 0
    artifact_count: int = 0
    camera: dict[str, Any] = field(default_factory=dict)
    arm: dict[str, Any] = field(default_factory=dict)
    battery: dict[str, Any] = field(default_factory=dict)
    status: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SceneSummarizer:
    """Deterministic scene summarizer for robot observations.

    This is intentionally lightweight. It gives the Agent a stable scene-memory
    contract before a VLM scene captioner is enabled. A VLM implementation can
    replace this class without changing storage or Agent service wiring.
    """

    def summarize(
        self, observation: RobotObservation, status: RobotStatus | None = None
    ) -> SceneMemoryRecord:
        metrics = status.metrics if status is not None else {}
        camera = _dict(metrics.get("camera") or observation.raw.get("camera"))
        arm = _dict(metrics.get("arm_status") or observation.raw.get("arm_status"))
        battery = _dict(metrics.get("battery") or observation.raw.get("battery"))
        state = status.state if status is not None else observation.raw.get("state")
        parts = [
            f"frame={observation.frame_id}",
            f"images={len(observation.images)}",
            f"state={state or 'unknown'}",
        ]
        if camera:
            parts.append(
                f"camera={'available' if camera.get('frame_available') else 'unavailable'}"
            )
            if camera.get("image_shape"):
                parts.append(f"shape={camera.get('image_shape')}")
        if battery:
            parts.append(f"battery={battery.get('status', 'unknown')}")
            if battery.get("voltage") is not None:
                parts.append(f"voltage={battery.get('voltage')}")
        if observation.task:
            parts.append(f"task={observation.task}")
        return SceneMemoryRecord(
            record_id=f"scene_{uuid.uuid4().hex[:16]}",
            episode_id=observation.envelope.episode_id,
            robot_id=observation.envelope.robot_id,
            frame_id=observation.frame_id,
            summary="; ".join(parts),
            task=observation.task,
            image_count=len(observation.images),
            artifact_count=len(observation.artifacts),
            camera=camera,
            arm=arm,
            battery=battery,
            status={
                "state": state,
                "skill_id": status.skill_id if status is not None else None,
                "success": status.success if status is not None else None,
                "error": status.error if status is not None else None,
            },
            confidence=0.65 if observation.images else 0.35,
            metadata={"source": "scene_summarizer.deterministic"},
        )

    def from_understanding(
        self,
        observation: RobotObservation,
        understanding: SceneUnderstanding,
        status: RobotStatus | None = None,
    ) -> SceneMemoryRecord:
        base = self.summarize(observation, status)
        return SceneMemoryRecord(
            record_id=base.record_id,
            episode_id=base.episode_id,
            robot_id=base.robot_id,
            frame_id=base.frame_id,
            summary=understanding.summary or base.summary,
            task=base.task,
            image_count=base.image_count,
            artifact_count=base.artifact_count,
            camera=base.camera,
            arm=base.arm,
            battery=base.battery,
            status=base.status,
            confidence=understanding.confidence or base.confidence,
            metadata={
                **base.metadata,
                "understanding": understanding.to_dict(),
                "source": "scene_captioner",
            },
        )


class SceneMemoryStore:
    def __init__(self, root: str | Path, *, max_items: int = 1000) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_items = max(1, int(max_items))

    def append(self, record: SceneMemoryRecord) -> SceneMemoryRecord:
        path = self._path(record.episode_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._is_duplicate(path, record):
            return record
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        self._trim(path)
        return record

    def _is_duplicate(self, path: Path, record: SceneMemoryRecord) -> bool:
        last = self._last_record(path)
        if last is None:
            return False
        return last.summary == record.summary

    def recent(
        self, episode_id: str | None = None, *, limit: int = 20
    ) -> list[SceneMemoryRecord]:
        entries = self._entries(episode_id)
        newest_first = sorted(
            entries, key=lambda item: (item[1].timestamp, item[0]), reverse=True
        )
        return [record for _, record in newest_first[: max(1, int(limit))]]

    def prompt_context(self, episode_id: str | None, *, limit: int = 6) -> str | None:
        entries = self._entries(episode_id)
        recent_entries = sorted(
            entries, key=lambda item: (item[1].timestamp, item[0]), reverse=True
        )[: max(1, int(limit))]
        records = [
            record
            for _, record in sorted(
                recent_entries, key=lambda item: (item[1].timestamp, item[0])
            )
        ]
        if not records:
            return None
        lines = ["Recent scene memory:"]
        for record in records:
            frame = (
                f"frame={record.frame_id}"
                if record.frame_id is not None
                else "frame=unknown"
            )
            lines.append(f"- {frame}: {record.summary}")
        return "\n".join(lines)

    def _entries(self, episode_id: str | None) -> list[tuple[int, SceneMemoryRecord]]:
        paths = (
            [self._path(episode_id)]
            if episode_id
            else sorted(self.root.glob("*.scene.jsonl"))
        )
        entries: list[tuple[int, SceneMemoryRecord]] = []
        sequence = 0
        for path in paths:
            for record in self._read(path):
                entries.append((sequence, record))
                sequence += 1
        return entries

    def _read(self, path: Path) -> list[SceneMemoryRecord]:
        if not path.exists():
            return []
        records: list[SceneMemoryRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                records.append(SceneMemoryRecord(**payload))
            except (TypeError, json.JSONDecodeError):
                continue
        return records

    def _trim(self, path: Path) -> None:
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) <= self.max_items:
            return
        tmp = path.with_suffix(
            f".{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
        )
        tmp.write_text("\n".join(lines[-self.max_items :]) + "\n", encoding="utf-8")
        for attempt in range(3):
            try:
                tmp.replace(path)
                return
            except PermissionError:
                if attempt == 2:
                    with suppress(OSError):
                        tmp.unlink(missing_ok=True)
                    return
                time.sleep(0.05 * (attempt + 1))

    def _last_record(self, path: Path) -> SceneMemoryRecord | None:
        if not path.exists():
            return None
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    return None
                handle.seek(max(0, handle.tell() - 4096))
                tail = handle.read().decode("utf-8")
                lines = [line.strip() for line in tail.splitlines() if line.strip()]
                if not lines:
                    return None
                payload = json.loads(lines[-1])
                return SceneMemoryRecord(**payload)
        except (OSError, json.JSONDecodeError, TypeError, UnicodeDecodeError):
            return None

    def _path(self, episode_id: str | None) -> Path:
        key = _sanitize(episode_id or "global")
        return self.root / f"{key}.scene.jsonl"


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value)
