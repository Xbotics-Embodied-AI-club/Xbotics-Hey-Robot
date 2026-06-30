from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from hey_robot.protocol import SkillEvent
from hey_robot.protocol.messages import to_payload


class SkillPhase(StrEnum):
    CREATED = "created"
    ISSUED = "issued"
    ACCEPTED = "accepted"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    FEEDBACK_PENDING = "feedback_pending"
    CONFIRMED = "confirmed"
    FEEDBACK_FAILED = "feedback_failed"


_TERMINAL = {
    SkillPhase.COMPLETED.value,
    SkillPhase.FAILED.value,
    SkillPhase.INTERRUPTED.value,
    SkillPhase.CONFIRMED.value,
    SkillPhase.FEEDBACK_FAILED.value,
}


@dataclass
class SkillRecord:
    skill_id: str
    phase: str
    name: str | None = None
    objective: str | None = None
    trace_id: str | None = None
    episode_id: str | None = None
    agent_id: str | None = None
    robot_id: str | None = None
    channel: str | None = None
    policy_id: str | None = None
    frame_id_start: int | None = None
    frame_id_latest: int | None = None
    steps_executed: int = 0
    progress: float = 0.0
    issued_at: float | None = None
    accepted_at: float | None = None
    started_at: float | None = None
    ended_at: float | None = None
    updated_at: float = field(default_factory=time.time)
    summary: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timeline: list[dict[str, Any]] = field(default_factory=list)

    @property
    def terminal(self) -> bool:
        return self.phase in _TERMINAL


class SkillStore:
    """Append-only skill event store plus materialized records."""

    def __init__(
        self, root: str | Path = "runtime/skills", *, max_items: int = 1000
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "skill_events.jsonl"
        self.records_path = self.root / "skills.json"
        self.max_items = max(1, int(max_items))
        self._records: dict[str, SkillRecord] = self._load_records()

    def append(self, event: SkillEvent) -> SkillRecord:
        payload = to_payload(event)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        record = self._apply(event)
        self._trim_events()
        self._write_records()
        return record

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        records = sorted(
            self._records.values(), key=lambda item: item.updated_at, reverse=True
        )
        return [
            asdict(record) for record in records[: max(1, min(limit, self.max_items))]
        ]

    def get(self, skill_id: str) -> SkillRecord | None:
        return self._records.get(skill_id)

    def _apply(self, event: SkillEvent) -> SkillRecord:
        now = event.envelope.timestamp or time.time()
        record = self._records.get(event.skill_id)
        if record is None:
            record = SkillRecord(
                skill_id=event.skill_id,
                phase=event.phase,
                name=event.name,
                objective=event.text,
                trace_id=event.envelope.trace_id,
                episode_id=event.envelope.episode_id,
                agent_id=event.envelope.agent_id,
                robot_id=event.envelope.robot_id,
                channel=event.envelope.channel,
            )
            self._records[event.skill_id] = record

        record.phase = event.phase
        record.name = event.name or record.name
        record.objective = event.text or record.objective
        record.trace_id = event.envelope.trace_id or record.trace_id
        record.episode_id = event.envelope.episode_id or record.episode_id
        record.agent_id = event.envelope.agent_id or record.agent_id
        record.robot_id = event.envelope.robot_id or record.robot_id
        record.channel = event.envelope.channel or record.channel
        record.policy_id = event.policy_id or record.policy_id
        if event.frame_id is not None:
            if record.frame_id_start is None:
                record.frame_id_start = event.frame_id
            record.frame_id_latest = event.frame_id
        if event.steps_executed is not None:
            record.steps_executed = int(event.steps_executed)
        if event.progress is not None:
            record.progress = float(event.progress)
        record.summary = event.summary or record.summary
        record.error = event.error or record.error
        record.metadata.update(event.metadata or {})
        record.updated_at = now
        self._mark_phase_time(record, event.phase, now)
        record.timeline.append(payload_with_timestamp(event))
        record.timeline = record.timeline[-100:]
        return record

    def _mark_phase_time(
        self, record: SkillRecord, phase: str, timestamp: float
    ) -> None:
        if phase == SkillPhase.ISSUED.value and record.issued_at is None:
            record.issued_at = timestamp
        elif phase == SkillPhase.ACCEPTED.value and record.accepted_at is None:
            record.accepted_at = timestamp
        elif phase == SkillPhase.EXECUTING.value and record.started_at is None:
            record.started_at = timestamp
        elif phase in _TERMINAL:
            record.ended_at = timestamp

    def _load_records(self) -> dict[str, SkillRecord]:
        if not self.records_path.exists():
            return {}
        try:
            with self.records_path.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}
        records: dict[str, SkillRecord] = {}
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("skill_id"):
                    records[str(item["skill_id"])] = SkillRecord(**item)
        return records

    def _write_records(self) -> None:
        records = sorted(
            self._records.values(), key=lambda item: item.updated_at, reverse=True
        )
        records = records[: self.max_items]
        self._records = {record.skill_id: record for record in records}
        tmp = self.records_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(
                [asdict(record) for record in records],
                handle,
                ensure_ascii=False,
                indent=2,
            )
        tmp.replace(self.records_path)

    def _trim_events(self) -> None:
        if not self.events_path.exists():
            return
        lines = self.events_path.read_text(encoding="utf-8").splitlines()
        if len(lines) <= self.max_items * 10:
            return
        self.events_path.write_text(
            "\n".join(lines[-self.max_items * 10 :]) + "\n", encoding="utf-8"
        )


def payload_with_timestamp(event: SkillEvent) -> dict[str, Any]:
    payload = to_payload(event)
    payload["timestamp"] = event.envelope.timestamp
    return payload
