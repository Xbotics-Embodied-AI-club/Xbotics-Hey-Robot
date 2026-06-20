from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from hey_robot.events.kind import EventKind, Severity


@dataclass(frozen=True)
class RuntimeEvent:
    kind: str
    source: str
    severity: str = Severity.INFO.value
    event_id: str = field(default_factory=lambda: f"ev_{uuid.uuid4().hex}")
    timestamp: float = field(default_factory=time.time)
    trace_id: str | None = None
    episode_id: str | None = None
    agent_id: str | None = None
    robot_id: str | None = None
    channel: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def make(
        cls,
        kind: EventKind | str,
        *,
        source: str,
        severity: Severity | str = Severity.INFO,
        trace_id: str | None = None,
        episode_id: str | None = None,
        agent_id: str | None = None,
        robot_id: str | None = None,
        channel: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        return cls(
            kind=kind.value if isinstance(kind, EventKind) else kind,
            source=source,
            severity=severity.value if isinstance(severity, Severity) else severity,
            trace_id=trace_id,
            episode_id=episode_id,
            agent_id=agent_id,
            robot_id=robot_id,
            channel=channel,
            payload=payload or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
