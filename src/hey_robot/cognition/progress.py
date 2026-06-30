from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RobotAgentProgress:
    phase: str
    summary: str
    episode_id: str | None = None
    agent_id: str | None = None
    robot_id: str | None = None
    skill_id: str | None = None
    trace_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
