from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from hey_robot.protocol import SkillIntent
from hey_robot.skills.catalog import RobotSkillSpec


@dataclass(frozen=True)
class CapabilityHealth:
    name: str
    online: bool
    loaded: bool = True
    busy: bool = False
    robot_id: str = ""
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    current_skill_id: str | None = None
    error_code: str | None = None
    version: str | None = None


@dataclass(frozen=True)
class CapabilityExecutionRequest:
    service_id: str
    intent: SkillIntent
    contract: RobotSkillSpec
    timeout_sec: float


@dataclass(frozen=True)
class CapabilityExecutionResult:
    success: bool
    summary: str
    status: str = "completed"
    failure_mode: str | None = None
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None


class CapabilityClient(Protocol):
    async def health(self) -> CapabilityHealth: ...

    async def execute(
        self, request: CapabilityExecutionRequest
    ) -> CapabilityExecutionResult: ...

    async def cancel(self, skill_id: str) -> None: ...
