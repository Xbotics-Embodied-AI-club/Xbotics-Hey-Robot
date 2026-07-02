from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from hey_robot.protocol import RobotSkillSpec, SkillIntent


@dataclass(frozen=True)
class ServiceHealth:
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
class ServiceInvocationRequest:
    service_id: str
    intent: SkillIntent
    contract: RobotSkillSpec
    timeout_sec: float


@dataclass(frozen=True)
class ServiceInvocationResult:
    success: bool
    summary: str
    status: str = "completed"
    failure_mode: str | None = None
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None


class ModelServiceClient(Protocol):
    async def health(self) -> ServiceHealth: ...

    async def execute(
        self, request: ServiceInvocationRequest
    ) -> ServiceInvocationResult: ...

    async def cancel(self, skill_id: str) -> None: ...
