"""Robot client contract exposed to Skills and other in-process consumers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from hey_robot.protocol import RobotObservation
from hey_robot.robot_api.driver import RobotActionSpec


@dataclass(frozen=True)
class RobotClientCapabilities:
    robot_id: str
    actions: tuple[RobotActionSpec, ...] = ()
    cameras: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RobotActionResult:
    success: bool
    summary: str
    status: str = "completed"
    failure_mode: str | None = None
    error: str | None = None
    frame_id: int | None = None
    data: dict[str, Any] = field(default_factory=dict)
    observation: RobotObservation | None = None
    observation_error: str | None = None


class RobotClient(Protocol):
    async def capabilities(self, robot_id: str) -> RobotClientCapabilities: ...

    async def observe(
        self,
        robot_id: str,
        *,
        after_frame_id: int | None = None,
        timeout_sec: float | None = None,
    ) -> RobotObservation: ...

    async def execute(
        self,
        robot_id: str,
        action: str,
        arguments: dict[str, Any],
        *,
        run_id: str,
        expected_frame_id: int | None = None,
    ) -> RobotActionResult: ...

    async def stop(self, robot_id: str, *, reason: str) -> None: ...

    async def emergency_stop(self, robot_id: str, *, reason: str) -> None: ...
