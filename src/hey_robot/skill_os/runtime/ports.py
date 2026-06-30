from __future__ import annotations

from typing import Any, Protocol

from hey_robot.protocol import RobotObservation
from hey_robot.skill_os.base import SkillResult


class SkillOSPort(Protocol):
    async def request_skill(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> SkillResult: ...


class FoundationCapabilityPort(Protocol):
    async def execute(
        self,
        service_id: str,
        payload: dict[str, Any],
    ) -> Any: ...

    async def health(self, service_id: str) -> Any: ...

    async def cancel(self, service_id: str, execution_id: str) -> Any: ...


class RobotRuntimePort(Protocol):
    async def execute_primitive(self, name: str, arguments: dict[str, Any]) -> Any: ...

    def current_observation(self) -> RobotObservation | None: ...
