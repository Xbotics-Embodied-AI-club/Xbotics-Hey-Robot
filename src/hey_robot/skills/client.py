"""Transport-neutral client contract between the harness and skill workers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from hey_robot.skills.models import SkillCommand, SkillEvent


class SkillClient(Protocol):
    """Submit bounded skill runs without turning the event stream into RPC."""

    async def submit(self, command: SkillCommand) -> str: ...

    async def cancel(self, run_id: str, *, reason: str) -> None: ...

    async def emergency_stop(self, robot_id: str, *, reason: str) -> None: ...

    def events(self) -> AsyncIterator[SkillEvent]: ...

    async def status(self, run_id: str) -> SkillEvent | None: ...
