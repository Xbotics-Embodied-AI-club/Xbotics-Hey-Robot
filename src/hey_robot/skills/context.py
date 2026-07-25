"""Small dependency surface exposed to individual skill handlers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hey_robot.protocol import RobotObservation

if TYPE_CHECKING:
    from hey_robot.foundation.clients.models import ModelRouter
    from hey_robot.robot_api import RobotClient


class SkillCancelledError(RuntimeError):
    pass


@dataclass
class SkillContext:
    run_id: str
    task_id: str
    robot_id: str
    robot: RobotClient | None = None
    models: ModelRouter | None = None
    _progress: Callable[[float, str | None], Awaitable[None]] | None = None
    _cancelled: Callable[[], bool] | None = None

    async def observe(
        self,
        *,
        after_frame_id: int | None = None,
        timeout_sec: float | None = None,
    ) -> RobotObservation:
        if self.robot is None:
            raise RuntimeError("robot client is unavailable")
        return await self.robot.observe(
            self.robot_id,
            after_frame_id=after_frame_id,
            timeout_sec=timeout_sec,
        )

    async def progress(self, value: float, summary: str | None = None) -> None:
        if self._progress is not None:
            await self._progress(value, summary)

    def raise_if_cancelled(self) -> None:
        if self._cancelled is not None and self._cancelled():
            raise SkillCancelledError(f"skill run {self.run_id} was cancelled")
