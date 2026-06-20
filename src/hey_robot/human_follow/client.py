from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from hey_robot.bus.client import BusClient
from hey_robot.protocol import Topics
from hey_robot.skills.base import SkillResult

ProgressCallback = Callable[..., Awaitable[None]]


@dataclass
class _PendingSession:
    future: asyncio.Future[SkillResult]
    progress: ProgressCallback | None


class HumanFollowServiceClient:
    def __init__(self, bus: BusClient, topics: Topics, robot_ids: list[str]) -> None:
        self.bus = bus
        self.topics = topics
        self.robot_ids = robot_ids
        self._pending: dict[str, _PendingSession] = {}

    async def start(self) -> None:
        await self.bus.subscribe(
            [
                self.topics.for_robot(self.topics.human_follow_status, robot_id)
                for robot_id in self.robot_ids
            ],
            self._on_status,
        )

    async def run(
        self,
        *,
        robot_id: str,
        skill_id: str,
        arguments: dict[str, Any],
        progress: ProgressCallback | None = None,
    ) -> SkillResult:
        session_id = f"follow_{uuid.uuid4().hex}"
        future: asyncio.Future[SkillResult] = asyncio.get_running_loop().create_future()
        self._pending[session_id] = _PendingSession(future, progress)
        topic = self.topics.for_robot(self.topics.human_follow_command, robot_id)
        await self.bus.publish(
            topic,
            {
                "action": "start",
                "robot_id": robot_id,
                "skill_id": skill_id,
                "session_id": session_id,
                "arguments": arguments,
            },
        )
        try:
            return await future
        except asyncio.CancelledError:
            await self.bus.publish(
                topic,
                {
                    "action": "stop",
                    "robot_id": robot_id,
                    "skill_id": skill_id,
                    "session_id": session_id,
                },
            )
            raise
        finally:
            self._pending.pop(session_id, None)

    async def _on_status(self, _topic: str, payload: dict[str, Any]) -> None:
        pending = self._pending.get(str(payload.get("session_id") or ""))
        if pending is None:
            return
        if payload.get("kind") == "progress":
            if pending.progress is not None:
                await pending.progress(
                    phase=str(payload.get("phase") or "following"),
                    summary=str(payload.get("phase") or "following"),
                    progress=0.6,
                    step=str(payload.get("phase") or "following"),
                    metadata={"ux": dict(payload)},
                )
            return
        if payload.get("kind") == "result" and not pending.future.done():
            success = bool(payload.get("success", False))
            pending.future.set_result(
                SkillResult(
                    success=success,
                    summary=str(payload.get("summary") or "human follow finished"),
                    status="completed" if success else "failed",
                    failure_mode=payload.get("failure_mode"),
                    error=payload.get("error"),
                    data={"session_id": payload.get("session_id")},
                )
            )
