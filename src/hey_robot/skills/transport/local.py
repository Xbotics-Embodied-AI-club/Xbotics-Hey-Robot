"""In-process SkillClient for tests and single-process development."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from hey_robot.skills.client import SkillClient
from hey_robot.skills.context import SkillContext
from hey_robot.skills.models import SkillCommand, SkillEvent
from hey_robot.skills.registry import SkillRegistry
from hey_robot.skills.resources import ResourceManager
from hey_robot.skills.worker import SkillWorker

if TYPE_CHECKING:
    from hey_robot.persistence.run_store import RunStore

logger = logging.getLogger(__name__)


class LocalSkillClient(SkillClient):
    """Run skills in background tasks and expose their append-only events."""

    def __init__(
        self,
        registry: SkillRegistry,
        *,
        resources: ResourceManager | None = None,
        context_factory: Callable[[SkillCommand], SkillContext] | None = None,
        run_store: RunStore,
        subscriber_queue_size: int = 128,
        cancel_model: Callable[[str], Awaitable[None]] | None = None,
        emergency_stop: Callable[[str, str], Awaitable[None]] | None = None,
        project_event: Callable[[SkillEvent], Awaitable[None]] | None = None,
        start_projection: Callable[[], Awaitable[None]] | None = None,
        stop_projection: Callable[[], Awaitable[None]] | None = None,
        projection_health: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._emergency_stop = emergency_stop
        self._project_event = project_event
        self._start_projection = start_projection
        self._stop_projection = stop_projection
        self._projection_health = projection_health
        self._projection_published = 0
        self._projection_failed = 0
        self._projection_dropped = 0
        self._projection_queue: asyncio.Queue[SkillEvent] = asyncio.Queue(
            maxsize=subscriber_queue_size
        )
        self._projection_task: asyncio.Task[object] | None = None
        self._worker = SkillWorker(
            registry,
            resources=resources or ResourceManager(),
            context_factory=context_factory,
            run_store=run_store,
            subscriber_queue_size=subscriber_queue_size,
            cancel_model=cancel_model,
            project_event=self._enqueue_projection if project_event else None,
        )
        self._started = False

    async def submit(self, command: SkillCommand) -> str:
        await self._ensure_started()
        return await self._worker.submit(command)

    async def start(self) -> None:
        await self._ensure_started()

    async def cancel(self, run_id: str, *, reason: str) -> None:
        await self._ensure_started()
        await self._worker.cancel(run_id, reason=reason)

    async def emergency_stop(self, robot_id: str, *, reason: str) -> None:
        if self._emergency_stop is None:
            raise RuntimeError("robot emergency-stop control plane is unavailable")
        stop_result, _ = await asyncio.gather(
            self._emergency_stop(robot_id, reason),
            self._worker.cancel_robot(robot_id, reason=reason),
            return_exceptions=True,
        )
        if isinstance(stop_result, BaseException):
            raise stop_result

    async def events(self) -> AsyncIterator[SkillEvent]:
        await self._ensure_started()
        async for event in self._worker.events():
            yield event

    async def status(self, run_id: str) -> SkillEvent | None:
        return await self._worker.status(run_id)

    async def close(self) -> None:
        await self._worker.close()
        if self._projection_task is not None:
            self._projection_task.cancel()
            await asyncio.gather(self._projection_task, return_exceptions=True)
            self._projection_task = None
        if self._stop_projection is not None:
            await self._stop_projection()
        self._started = False

    async def _ensure_started(self) -> None:
        if not self._started:
            if self._start_projection is not None:
                await self._start_projection()
            if self._project_event is not None:
                self._projection_task = asyncio.create_task(
                    self._projection_loop(), name="skill-event-projection"
                )
            await self._worker.start()
            self._started = True

    async def _enqueue_projection(self, event: SkillEvent) -> None:
        if self._projection_queue.full():
            self._projection_queue.get_nowait()
            self._projection_dropped += 1
            self._report_projection_health()
        self._projection_queue.put_nowait(event)

    async def _projection_loop(self) -> None:
        assert self._project_event is not None
        while True:
            event = await self._projection_queue.get()
            try:
                await self._project_event(event)
            except Exception as exc:
                self._projection_failed += 1
                self._report_projection_health()
                logger.warning(
                    "skill event projection failed for run %s: %s",
                    event.run_id,
                    exc,
                )
            else:
                self._projection_published += 1
                self._report_projection_health()

    @property
    def projection_stats(self) -> dict[str, int]:
        return {
            "published": self._projection_published,
            "failed": self._projection_failed,
            "dropped": self._projection_dropped,
            "queued": self._projection_queue.qsize(),
        }

    def _report_projection_health(self) -> None:
        if self._projection_health is not None:
            self._projection_health(self.projection_stats)
