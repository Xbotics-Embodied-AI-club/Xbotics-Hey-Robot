"""In-process skill worker shell shared by local tests and composition roots."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from hey_robot.skills.context import SkillContext
from hey_robot.skills.models import SkillCommand, SkillEvent, SkillResult
from hey_robot.skills.registry import SkillRegistry
from hey_robot.skills.resources import ResourceManager
from hey_robot.skills.runner import SkillEventSink, SkillRunner

if TYPE_CHECKING:
    from hey_robot.persistence.run_store import RunStore

logger = logging.getLogger(__name__)


class SkillWorker:
    """Queue commands, run skills in managed tasks, and broadcast events."""

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
        self._commands: asyncio.Queue[SkillCommand] = asyncio.Queue()
        self._pending: dict[str, SkillCommand] = {}
        self._tasks: dict[str, asyncio.Task[object]] = {}
        self._subscribers: set[asyncio.Queue[SkillEvent]] = set()
        self._subscriber_queue_size = max(1, int(subscriber_queue_size))
        self._status_lock = asyncio.Lock()
        self._cancel_model = cancel_model
        self._emergency_stop = emergency_stop
        self._project_event = project_event
        self._start_projection = start_projection
        self._stop_projection = stop_projection
        self._projection_health = projection_health
        self._projection_published = 0
        self._projection_failed = 0
        self._projection_dropped = 0
        self._projection_queue: asyncio.Queue[SkillEvent] = asyncio.Queue(
            maxsize=self._subscriber_queue_size
        )
        self._projection_task: asyncio.Task[object] | None = None
        self._cancelling: set[str] = set()
        self._run_store = run_store
        self._runner = SkillRunner(
            registry,
            resources=resources or ResourceManager(),
            events=_WorkerEventSink(self),
            context_factory=context_factory,
        )
        self._consumer: asyncio.Task[object] | None = None
        self._closed = False

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("skill worker is closed")
        if self._consumer is None:
            if self._start_projection is not None:
                await self._start_projection()
            if self._project_event is not None:
                self._projection_task = asyncio.create_task(
                    self._projection_loop(), name="skill-event-projection"
                )
            self._consumer = asyncio.create_task(
                self._consume(), name="skill-worker:commands"
            )

    async def submit(self, command: SkillCommand) -> str:
        if self._closed:
            raise RuntimeError("skill worker is closed")
        if self._consumer is None:
            await self.start()
        existing = self._pending.get(command.run_id)
        if existing is not None:
            if existing != command:
                raise ValueError(
                    f"run_id {command.run_id!r} was submitted with a different command"
                )
            return command.run_id
        existing = self._run_store.submission(command.run_id)
        if existing is not None:
            if existing != command:
                raise ValueError(
                    f"run_id {command.run_id!r} was submitted with a different command"
                )
            return command.run_id
        self._run_store.record_submission(command)
        self._pending[command.run_id] = command
        await self._commands.put(command)
        return command.run_id

    async def cancel(self, run_id: str, *, reason: str) -> None:
        del reason
        if run_id in self._cancelling:
            return
        if run_id not in self._pending and run_id not in self._tasks:
            return
        self._cancelling.add(run_id)
        self._runner.cancel(run_id)
        task = self._tasks.get(run_id)
        if task is not None:
            task.cancel()
        try:
            if self._cancel_model is not None:
                await self._cancel_model(run_id)
        except Exception as exc:
            logger.warning("model cancellation failed for run %s: %s", run_id, exc)
        finally:
            self._cancelling.discard(run_id)

    async def cancel_robot(self, robot_id: str, *, reason: str) -> None:
        run_ids = {
            run_id
            for run_id, command in self._pending.items()
            if command.robot_id == robot_id
        }
        await asyncio.gather(
            *(self.cancel(run_id, reason=reason) for run_id in run_ids)
        )

    async def emergency_stop(self, robot_id: str, *, reason: str) -> None:
        if self._emergency_stop is None:
            raise RuntimeError("robot emergency-stop control plane is unavailable")
        stop_result, _ = await asyncio.gather(
            self._emergency_stop(robot_id, reason),
            self.cancel_robot(robot_id, reason=reason),
            return_exceptions=True,
        )
        if isinstance(stop_result, BaseException):
            raise stop_result

    async def events(self) -> AsyncIterator[SkillEvent]:
        queue: asyncio.Queue[SkillEvent] = asyncio.Queue(
            maxsize=self._subscriber_queue_size
        )
        self._subscribers.add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers.discard(queue)

    async def status(self, run_id: str) -> SkillEvent | None:
        async with self._status_lock:
            return await self._status_locked(run_id)

    async def _status_locked(self, run_id: str) -> SkillEvent | None:
        active = run_id in self._pending or run_id in self._tasks
        latest = self._run_store.latest_event(run_id)
        if active or (latest is not None and _terminal(latest)):
            return latest
        command = self._run_store.submission(run_id)
        if command is None and latest is None:
            return None
        if command is None:
            assert latest is not None
            envelope = latest.envelope
            name = latest.name
        else:
            envelope = command.envelope
            name = command.name
        event = SkillEvent(
            envelope=envelope,
            run_id=run_id,
            sequence=(latest.sequence + 1 if latest is not None else 1),
            name=name,
            phase="failed",
            timestamp=time.time(),
            summary="skill execution ownership was lost during restart",
            result=SkillResult(
                False,
                "skill execution ownership was lost during restart",
                "failed",
                failure_mode="execution_lost",
                error="no active worker task owns this persisted non-terminal run",
            ),
        )
        await self._emit(event)
        return event

    async def close(self) -> None:
        self._closed = True
        if self._consumer is not None:
            self._consumer.cancel()
            await asyncio.gather(self._consumer, return_exceptions=True)
            self._consumer = None
        for run_id in tuple(self._tasks):
            self._runner.cancel(run_id)
            self._tasks[run_id].cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        if self._projection_task is not None:
            self._projection_task.cancel()
            await asyncio.gather(self._projection_task, return_exceptions=True)
            self._projection_task = None
        if self._stop_projection is not None:
            await self._stop_projection()
        self._tasks.clear()
        self._pending.clear()
        self._cancelling.clear()

    async def _consume(self) -> None:
        while True:
            command = await self._commands.get()
            if command.run_id in self._tasks:
                continue
            self._tasks[command.run_id] = asyncio.create_task(
                self._run_command(command),
                name=f"skill:{command.run_id}",
            )

    async def _run_command(self, command: SkillCommand) -> None:
        try:
            await self._runner.execute(command)
        finally:
            self._tasks.pop(command.run_id, None)
            self._pending.pop(command.run_id, None)

    async def _emit(self, event: SkillEvent) -> None:
        event = self._run_store.append_event(event)
        if self._project_event is not None:
            self._enqueue_projection(event)
        for queue in tuple(self._subscribers):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(event)

    def _enqueue_projection(self, event: SkillEvent) -> None:
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


class _WorkerEventSink(SkillEventSink):
    def __init__(self, worker: SkillWorker) -> None:
        self._worker = worker

    async def emit(self, event: SkillEvent) -> None:
        await self._worker._emit(event)


def _terminal(event: SkillEvent) -> bool:
    return event.phase in {"completed", "failed", "cancelled"}
