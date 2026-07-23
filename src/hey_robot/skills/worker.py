"""In-process skill worker shell shared by local tests and composition roots."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Protocol

from hey_robot.skills.client import SkillClient
from hey_robot.skills.context import SkillContext
from hey_robot.skills.models import SkillCommand, SkillEvent
from hey_robot.skills.registry import SkillRegistry
from hey_robot.skills.resources import ResourceManager
from hey_robot.skills.runner import SkillEventSink, SkillRunner


class RunEventStore(Protocol):
    def append_event(self, event: SkillEvent) -> None: ...


class SkillWorker(SkillClient):
    """Queue commands, run skills in managed tasks, and broadcast events."""

    def __init__(
        self,
        registry: SkillRegistry,
        *,
        resources: ResourceManager | None = None,
        context_factory: Callable[[SkillCommand], SkillContext] | None = None,
        run_store: RunEventStore | None = None,
    ) -> None:
        self._commands: asyncio.Queue[SkillCommand] = asyncio.Queue()
        self._submitted: dict[str, SkillCommand] = {}
        self._tasks: dict[str, asyncio.Task[object]] = {}
        self._subscribers: set[asyncio.Queue[SkillEvent]] = set()
        self._latest: dict[str, SkillEvent] = {}
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
        if self._consumer is None:
            self._consumer = asyncio.create_task(
                self._consume(), name="skill-worker:commands"
            )

    async def submit(self, command: SkillCommand) -> str:
        if self._closed:
            raise RuntimeError("skill worker is closed")
        existing = self._submitted.get(command.run_id)
        if existing is not None:
            if existing != command:
                raise ValueError(
                    f"run_id {command.run_id!r} was submitted with a different command"
                )
            return command.run_id
        self._submitted[command.run_id] = command
        await self._commands.put(command)
        return command.run_id

    async def cancel(self, run_id: str, *, reason: str) -> None:
        del reason
        if run_id in self._submitted:
            self._runner.cancel(run_id)

    async def events(self) -> AsyncIterator[SkillEvent]:
        queue: asyncio.Queue[SkillEvent] = asyncio.Queue()
        self._subscribers.add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers.discard(queue)

    async def status(self, run_id: str) -> SkillEvent | None:
        return self._latest.get(run_id)

    async def close(self) -> None:
        self._closed = True
        if self._consumer is not None:
            self._consumer.cancel()
            await asyncio.gather(self._consumer, return_exceptions=True)
            self._consumer = None
        for run_id in self._tasks:
            self._runner.cancel(run_id)
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()

    async def _consume(self) -> None:
        while True:
            command = await self._commands.get()
            if command.run_id in self._tasks:
                continue
            self._tasks[command.run_id] = asyncio.create_task(
                self._runner.execute(command),
                name=f"skill:{command.run_id}",
            )

    async def _emit(self, event: SkillEvent) -> None:
        self._latest[event.run_id] = event
        if self._run_store is not None:
            self._run_store.append_event(event)
        for queue in tuple(self._subscribers):
            queue.put_nowait(event)


class _WorkerEventSink(SkillEventSink):
    def __init__(self, worker: SkillWorker) -> None:
        self._worker = worker

    async def emit(self, event: SkillEvent) -> None:
        await self._worker._emit(event)
