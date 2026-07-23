"""NATS-compatible command and event transport for distributed skill workers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

from hey_robot.bus.types import MessageBus
from hey_robot.protocol import Topics
from hey_robot.protocol.messages import from_payload, to_payload
from hey_robot.skills.client import SkillClient
from hey_robot.skills.context import SkillContext
from hey_robot.skills.models import SkillCancel, SkillCommand, SkillEvent
from hey_robot.skills.registry import SkillRegistry
from hey_robot.skills.resources import ResourceManager
from hey_robot.skills.runner import SkillRunner


class NatsSkillClient(SkillClient):
    """Harness-side transport. It never executes or waits for a skill run."""

    def __init__(self, bus: MessageBus, *, topics: Topics | None = None) -> None:
        self._bus = bus
        self._topics = topics or Topics()
        self._subscribers: set[asyncio.Queue[SkillEvent]] = set()
        self._latest: dict[str, SkillEvent] = {}
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        await self._bus.subscribe([self._topics.skill_run_event], self._on_event)
        self._started = True

    async def submit(self, command: SkillCommand) -> str:
        self._require_started()
        await self._bus.publish(self._topics.skill_command, to_payload(command))
        return command.run_id

    async def cancel(self, run_id: str, *, reason: str) -> None:
        self._require_started()
        event = SkillCancel(envelope=_cancel_envelope(), run_id=run_id, reason=reason)
        await self._bus.publish(self._topics.skill_cancel, to_payload(event))

    async def events(self) -> AsyncIterator[SkillEvent]:
        self._require_started()
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
        if self._started:
            await self._bus.unsubscribe([self._topics.skill_run_event])
            self._started = False

    async def _on_event(self, _topic: str, payload: dict[str, object]) -> None:
        event = from_payload(SkillEvent, payload)
        current = self._latest.get(event.run_id)
        if current is not None and event.sequence <= current.sequence:
            return
        self._latest[event.run_id] = event
        for queue in tuple(self._subscribers):
            queue.put_nowait(event)

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("NatsSkillClient.start() must be called before use")


class NatsSkillWorker:
    """Worker-side shell: transport owns subscriptions, runner owns execution."""

    def __init__(
        self,
        bus: MessageBus,
        registry: SkillRegistry,
        *,
        topics: Topics | None = None,
        resources: ResourceManager | None = None,
        context_factory: Callable[[SkillCommand], SkillContext] | None = None,
    ) -> None:
        self._bus = bus
        self._topics = topics or Topics()
        self._commands: dict[str, SkillCommand] = {}
        self._tasks: dict[str, asyncio.Task[object]] = {}
        self._runner = SkillRunner(
            registry,
            resources=resources or ResourceManager(),
            events=_NatsEventSink(bus, self._topics),
            context_factory=context_factory,
        )
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        await self._bus.subscribe([self._topics.skill_command], self._on_command)
        await self._bus.subscribe([self._topics.skill_cancel], self._on_cancel)
        self._started = True

    async def close(self) -> None:
        if not self._started:
            return
        for run_id in self._tasks:
            self._runner.cancel(run_id)
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        await self._bus.unsubscribe(
            [self._topics.skill_command, self._topics.skill_cancel]
        )
        self._started = False

    async def _on_command(self, _topic: str, payload: dict[str, object]) -> None:
        command = from_payload(SkillCommand, payload)
        existing = self._commands.get(command.run_id)
        if existing is not None:
            if existing != command:
                raise ValueError(
                    f"run_id {command.run_id!r} was submitted with a different command"
                )
            return
        self._commands[command.run_id] = command
        self._tasks[command.run_id] = asyncio.create_task(
            self._runner.execute(command), name=f"skill:{command.run_id}"
        )

    async def _on_cancel(self, _topic: str, payload: dict[str, object]) -> None:
        cancel = from_payload(SkillCancel, payload)
        if cancel.run_id in self._commands:
            self._runner.cancel(cancel.run_id)


class _NatsEventSink:
    def __init__(self, bus: MessageBus, topics: Topics) -> None:
        self._bus = bus
        self._topics = topics

    async def emit(self, event: SkillEvent) -> None:
        await self._bus.publish(self._topics.skill_run_event, to_payload(event))


def _cancel_envelope():
    from hey_robot.protocol import Envelope

    return Envelope()
