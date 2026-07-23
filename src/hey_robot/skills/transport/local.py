"""In-process SkillClient for tests and single-process development."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

from hey_robot.skills.client import SkillClient
from hey_robot.skills.context import SkillContext
from hey_robot.skills.models import SkillCommand, SkillEvent
from hey_robot.skills.registry import SkillRegistry
from hey_robot.skills.resources import ResourceManager
from hey_robot.skills.worker import RunEventStore, SkillWorker


class LocalSkillClient(SkillClient):
    """Run skills in background tasks and expose their append-only events."""

    def __init__(
        self,
        registry: SkillRegistry,
        *,
        resources: ResourceManager | None = None,
        context_factory: Callable[[SkillCommand], SkillContext] | None = None,
        run_store: RunEventStore | None = None,
    ) -> None:
        self._worker = SkillWorker(
            registry,
            resources=resources or ResourceManager(),
            context_factory=context_factory,
            run_store=run_store,
        )
        self._started = False

    async def submit(self, command: SkillCommand) -> str:
        await self._ensure_started()
        return await self._worker.submit(command)

    async def cancel(self, run_id: str, *, reason: str) -> None:
        await self._ensure_started()
        await self._worker.cancel(run_id, reason=reason)

    async def events(self) -> AsyncIterator[SkillEvent]:
        await self._ensure_started()
        async for event in self._worker.events():
            yield event

    async def status(self, run_id: str) -> SkillEvent | None:
        return await self._worker.status(run_id)

    async def close(self) -> None:
        await self._worker.close()
        self._started = False

    async def _ensure_started(self) -> None:
        if not self._started:
            await self._worker.start()
            self._started = True
