"""NATS transport for native SkillClient and SkillWorker boundaries."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator

from hey_robot.bus.factory import create_bus_client
from hey_robot.config import DeploymentConfig
from hey_robot.protocol import (
    Envelope,
    SkillControl,
    SkillControlResult,
    Topics,
)
from hey_robot.protocol.messages import from_payload, to_payload
from hey_robot.skills.models import SkillCancel, SkillCommand, SkillEvent
from hey_robot.skills.worker import SkillWorker


class BusSkillClient:
    """Submit native skill runs to a remote worker over the deployment bus."""

    def __init__(
        self,
        config: DeploymentConfig,
        *,
        subscriber_queue_size: int = 128,
        control_timeout_sec: float = 10.0,
    ) -> None:
        self._bus = create_bus_client(config.deployment.bus, role="robot-agent")
        self._topics = Topics()
        self._queue: asyncio.Queue[SkillEvent] = asyncio.Queue(
            maxsize=max(1, subscriber_queue_size)
        )
        self._latest: dict[str, SkillEvent] = {}
        self._control_waiters: dict[str, asyncio.Future[SkillControlResult]] = {}
        self._connect_lock = asyncio.Lock()
        self._connected = False
        self._control_timeout_sec = control_timeout_sec

    async def connect(self) -> None:
        if self._connected:
            return
        async with self._connect_lock:
            if self._connected:
                return
            await self._bus.connect()
            await self._bus.subscribe([self._topics.skill_run_event], self._on_event)
            await self._bus.subscribe(
                [self._topics.skill_control_result], self._on_control_result
            )
            self._connected = True

    async def submit(self, command: SkillCommand) -> str:
        await self.connect()
        await self._bus.publish(self._topics.skill_command, to_payload(command))
        return command.run_id

    async def cancel(self, run_id: str, *, reason: str) -> None:
        await self.connect()
        await self._bus.publish(
            self._topics.skill_cancel,
            to_payload(SkillCancel(Envelope(), run_id, reason)),
        )

    async def emergency_stop(self, robot_id: str, *, reason: str) -> None:
        await self.connect()
        control_id = f"ctrl_{uuid.uuid4().hex}"
        future = asyncio.get_running_loop().create_future()
        self._control_waiters[control_id] = future
        control = SkillControl(
            envelope=Envelope(robot_id=robot_id),
            control_id=control_id,
            action="emergency_stop",
            target_skill_id=None,
            task_id=None,
            reason=reason,
        )
        try:
            await self._bus.publish(self._topics.skill_control, to_payload(control))
            result = await asyncio.wait_for(future, timeout=self._control_timeout_sec)
        finally:
            self._control_waiters.pop(control_id, None)
        if result.status != "completed":
            raise RuntimeError(result.error or "remote emergency stop failed")

    async def events(self) -> AsyncIterator[SkillEvent]:
        await self.connect()
        while True:
            yield await self._queue.get()

    async def status(self, run_id: str) -> SkillEvent | None:
        await self.connect()
        return self._latest.get(run_id)

    async def close(self) -> None:
        if self._connected:
            await self._bus.close()
        self._connected = False
        for future in self._control_waiters.values():
            if not future.done():
                future.cancel()
        self._control_waiters.clear()

    async def _on_event(self, _topic: str, payload: dict) -> None:
        event = from_payload(SkillEvent, payload)
        self._latest[event.run_id] = event
        if self._queue.full():
            self._queue.get_nowait()
        self._queue.put_nowait(event)

    async def _on_control_result(self, _topic: str, payload: dict) -> None:
        result = from_payload(SkillControlResult, payload)
        future = self._control_waiters.get(result.control_id)
        if future is not None and not future.done():
            future.set_result(result)


class BusSkillServer:
    """Expose one native SkillWorker over the deployment bus."""

    def __init__(self, config: DeploymentConfig, worker: SkillWorker) -> None:
        self._bus = create_bus_client(config.deployment.bus, role="skill_controller")
        self._worker = worker
        self._topics = Topics()
        self._relay_task: asyncio.Task[object] | None = None
        self._stop = asyncio.Event()
        self.ready = asyncio.Event()

    async def start(self) -> None:
        await self._bus.connect()
        await self._bus.subscribe([self._topics.skill_command], self._on_command)
        await self._bus.subscribe([self._topics.skill_cancel], self._on_cancel)
        await self._bus.subscribe([self._topics.skill_control], self._on_control)
        await self._worker.start()
        self._relay_task = asyncio.create_task(
            self._relay_events(), name="skill-bus-event-relay"
        )
        # Let the async generator register its subscriber before accepting commands.
        await asyncio.sleep(0)
        self.ready.set()
        await self._stop.wait()

    async def close(self) -> None:
        self._stop.set()
        if self._relay_task is not None:
            self._relay_task.cancel()
            await asyncio.gather(self._relay_task, return_exceptions=True)
            self._relay_task = None
        await self._worker.close()
        await self._bus.close()

    async def _on_command(self, _topic: str, payload: dict) -> None:
        await self._worker.submit(from_payload(SkillCommand, payload))

    async def _on_cancel(self, _topic: str, payload: dict) -> None:
        cancel = from_payload(SkillCancel, payload)
        await self._worker.cancel(cancel.run_id, reason=cancel.reason)

    async def _on_control(self, _topic: str, payload: dict) -> None:
        control = from_payload(SkillControl, payload)
        robot_id = control.envelope.robot_id or ""
        error: str | None = None
        try:
            if control.action == "emergency_stop":
                if not robot_id:
                    raise ValueError("emergency stop requires robot_id")
                await self._worker.emergency_stop(robot_id, reason=control.reason)
            elif control.target_skill_id:
                await self._worker.cancel(
                    control.target_skill_id, reason=control.reason
                )
        except Exception as exc:
            error = str(exc) or type(exc).__name__
        await self._bus.publish(
            self._topics.skill_control_result,
            to_payload(
                SkillControlResult(
                    envelope=control.envelope,
                    control_id=control.control_id,
                    action=control.action,
                    target_skill_id=control.target_skill_id,
                    status="failed" if error else "completed",
                    robot_idle_confirmed=error is None,
                    error=error,
                )
            ),
        )

    async def _relay_events(self) -> None:
        async for event in self._worker.events():
            await self._bus.publish(self._topics.skill_run_event, to_payload(event))
