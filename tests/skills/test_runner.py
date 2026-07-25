from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from hey_robot.persistence import FileRunStore
from hey_robot.protocol import Envelope
from hey_robot.skills import (
    ResourceManager,
    Skill,
    SkillCommand,
    SkillEvent,
    SkillRegistry,
    SkillResult,
    SkillRunner,
    SkillWorker,
)


@dataclass
class EventSink:
    events: list[SkillEvent] = field(default_factory=list)

    async def emit(self, event: SkillEvent) -> None:
        self.events.append(event)


def _command(name: str, arguments: dict, *, run_id: str = "run-1") -> SkillCommand:
    return SkillCommand(
        envelope=Envelope(robot_id="mock0"),
        run_id=run_id,
        task_id="task-1",
        robot_id="mock0",
        name=name,
        arguments=arguments,
    )


def _runner(registry: SkillRegistry, sink: EventSink) -> SkillRunner:
    return SkillRunner(registry, resources=ResourceManager(), events=sink)


async def test_runner_applies_defaults_validates_and_emits_terminal_event() -> None:
    seen: list[dict] = []

    async def handler(_ctx, arguments):
        seen.append(arguments)
        return SkillResult(True, "done", "completed")

    registry = SkillRegistry()
    registry.register(
        Skill(
            "demo",
            "Demo skill.",
            {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "retries": {"type": "integer", "default": 2},
                },
                "required": ["label"],
                "additionalProperties": False,
            },
            handler,
        )
    )
    sink = EventSink()

    result = await _runner(registry, sink).execute(_command("demo", {"label": "cup"}))

    assert result.success is True
    assert seen == [{"label": "cup", "retries": 2}]
    assert [event.phase for event in sink.events] == [
        "accepted",
        "running",
        "completed",
    ]
    assert [event.sequence for event in sink.events] == [1, 2, 3]
    assert sink.events[-1].result == result


async def test_runner_rejects_invalid_arguments_with_a_terminal_event() -> None:
    registry = SkillRegistry()

    async def handler(_ctx, _arguments):
        return SkillResult(True, "unused", "completed")

    registry.register(
        Skill(
            "demo",
            "Demo skill.",
            {"type": "object", "properties": {}, "additionalProperties": False},
            handler,
        )
    )

    sink = EventSink()
    result = await _runner(registry, sink).execute(_command("demo", {"extra": 1}))

    assert result.status == "failed"
    assert result.failure_mode == "invalid_request"
    assert "unexpected arguments" in (result.error or "")
    assert [event.phase for event in sink.events] == ["accepted", "failed"]


async def test_runner_normalizes_an_inconsistent_failure_status() -> None:
    registry = SkillRegistry()

    async def handler(_ctx, _arguments):
        return SkillResult(False, "blocked", "completed")

    registry.register(Skill("blocked", "Always blocked.", {}, handler))
    sink = EventSink()

    result = await _runner(registry, sink).execute(_command("blocked", {}))

    assert result.status == "failed"
    assert sink.events[-1].phase == "failed"


async def test_runner_cancelled_skill_emits_cancelled_terminal_event() -> None:
    registry = SkillRegistry()
    started = asyncio.Event()

    async def handler(ctx, _arguments):
        started.set()
        while True:
            await asyncio.sleep(0)
            ctx.raise_if_cancelled()

    registry.register(Skill("wait", "Wait.", {}, handler, timeout_sec=1))
    sink = EventSink()
    runner = _runner(registry, sink)
    task = asyncio.create_task(runner.execute(_command("wait", {})))
    await started.wait()
    runner.cancel("run-1")

    result = await task

    assert result.status == "cancelled"
    assert sink.events[-1].phase == "cancelled"


async def test_resources_serialize_conflicting_runs() -> None:
    registry = SkillRegistry()
    starts: list[float] = []
    first_started = asyncio.Event()
    release = asyncio.Event()

    async def handler(_ctx, _arguments):
        starts.append(time.monotonic())
        first_started.set()
        await release.wait()
        return SkillResult(True, "done", "completed")

    registry.register(Skill("arm", "Arm.", {}, handler, resources=("arm",)))
    runner = _runner(registry, EventSink())
    first = asyncio.create_task(runner.execute(_command("arm", {}, run_id="run-1")))
    await first_started.wait()
    second = asyncio.create_task(runner.execute(_command("arm", {}, run_id="run-2")))
    await asyncio.sleep(0.01)
    assert len(starts) == 1
    release.set()
    await asyncio.gather(first, second)
    assert len(starts) == 2


async def test_local_client_submits_once_and_streams_terminal_event(tmp_path) -> None:
    registry = SkillRegistry()

    async def handler(_ctx, arguments):
        return SkillResult(True, f"moved {arguments['distance']}", "completed")

    registry.register(
        Skill(
            "move",
            "Move once.",
            {"type": "object", "properties": {"distance": {"type": "number"}}},
            handler,
        )
    )
    client = SkillWorker(registry, run_store=FileRunStore(tmp_path / "runs"))
    stream = client.events()
    first_event = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    command = _command("move", {"distance": 0.2})

    assert await client.submit(command) == command.run_id
    assert await client.submit(command) == command.run_id
    assert (await first_event).phase == "accepted"
    terminal = await anext(stream)
    while terminal.phase not in {"completed", "failed", "cancelled"}:
        terminal = await anext(stream)

    assert terminal.phase == "completed"
    assert (await client.status(command.run_id)) == terminal
    await stream.aclose()
    await client.close()


async def test_local_client_emergency_stop_preempts_blocked_skill(tmp_path) -> None:
    started = asyncio.Event()
    calls: list[tuple[str, str]] = []

    async def handler(_ctx, _arguments):
        started.set()
        await asyncio.Event().wait()

    async def stop(robot_id: str, reason: str) -> None:
        calls.append(("emergency", robot_id))
        assert reason == "operator estop"

    async def cancel_model(run_id: str) -> None:
        calls.append(("model_cancel", run_id))

    registry = SkillRegistry()
    registry.register(Skill("wait", "Wait.", {}, handler, resources=("arm",)))
    store = FileRunStore(tmp_path / "runs")
    client = SkillWorker(
        registry,
        run_store=store,
        cancel_model=cancel_model,
        emergency_stop=stop,
    )
    await client.submit(_command("wait", {}))
    await started.wait()

    await client.emergency_stop("mock0", reason="operator estop")
    for _ in range(20):
        event = await client.status("run-1")
        if event is not None and event.phase == "cancelled":
            break
        await asyncio.sleep(0)

    assert calls == [("emergency", "mock0"), ("model_cancel", "run-1")]
    assert event is not None
    assert event.phase == "cancelled"
    await client.close()


async def test_event_projection_never_blocks_durable_execution(tmp_path) -> None:
    projection_started = asyncio.Event()
    release_projection = asyncio.Event()
    projected: list[SkillEvent] = []
    health: list[dict[str, int]] = []
    store = FileRunStore(tmp_path / "runs")

    async def handler(_ctx, _arguments):
        return SkillResult(True, "done", "completed")

    async def project(event: SkillEvent) -> None:
        assert event in store.events(event.run_id)
        projection_started.set()
        await release_projection.wait()
        projected.append(event)

    registry = SkillRegistry()
    registry.register(Skill("wait", "Wait.", {}, handler))
    client = SkillWorker(
        registry,
        run_store=store,
        project_event=project,
        subscriber_queue_size=1,
        projection_health=lambda stats: health.append(dict(stats)),
    )
    await client.submit(_command("wait", {}))
    await projection_started.wait()

    for _ in range(20):
        terminal = store.latest_event("run-1")
        if terminal is not None and terminal.phase == "completed":
            break
        await asyncio.sleep(0)

    assert terminal is not None
    assert terminal.phase == "completed"
    release_projection.set()
    await asyncio.sleep(0)
    await client.close()

    assert health
    assert max(item["dropped"] for item in health) >= 1
    assert client.projection_stats["failed"] == 0
