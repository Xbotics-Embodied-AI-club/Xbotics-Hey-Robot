from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from hey_robot.bus.in_memory import InMemoryBusClient, InMemoryBusHub
from hey_robot.protocol import (
    Envelope,
    SkillControl,
    SkillResult as LegacyProtocolSkillResult,
)
from hey_robot.protocol.messages import from_payload, to_payload
from hey_robot.skill_os.base import (
    BaseSkill,
    SkillResult as LegacySkillResult,
    SkillSpec,
)
from hey_robot.skill_os.context import SkillContext as LegacySkillContext
from hey_robot.skills import (
    ResourceManager,
    Skill,
    SkillCancel,
    SkillCommand,
    SkillEvent,
    SkillRegistry,
    SkillResult,
    SkillRunner,
)
from hey_robot.skills.transport import (
    LegacySkillWorkerBridge,
    LocalSkillClient,
    NatsSkillClient,
    NatsSkillWorker,
    adapt_legacy_skill,
    to_legacy_result,
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


async def test_nested_skill_reuses_root_resource_owner() -> None:
    registry = SkillRegistry()
    calls: list[str] = []

    async def child(_ctx, _arguments):
        calls.append("child")
        return SkillResult(True, "child", "completed")

    async def parent(ctx, _arguments):
        calls.append("parent")
        return await ctx.run("child")

    registry.register(Skill("child", "Child.", {}, child, resources=("arm",)))
    registry.register(Skill("parent", "Parent.", {}, parent, resources=("arm",)))

    result = await asyncio.wait_for(
        _runner(registry, EventSink()).execute(_command("parent", {})), timeout=0.5
    )

    assert result.success is True
    assert calls == ["parent", "child"]


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


async def test_local_client_submits_once_and_streams_terminal_event() -> None:
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
    client = LocalSkillClient(registry)
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


async def test_legacy_skill_adapter_isolated_at_the_worker_boundary() -> None:
    class LegacyEchoSkill(BaseSkill):
        spec = SkillSpec(
            name="legacy_echo",
            description="Echo a label.",
            input_schema={
                "type": "object",
                "properties": {"label": {"type": "string"}},
                "required": ["label"],
            },
            required_resources=("camera",),
            driver_primitives=("inspect_scene",),
            required_model_service="vision",
            timeout_sec=12.0,
        )

        async def execute(self, ctx, arguments):
            assert isinstance(ctx, LegacySkillContext)
            return LegacySkillResult(True, arguments["label"], data={"legacy": True})

    adapted = adapt_legacy_skill(
        LegacyEchoSkill(), context_factory=lambda _context: LegacySkillContext()
    )
    registry = SkillRegistry()
    registry.register(adapted)
    result = await _runner(registry, EventSink()).execute(
        _command("legacy_echo", {"label": "cup"})
    )

    assert result.success is True
    assert result.data == {"legacy": True}
    assert adapted.resources == ("camera",)
    assert adapted.required_actions == ("inspect_scene",)
    assert adapted.required_models == ("vision",)


async def test_nats_transport_runs_on_a_worker_and_streams_events() -> None:
    registry = SkillRegistry()

    async def handler(_ctx, arguments):
        return SkillResult(True, f"observed {arguments['target']}", "completed")

    registry.register(
        Skill(
            "inspect",
            "Inspect a target.",
            {"type": "object", "properties": {"target": {"type": "string"}}},
            handler,
        )
    )
    hub = InMemoryBusHub()
    harness_bus = InMemoryBusClient(hub)
    worker_bus = InMemoryBusClient(hub)
    await harness_bus.connect()
    await worker_bus.connect()
    client = NatsSkillClient(harness_bus)
    worker = NatsSkillWorker(worker_bus, registry)
    await worker.start()
    await client.start()
    stream = client.events()
    first_event = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)

    await client.submit(_command("inspect", {"target": "desk"}))
    assert (await first_event).phase == "accepted"
    terminal = await anext(stream)
    while terminal.phase not in {"completed", "failed", "cancelled"}:
        terminal = await anext(stream)

    assert terminal.phase == "completed"
    assert terminal.result is not None
    assert terminal.result.success
    assert await client.status(terminal.run_id) == terminal
    await stream.aclose()
    await client.close()
    await worker.close()
    await harness_bus.close()
    await worker_bus.close()


async def test_legacy_bridge_translates_controller_result_to_new_event() -> None:
    hub = InMemoryBusHub()
    harness_bus = InMemoryBusClient(hub)
    bridge_bus = InMemoryBusClient(hub)
    controller_bus = InMemoryBusClient(hub)
    await harness_bus.connect()
    await bridge_bus.connect()
    await controller_bus.connect()
    client = NatsSkillClient(harness_bus)
    bridge = LegacySkillWorkerBridge(bridge_bus)
    await bridge.start()
    await client.start()
    stream = client.events()
    first_event = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    command = _command("inspect_scene", {"question": "desk"})

    await client.submit(command)
    assert (await first_event).phase == "accepted"
    await controller_bus.publish(
        "skill.result",
        to_payload(
            LegacyProtocolSkillResult(
                envelope=command.envelope,
                skill_id=command.run_id,
                name=command.name,
                status="completed",
                success=True,
                summary="scene=desk",
            )
        ),
    )
    event = await anext(stream)
    while event.phase != "completed":
        event = await anext(stream)

    assert event.result is not None
    assert event.result.success
    await stream.aclose()
    await client.close()
    await bridge.close()
    await harness_bus.close()
    await bridge_bus.close()
    await controller_bus.close()


async def test_legacy_bridge_maps_interrupted_result_to_cancelled_event() -> None:
    hub = InMemoryBusHub()
    harness_bus = InMemoryBusClient(hub)
    bridge_bus = InMemoryBusClient(hub)
    controller_bus = InMemoryBusClient(hub)
    await harness_bus.connect()
    await bridge_bus.connect()
    await controller_bus.connect()
    client = NatsSkillClient(harness_bus)
    bridge = LegacySkillWorkerBridge(bridge_bus)
    await bridge.start()
    await client.start()
    stream = client.events()
    first_event = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    command = _command("move_base", {"direction": "forward"})

    await client.submit(command)
    assert (await first_event).phase == "accepted"
    await controller_bus.publish(
        "skill.result",
        to_payload(
            LegacyProtocolSkillResult(
                envelope=command.envelope,
                skill_id=command.run_id,
                name=command.name,
                status="interrupted",
                success=False,
                summary="interrupted by user",
            )
        ),
    )
    event = await anext(stream)
    while event.phase != "cancelled":
        event = await anext(stream)

    assert event.result is not None
    assert event.result.status == "cancelled"
    await stream.aclose()
    await client.close()
    await bridge.close()
    await harness_bus.close()
    await bridge_bus.close()
    await controller_bus.close()


async def test_legacy_bridge_maps_cancel_to_old_skill_control() -> None:
    hub = InMemoryBusHub()
    harness_bus = InMemoryBusClient(hub)
    bridge_bus = InMemoryBusClient(hub)
    controller_bus = InMemoryBusClient(hub)
    await harness_bus.connect()
    await bridge_bus.connect()
    await controller_bus.connect()
    controls: list[SkillControl] = []

    async def on_control(_topic, payload):
        controls.append(from_payload(SkillControl, payload))

    await controller_bus.subscribe(["skill.control"], on_control)
    client = NatsSkillClient(harness_bus)
    bridge = LegacySkillWorkerBridge(bridge_bus)
    await bridge.start()
    await client.start()
    command = _command("move_base", {"direction": "forward"})

    await client.submit(command)
    await harness_bus.publish(
        "skill.cancel",
        to_payload(
            SkillCancel(
                envelope=command.envelope,
                run_id=command.run_id,
                reason="user stopped task",
            )
        ),
    )

    assert controls == [
        SkillControl(
            envelope=command.envelope,
            control_id=f"cancel_{command.run_id}",
            action="interrupt",
            target_skill_id=command.run_id,
            task_id=command.task_id,
            reason="user stopped task",
        )
    ]
    await client.close()
    await bridge.close()
    await harness_bus.close()
    await bridge_bus.close()
    await controller_bus.close()


def test_new_terminal_event_converts_to_legacy_protocol_result() -> None:
    event = SkillEvent(
        envelope=Envelope(robot_id="mock0"),
        run_id="run-1",
        sequence=3,
        name="move_base",
        phase="cancelled",
        timestamp=1.0,
        result=SkillResult(
            False,
            "cancelled by user",
            "cancelled",
            data={"detail": "operator stop"},
        ),
    )

    legacy = to_legacy_result(event)

    assert legacy.skill_id == "run-1"
    assert legacy.status == "interrupted"
    assert legacy.success is False
    assert legacy.metadata == {"detail": "operator stop"}
