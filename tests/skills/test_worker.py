from __future__ import annotations

import asyncio

import pytest

from hey_robot.persistence import FileRunStore
from hey_robot.protocol import Envelope
from hey_robot.skills import (
    Skill,
    SkillCommand,
    SkillRegistry,
    SkillResult,
    SkillWorker,
)


def _command(run_id: str = "run-1") -> SkillCommand:
    return SkillCommand(
        envelope=Envelope(robot_id="mock0"),
        run_id=run_id,
        task_id="task-1",
        robot_id="mock0",
        name="inspect",
        arguments={"target": "desk"},
    )


@pytest.mark.asyncio
async def test_skill_worker_queues_command_and_persists_run_events(tmp_path) -> None:
    calls: list[dict] = []

    async def handler(_ctx, arguments):
        calls.append(arguments)
        return SkillResult(True, "observed desk", "completed")

    registry = SkillRegistry()
    registry.register(
        Skill(
            "inspect",
            "Inspect a target.",
            {"type": "object", "properties": {"target": {"type": "string"}}},
            handler,
        )
    )
    run_store = FileRunStore(tmp_path / "runs")
    worker = SkillWorker(registry, run_store=run_store)
    await worker.start()
    stream = worker.events()
    first_event = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)

    await worker.submit(_command())
    assert (await first_event).phase == "accepted"
    event = await anext(stream)
    while event.phase != "completed":
        event = await anext(stream)

    assert calls == [{"target": "desk"}]
    assert event.result is not None
    assert run_store.result("run-1") == event.result
    assert [item.phase for item in run_store.events("run-1")] == [
        "accepted",
        "running",
        "completed",
    ]
    await stream.aclose()
    await worker.close()


@pytest.mark.asyncio
async def test_skill_worker_deduplicates_identical_submit() -> None:
    calls = 0
    release = asyncio.Event()

    async def handler(_ctx, _arguments):
        nonlocal calls
        calls += 1
        await release.wait()
        return SkillResult(True, "done", "completed")

    registry = SkillRegistry()
    registry.register(Skill("inspect", "Inspect.", {}, handler))
    worker = SkillWorker(registry)
    await worker.start()
    command = _command()

    assert await worker.submit(command) == "run-1"
    assert await worker.submit(command) == "run-1"
    await asyncio.sleep(0)
    release.set()
    await asyncio.sleep(0)

    assert calls == 1
    await worker.close()
