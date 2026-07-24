from __future__ import annotations

import asyncio

import pytest

from hey_robot.persistence import FileRunStore
from hey_robot.protocol import Envelope
from hey_robot.skills import (
    ResourceManager,
    Skill,
    SkillCommand,
    SkillEvent,
    SkillRegistry,
    SkillResult,
    SkillWorker,
)


def _event(
    command: SkillCommand,
    sequence: int,
    phase: str,
    *,
    result: SkillResult | None = None,
) -> SkillEvent:
    return SkillEvent(
        envelope=command.envelope,
        run_id=command.run_id,
        sequence=sequence,
        name=command.name,
        phase=phase,  # type: ignore[arg-type]
        timestamp=float(sequence),
        result=result,
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
async def test_skill_worker_deduplicates_identical_submit(tmp_path) -> None:
    calls = 0
    release = asyncio.Event()

    async def handler(_ctx, _arguments):
        nonlocal calls
        calls += 1
        await release.wait()
        return SkillResult(True, "done", "completed")

    registry = SkillRegistry()
    registry.register(Skill("inspect", "Inspect.", {}, handler))
    worker = SkillWorker(registry, run_store=FileRunStore(tmp_path / "runs"))
    await worker.start()
    command = _command()

    assert await worker.submit(command) == "run-1"
    assert await worker.submit(command) == "run-1"
    await asyncio.sleep(0)
    release.set()
    await asyncio.sleep(0)

    assert calls == 1
    await worker.close()


@pytest.mark.parametrize("phase", ["accepted", "running"])
async def test_restarted_worker_marks_persisted_non_terminal_run_lost(
    tmp_path, phase
) -> None:
    run_store = FileRunStore(tmp_path / "runs")
    command = _command()
    run_store.record_submission(command)
    run_store.append_event(_event(command, 1, "accepted"))
    if phase == "running":
        run_store.append_event(_event(command, 2, "running"))
    worker = SkillWorker(SkillRegistry(), run_store=run_store)

    lost = await worker.status(command.run_id)
    replay = await worker.status(command.run_id)

    assert lost is not None
    assert lost.phase == "failed"
    assert lost.result is not None
    assert lost.result.failure_mode == "execution_lost"
    assert replay == lost
    assert [event.phase for event in run_store.events(command.run_id)][-1] == "failed"


async def test_restarted_worker_marks_submission_without_first_event_lost(
    tmp_path,
) -> None:
    run_store = FileRunStore(tmp_path / "runs")
    command = _command()
    run_store.record_submission(command)
    worker = SkillWorker(SkillRegistry(), run_store=run_store)

    event = await worker.status(command.run_id)

    assert event is not None
    assert event.sequence == 1
    assert event.result is not None
    assert event.result.failure_mode == "execution_lost"


async def test_concurrent_restart_status_writes_one_lost_terminal(tmp_path) -> None:
    run_store = FileRunStore(tmp_path / "runs")
    command = _command()
    run_store.record_submission(command)
    run_store.append_event(_event(command, 1, "running"))
    worker = SkillWorker(SkillRegistry(), run_store=run_store)

    first, second = await asyncio.gather(
        worker.status(command.run_id),
        worker.status(command.run_id),
    )

    assert first == second
    assert [event.phase for event in run_store.events(command.run_id)] == [
        "running",
        "failed",
    ]


@pytest.mark.parametrize("phase", ["completed", "failed", "cancelled"])
async def test_restarted_worker_returns_persisted_terminal_without_reexecution(
    tmp_path, phase
) -> None:
    calls = 0

    async def handler(_ctx, _arguments):
        nonlocal calls
        calls += 1
        return SkillResult(True, "unexpected", "completed")

    registry = SkillRegistry()
    registry.register(Skill("inspect", "Inspect.", {}, handler))
    run_store = FileRunStore(tmp_path / "runs")
    command = _command()
    run_store.record_submission(command)
    result = SkillResult(
        phase == "completed",
        phase,
        phase,  # type: ignore[arg-type]
        failure_mode=None if phase == "completed" else phase,
    )
    terminal = _event(command, 1, phase, result=result)
    run_store.append_event(terminal)
    worker = SkillWorker(registry, run_store=run_store)

    assert await worker.status(command.run_id) == terminal
    assert await worker.submit(command) == command.run_id
    await worker.start()
    await asyncio.sleep(0)

    assert calls == 0
    assert not worker._tasks
    await worker.close()


async def test_worker_removes_completed_tasks_and_rejects_restart(tmp_path) -> None:
    completed = asyncio.Event()

    async def handler(_ctx, _arguments):
        completed.set()
        return SkillResult(True, "done", "completed")

    registry = SkillRegistry()
    registry.register(Skill("inspect", "Inspect.", {}, handler))
    worker = SkillWorker(registry, run_store=FileRunStore(tmp_path / "runs"))
    await worker.start()
    await worker.submit(_command())
    await completed.wait()
    for _ in range(10):
        if not worker._tasks:
            break
        await asyncio.sleep(0)

    assert not worker._tasks
    assert not worker._pending
    await worker.close()
    with pytest.raises(RuntimeError, match="closed"):
        await worker.start()
    with pytest.raises(RuntimeError, match="closed"):
        await worker.submit(_command("run-2"))


async def test_worker_releases_per_run_state_after_many_completed_runs(
    tmp_path,
) -> None:
    completed = asyncio.Event()
    completed_count = 0

    async def handler(_ctx, _arguments):
        nonlocal completed_count
        completed_count += 1
        if completed_count == 50:
            completed.set()
        return SkillResult(True, "done", "completed")

    registry = SkillRegistry()
    registry.register(Skill("inspect", "Inspect.", {}, handler))
    worker = SkillWorker(registry, run_store=FileRunStore(tmp_path / "runs"))
    await worker.start()

    for index in range(50):
        await worker.submit(_command(f"run-{index}"))
    await asyncio.wait_for(completed.wait(), timeout=5)
    for _ in range(10):
        if not worker._tasks:
            break
        await asyncio.sleep(0)

    assert not worker._tasks
    assert not worker._pending
    assert not worker._runner._sequences
    assert not worker._runner._cancelled
    await worker.close()


async def test_worker_slow_subscriber_drops_old_progress_without_blocking(
    tmp_path,
) -> None:
    worker = SkillWorker(
        SkillRegistry(),
        run_store=FileRunStore(tmp_path / "runs"),
        subscriber_queue_size=2,
    )
    command = _command()
    stream = worker.events()
    first = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    await worker._emit(_event(command, 1, "accepted"))
    assert (await first).sequence == 1

    await worker._emit(_event(command, 2, "running"))
    await worker._emit(_event(command, 3, "progress"))
    await worker._emit(_event(command, 4, "progress"))

    assert (await anext(stream)).sequence == 3
    assert (await anext(stream)).sequence == 4
    await stream.aclose()
    await worker.close()


async def test_worker_cancel_propagates_to_model_and_releases_resources(
    tmp_path,
) -> None:
    started = asyncio.Event()
    model_cancels: list[str] = []
    resources = ResourceManager()

    async def handler(_ctx, _arguments):
        started.set()
        await asyncio.Event().wait()

    async def cancel_model(run_id: str) -> None:
        model_cancels.append(run_id)

    registry = SkillRegistry()
    registry.register(Skill("inspect", "Inspect.", {}, handler, resources=("arm",)))
    store = FileRunStore(tmp_path / "runs")
    worker = SkillWorker(
        registry,
        resources=resources,
        run_store=store,
        cancel_model=cancel_model,
    )
    await worker.start()
    await worker.submit(_command())
    await started.wait()

    await worker.cancel("run-1", reason="operator cancel")
    for _ in range(20):
        terminal = store.latest_event("run-1")
        if terminal is not None and terminal.phase == "cancelled":
            break
        await asyncio.sleep(0)
    await worker.cancel("run-1", reason="duplicate")

    assert model_cancels == ["run-1"]
    assert [event.phase for event in store.events("run-1")].count("cancelled") == 1
    assert not resources._owners
    await worker.close()
