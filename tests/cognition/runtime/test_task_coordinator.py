from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from hey_robot.cognition.runtime.agent_task_store import AgentTaskStore
from hey_robot.cognition.runtime.task_coordinator import TaskCoordinator
from hey_robot.cognition.tools.models import PhysicalToolCall
from hey_robot.persistence import FileRunStore
from hey_robot.protocol import Envelope, ToolOutcome
from hey_robot.skills import SkillRegistry, SkillWorker
from hey_robot.skills.models import SkillCommand, SkillEvent, SkillResult


@dataclass
class Client:
    commands: list[SkillCommand] = field(default_factory=list)
    statuses: dict[str, SkillEvent] = field(default_factory=dict)

    async def submit(self, command: SkillCommand) -> str:
        self.commands.append(command)
        return command.run_id

    async def status(self, run_id: str) -> SkillEvent | None:
        return self.statuses.get(run_id)


class FailingClient(Client):
    async def submit(self, command: SkillCommand) -> str:
        self.commands.append(command)
        raise ConnectionError("skill worker unavailable")


async def test_coordinator_persists_before_submit_and_applies_terminal_event(
    tmp_path,
) -> None:
    store = AgentTaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(
        session_key="session", envelope=Envelope(robot_id="robot"), objective="inspect"
    )
    client = Client()
    coordinator = TaskCoordinator(store, client)  # type: ignore[arg-type]
    step = await coordinator.submit(
        task_id=task.task_id,
        proposal=PhysicalToolCall("inspect_scene", {}),
        envelope=Envelope(robot_id="robot"),
        tool_call_id="call-1",
    )

    assert step.status == "pending"
    assert client.commands[0].run_id == step.run_id
    result = SkillResult(True, "desk observed", "completed", evidence_ids=("e1",))
    resolved = coordinator.apply(
        SkillEvent(
            envelope=Envelope(robot_id="robot"),
            run_id=step.run_id or "",
            sequence=3,
            name="inspect_scene",
            phase="completed",
            timestamp=0.0,
            result=result,
        )
    )

    assert resolved is not None
    assert resolved.status == "completed"
    assert resolved.outcome.data["evidence_ids"] == ["e1"]
    store.close()


async def test_coordinator_rejects_second_concurrent_skill_run(tmp_path) -> None:
    store = AgentTaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(
        session_key="session", envelope=Envelope(robot_id="robot"), objective="move"
    )
    store.add_pending_step(
        task.task_id,
        PhysicalToolCall("move_base", {}),
        run_id="run-active",
        tool_call_id="call-1",
    )
    client = Client()
    coordinator = TaskCoordinator(store, client)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="already has an active skill run"):
        await coordinator.submit(
            task_id=task.task_id,
            proposal=PhysicalToolCall("turn_base", {}),
            envelope=Envelope(robot_id="robot"),
            tool_call_id="call-2",
        )

    assert client.commands == []
    store.close()


async def test_environment_done_converges_step_and_task_terminal_state(
    tmp_path,
) -> None:
    store = AgentTaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(
        session_key="session", envelope=Envelope(robot_id="robot"), objective="pick"
    )
    coordinator = TaskCoordinator(store, Client())  # type: ignore[arg-type]
    step = await coordinator.submit(
        task_id=task.task_id,
        proposal=PhysicalToolCall("manipulate", {}),
        envelope=Envelope(robot_id="robot"),
        tool_call_id="call-1",
    )

    resolved = coordinator.apply(
        SkillEvent(
            envelope=Envelope(robot_id="robot"),
            run_id=step.run_id or "",
            sequence=3,
            name="manipulate",
            phase="completed",
            timestamp=0.0,
            result=SkillResult(
                True,
                "environment success",
                "completed",
                data={"termination_reason": "environment_done"},
            ),
        )
    )

    assert resolved is not None
    assert resolved.status == "completed"
    completed = store.task(task.task_id)
    assert completed is not None
    assert completed.status == "completed"
    assert completed.final_recap == "environment success"
    store.close()


def test_task_store_lists_only_active_run_ids(tmp_path) -> None:
    store = AgentTaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(
        session_key="session", envelope=Envelope(robot_id="robot"), objective="inspect"
    )
    pending = store.add_pending_step(
        task.task_id,
        PhysicalToolCall("move_base", {}),
        run_id="run-pending",
        tool_call_id="call-1",
    )
    store.resolve_pending_step(
        pending.run_id or "",
        outcome=ToolOutcome("completed", "done"),
        status="completed",
        event_sequence=1,
    )

    next_pending = store.add_pending_step(
        task.task_id,
        PhysicalToolCall("turn_base", {}),
        run_id="run-active",
        tool_call_id="call-2",
    )

    assert next_pending.run_id == "run-active"
    assert store.active_run_ids(task.task_id) == ("run-active",)
    store.close()


async def test_coordinator_marks_step_failed_when_submit_is_rejected(tmp_path) -> None:
    store = AgentTaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(
        session_key="session", envelope=Envelope(robot_id="robot"), objective="inspect"
    )
    client = FailingClient()
    coordinator = TaskCoordinator(store, client)  # type: ignore[arg-type]

    step = await coordinator.submit(
        task_id=task.task_id,
        proposal=PhysicalToolCall("inspect_scene", {}),
        envelope=Envelope(robot_id="robot"),
        tool_call_id="call-1",
    )

    assert step.status == "failed"
    assert step.outcome.data["failure_mode"] == "transport_submit_failed"
    assert step.outcome.retryable is True
    assert store.active_run_ids(task.task_id) == ()
    store.close()


async def test_coordinator_reconciles_transport_known_terminal_event(tmp_path) -> None:
    store = AgentTaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(
        session_key="session", envelope=Envelope(robot_id="robot"), objective="inspect"
    )
    pending = store.add_pending_step(
        task.task_id,
        PhysicalToolCall("inspect_scene", {}),
        run_id="run-reconcile",
        tool_call_id="call-1",
    )
    client = Client(
        statuses={
            "run-reconcile": SkillEvent(
                envelope=Envelope(robot_id="robot"),
                run_id="run-reconcile",
                sequence=2,
                name="inspect_scene",
                phase="completed",
                timestamp=0.0,
                result=SkillResult(True, "desk observed", "completed"),
            )
        }
    )
    coordinator = TaskCoordinator(store, client)  # type: ignore[arg-type]

    reconciled = await coordinator.reconcile_active_run_results()

    assert [applied.step.run_id for _event, applied in reconciled] == [pending.run_id]
    assert reconciled[0][1].step.status == "completed"
    assert store.active_run_ids(task.task_id) == ()
    store.close()


async def test_coordinator_reconciles_restarted_worker_without_replaying_action(
    tmp_path,
) -> None:
    tasks = AgentTaskStore(tmp_path / "tasks.sqlite3")
    task = tasks.create_task(
        session_key="session", envelope=Envelope(robot_id="robot"), objective="inspect"
    )
    run_id = "run-crashed"
    tasks.add_pending_step(
        task.task_id,
        PhysicalToolCall("inspect_scene", {}),
        run_id=run_id,
        tool_call_id="call-1",
    )
    runs = FileRunStore(tmp_path / "runs")
    command = SkillCommand(
        envelope=Envelope(robot_id="robot"),
        run_id=run_id,
        task_id=task.task_id,
        robot_id="robot",
        name="inspect_scene",
        arguments={},
    )
    runs.record_submission(command)
    runs.append_event(
        SkillEvent(
            envelope=command.envelope,
            run_id=run_id,
            sequence=1,
            name=command.name,
            phase="running",
            timestamp=1.0,
        )
    )
    restarted_worker = SkillWorker(SkillRegistry(), run_store=runs)
    coordinator = TaskCoordinator(tasks, restarted_worker)

    reconciled = await coordinator.reconcile_active_run_results()

    assert len(reconciled) == 1
    event, applied = reconciled[0]
    assert event.result is not None
    assert event.result.failure_mode == "execution_lost"
    assert applied.step.status == "failed"
    assert tasks.active_run_ids(task.task_id) == ()
    assert await coordinator.reconcile_active_run_results() == ()
    tasks.close()


def test_terminal_step_persists_wakeup_until_next_physical_receipt(tmp_path) -> None:
    path = tmp_path / "tasks.sqlite3"
    store = AgentTaskStore(path)
    task = store.create_task(
        session_key="session-1",
        envelope=Envelope(robot_id="sim_robot"),
        objective="inspect",
    )
    proposal = PhysicalToolCall("inspect_scene", {})
    store.add_pending_step(
        task.task_id, proposal, run_id="run-1", tool_call_id="call-1"
    )
    store.resolve_pending_step(
        "run-1",
        outcome=ToolOutcome("completed", "seen", operation_id="run-1"),
        status="completed",
        event_sequence=2,
    )
    store.close()

    restarted = AgentTaskStore(path)
    resumable = restarted.resumable_tasks()
    assert [item.task_id for item in resumable] == [task.task_id]
    assert restarted.resume_after_sequence(task.task_id) == 1
    restarted.add_pending_step(
        task.task_id, proposal, run_id="run-2", tool_call_id="call-2"
    )
    assert restarted.resumable_tasks() == ()
    restarted.close()


def test_pause_keeps_one_open_task(tmp_path) -> None:
    store = AgentTaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(
        session_key="session-1",
        envelope=Envelope(robot_id="sim_robot"),
        objective="put the cup in the kitchen",
    )
    store.pause_task(task.task_id, "paused")

    assert store.active_task("session-1") is None
    assert store.current_task("session-1").status == "paused"  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="\u5df2有"):
        store.create_task(
            session_key="session-1",
            envelope=Envelope(robot_id="sim_robot"),
            objective="another task",
        )

    resumed = store.resume_task(task.task_id)
    assert resumed.status == "active"
    assert store.resumable_tasks()[0].task_id == task.task_id
    store.close()
