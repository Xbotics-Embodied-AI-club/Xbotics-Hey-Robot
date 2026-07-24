from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from hey_robot.cognition.runtime.agent_task_store import AgentTaskStore
from hey_robot.cognition.runtime.harness_store import HarnessStore
from hey_robot.cognition.runtime.task_coordinator import TaskCoordinator
from hey_robot.cognition.tools.skill_tools import SkillCallProposal
from hey_robot.protocol import Envelope, ToolOutcome
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
        proposal=SkillCallProposal("observation", "inspect_scene", "inspect", {}),
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


def test_task_store_migrates_legacy_task_route_columns(tmp_path) -> None:
    path = tmp_path / "legacy.sqlite3"
    db = sqlite3.connect(path)
    db.execute(
        """
        CREATE TABLE sustained_tasks (
            task_id TEXT PRIMARY KEY,
            session_key TEXT NOT NULL,
            robot_id TEXT NOT NULL,
            objective TEXT NOT NULL,
            ui_summary TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            step_count INTEGER NOT NULL DEFAULT 0,
            continuation_count INTEGER NOT NULL DEFAULT 0,
            deadline_at REAL,
            last_error TEXT,
            final_recap TEXT
        )
        """
    )
    db.execute(
        """
        CREATE TABLE task_steps (
            step_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            proposal_json TEXT NOT NULL,
            outcome_json TEXT NOT NULL,
            started_at REAL NOT NULL,
            completed_at REAL,
            evidence_json TEXT NOT NULL
        )
        """
    )
    db.execute(
        """
        INSERT INTO sustained_tasks (
            task_id, session_key, robot_id, objective, ui_summary, status,
            created_at, updated_at
        ) VALUES ('task-1', 'session-1', 'robot', 'inspect', '', 'active', 1.0, 1.0)
        """
    )
    db.commit()
    db.close()

    store = AgentTaskStore(path)
    task = store.active_task("session-1")
    envelope = store.task_envelope("task-1")

    assert task is not None
    assert task.channel is None
    assert envelope is not None
    assert envelope.robot_id == "robot"
    store.close()


def test_task_store_lists_only_active_run_ids(tmp_path) -> None:
    store = AgentTaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(
        session_key="session", envelope=Envelope(robot_id="robot"), objective="inspect"
    )
    pending = store.add_pending_step(
        task.task_id,
        SkillCallProposal("skill", "move_base", "move", {}),
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
        SkillCallProposal("skill", "turn_base", "turn", {}),
        run_id="run-active",
        tool_call_id="call-2",
    )

    assert next_pending.run_id == "run-active"
    assert store.active_run_ids(task.task_id) == ("run-active",)
    store.close()


def test_task_store_start_skill_step_matches_target_api(tmp_path) -> None:
    store = AgentTaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(
        session_key="session", envelope=Envelope(robot_id="robot"), objective="inspect"
    )

    step = store.start_skill_step(
        task.task_id,
        run_id="run-1",
        tool_call_id="call-1",
        tool_name="inspect_scene",
        arguments={"question": "desk"},
    )
    columns = {
        str(row[1])
        for row in store._db.execute("PRAGMA table_info(task_steps)").fetchall()
    }

    assert step.status == "pending"
    assert step.proposal.skill_name == "inspect_scene"
    assert step.proposal.objective == "desk"
    assert "arguments_json" in columns
    assert (
        store.apply_skill_event(
            "run-1",
            outcome=ToolOutcome("completed", "desk observed"),
            status="completed",
            event_sequence=1,
        ).status
        == "completed"
    )
    store.close()


def test_harness_store_alias_uses_agent_task_store(tmp_path) -> None:
    store = HarnessStore(tmp_path / "tasks.sqlite3")

    task = store.create_task(
        session_key="session", envelope=Envelope(robot_id="robot"), objective="inspect"
    )

    assert task.task_id.startswith("task_")
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
        proposal=SkillCallProposal("observation", "inspect_scene", "inspect", {}),
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
    pending = store.start_skill_step(
        task.task_id,
        run_id="run-reconcile",
        tool_call_id="call-1",
        tool_name="inspect_scene",
        arguments={},
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

    reconciled = await coordinator.reconcile_active_runs()

    assert [step.run_id for step in reconciled] == [pending.run_id]
    assert reconciled[0].status == "completed"
    assert store.active_run_ids(task.task_id) == ()
    store.close()
