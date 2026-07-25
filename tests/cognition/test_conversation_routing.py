from __future__ import annotations

from typing import Any

from hey_robot.cognition.runtime.agent_task_store import AgentTaskStore
from hey_robot.cognition.tools.models import PhysicalToolCall
from hey_robot.cognition.tools.registry import ToolDependencies, ToolRegistry
from hey_robot.protocol import Envelope, ToolOutcome
from hey_robot.skills.models import Skill, SkillResult


class SkillList:
    def __init__(self, skills: tuple[Skill, ...]) -> None:
        self._skills = {skill.name: skill for skill in skills}

    def get(self, name: str) -> Skill:
        return self._skills[name]

    def list(self) -> tuple[Skill, ...]:
        return tuple(self._skills.values())


async def _noop(*_args: Any, **_kwargs: Any) -> SkillResult:
    return SkillResult(True, "ok", "completed")


def test_conversation_skill_does_not_upgrade_by_category() -> None:
    catalog = SkillList(
        (
            Skill(
                name="navigate_once",
                description="bounded navigation skill",
                parameters={"type": "object", "properties": {}},
                handler=_noop,
            ),
        )
    )
    tools = ToolRegistry(ToolDependencies(catalog.list()))

    proposal = tools.prepare("navigate_once", {})

    assert proposal == PhysicalToolCall("navigate_once", {})


def test_complete_task_is_not_model_visible() -> None:
    tools = ToolRegistry(ToolDependencies(()))

    assert "complete_task" not in tools.names


def test_durable_task_closes_on_final_assistant_text(tmp_path) -> None:
    store = AgentTaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(
        session_key="session-1",
        envelope=Envelope(robot_id="sim_robot"),
        objective="进入门廊并观察里面有什么",
    )
    store.close_task(task.task_id, recap="需要更多信息。")

    assert store.active_task("session-1") is None
    assert store.task(task.task_id).final_recap == "需要更多信息。"  # type: ignore[union-attr]
    store.close()


def test_failed_physical_step_closes_task_as_failed(tmp_path) -> None:
    store = AgentTaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(
        session_key="session-1",
        envelope=Envelope(robot_id="sim_robot"),
        objective="turn left",
    )
    store.add_step(
        task.task_id,
        PhysicalToolCall("turn_base", {"direction": "left", "angle_deg": 90}),
        ToolOutcome("failed", "turn failed"),
    )

    store.close_task(task.task_id, recap="无法完成左转。")

    closed = store.task(task.task_id)
    assert closed is not None
    assert closed.status == "failed"
    assert closed.final_recap == "无法完成左转。"
    store.close()


def test_task_store_persists_pending_run_and_ignores_replayed_events(tmp_path) -> None:
    store = AgentTaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(
        session_key="session-1",
        envelope=Envelope(robot_id="sim_robot"),
        objective="inspect the desk",
    )
    proposal = PhysicalToolCall("inspect_scene", {})
    pending = store.add_pending_step(
        task.task_id,
        proposal,
        run_id="run-1",
        tool_call_id="call-1",
    )

    assert pending.status == "pending"
    assert pending.outcome.status == "accepted"
    resolved = store.resolve_pending_step(
        "run-1",
        outcome=ToolOutcome("completed", "desk observed", operation_id="run-1"),
        status="completed",
        event_sequence=3,
    )

    assert resolved is not None
    assert resolved.status == "completed"
    assert resolved.completed_at is not None
    assert (
        store.resolve_pending_step(
            "run-1",
            outcome=ToolOutcome("failed", "stale"),
            status="failed",
            event_sequence=2,
        )
        is None
    )
    store.close()
