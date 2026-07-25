from __future__ import annotations

from typing import Any

from hey_robot.cognition.runtime.agent_task_store import AgentTaskStore
from hey_robot.cognition.tools.models import CompleteTaskProposal
from hey_robot.cognition.tools.registry import ToolDependencies, ToolRegistry
from hey_robot.cognition.tools.skill_tools import SkillCallProposal
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
    tools = ToolRegistry(ToolDependencies(catalog))

    proposal = tools.prepare("navigate_once", {})

    assert proposal.intent_kind == "skill"
    assert proposal.skill_name == "navigate_once"


def test_complete_task_only_requires_recap() -> None:
    tools = ToolRegistry(ToolDependencies(SkillList(())))

    proposal = tools.prepare(
        "complete_task",
        {"recap": "已经进入并观察。"},
    )

    assert isinstance(proposal, CompleteTaskProposal)
    assert proposal.recap == "已经进入并观察。"


def test_sustained_task_completion_requires_a_successful_step(tmp_path) -> None:
    store = AgentTaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(
        session_key="session-1",
        envelope=Envelope(robot_id="sim_robot"),
        objective="进入门廊并观察里面有什么",
    )
    check = store.complete_task(task.task_id, recap="尚未执行。")
    assert not check.accepted

    store.add_step(
        task.task_id,
        SkillCallProposal("skill", "move_base", "move", {"direction": "forward"}),
        ToolOutcome("completed", "Base motion completed.", operation_id="move1"),
    )

    check = store.complete_task(task.task_id, recap="已经进入。")

    assert check.accepted
    assert store.active_task("session-1") is None
    store.close()


def test_task_store_persists_pending_run_and_ignores_replayed_events(tmp_path) -> None:
    store = AgentTaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(
        session_key="session-1",
        envelope=Envelope(robot_id="sim_robot"),
        objective="inspect the desk",
    )
    proposal = SkillCallProposal("observation", "inspect_scene", "inspect", {})
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
