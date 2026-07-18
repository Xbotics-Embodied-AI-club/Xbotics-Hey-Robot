from __future__ import annotations

from hey_robot.cognition.runtime.agent_task_store import AgentTaskStore
from hey_robot.cognition.tools.robot import (
    CompleteTaskProposal,
    ToolDependencies,
    ToolRegistry,
)
from hey_robot.protocol import (
    ActionProposal,
    Envelope,
    ToolOutcome,
)
from hey_robot.skill_os.base import SkillCatalog, SkillSpec


def test_conversation_skill_does_not_upgrade_by_category() -> None:
    catalog = SkillCatalog(
        (
            SkillSpec(
                name="navigate_once",
                description="bounded navigation skill",
                category="navigation",
                input_schema={"type": "object", "properties": {}},
            ),
        )
    )
    tools = ToolRegistry(ToolDependencies(catalog))

    proposal = tools.proposal("request_skill", {"skill": "navigate_once"})

    assert proposal.intent_kind == "skill"
    assert proposal.skill_name == "navigate_once"


def test_complete_task_requires_evidence_ids() -> None:
    tools = ToolRegistry(ToolDependencies(SkillCatalog(())))

    proposal = tools.proposal(
        "complete_task",
        {"recap": "已经进入并观察。", "evidence_ids": ["observation:op1"]},
    )

    assert isinstance(proposal, CompleteTaskProposal)
    assert proposal.evidence_ids == ("observation:op1",)


def test_sustained_task_completion_requires_post_motion_observation(
    tmp_path,
) -> None:
    store = AgentTaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(
        session_key="session-1",
        envelope=Envelope(robot_id="sim_robot"),
        objective="进入门廊并观察里面有什么",
    )
    move = store.add_step(
        task.task_id,
        ActionProposal("skill", "move_base", "move", {"direction": "forward"}),
        ToolOutcome("completed", "Base motion completed.", operation_id="move1"),
    )

    check = store.complete_task(
        task.task_id,
        recap="已经进入。",
        evidence_ids=move.evidence_ids,
    )

    assert not check.accepted
    observation = store.add_step(
        task.task_id,
        ActionProposal(
            "observation", "inspect_scene", "inspect", {"question": "inside"}
        ),
        ToolOutcome("completed", "里面有桌椅。", operation_id="obs1"),
    )
    check = store.complete_task(
        task.task_id,
        recap="看到里面有桌椅。",
        evidence_ids=observation.evidence_ids,
    )

    assert check.accepted
    assert store.active_task("session-1") is None
    store.close()
