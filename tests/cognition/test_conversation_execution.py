from __future__ import annotations

import asyncio

from hey_robot.cognition.conversation_execution import (
    RobotExecutionAdapter,
    _trusted_observation_summary,
)
from hey_robot.cognition.runtime.conversation_store import ConversationStore
from hey_robot.protocol import ActionProposal, Envelope, GoalCommand, Topics
from hey_robot.protocol.messages import from_payload
from hey_robot.skill_os.base import SkillCatalog, SkillSpec


class _Bus:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict]] = []

    async def publish(self, topic: str, payload: dict) -> None:
        self.published.append((topic, payload))


def test_navigation_proposal_creates_linked_long_horizon_goal(tmp_path) -> None:
    bus = _Bus()
    store = ConversationStore(tmp_path / "conversation.sqlite3")
    adapter = RobotExecutionAdapter(
        bus,
        Topics(),
        SkillCatalog(
            (
                SkillSpec(
                    name="navigate_to",
                    description="navigate",
                    category="navigation",
                ),
            )
        ),
        store,
        known_entities=("room:kitchen",),
    )
    envelope = Envelope(robot_id="mock0", channel="web", chat_id="chat")

    outcome = asyncio.run(
        adapter.execute(
            ActionProposal(
                "skill", "navigate_to", "去厨房", {"target": "room:kitchen"}
            ),
            envelope,
            "d1:main:web:chat",
        )
    )

    assert outcome.status == "accepted"
    assert outcome.goal_id is not None
    command = from_payload(GoalCommand, bus.published[0][1])
    assert bus.published[0][0] == Topics().goal_command
    assert command.goal_id == outcome.goal_id
    assert command.success_criteria[0].object_id == "room:kitchen"
    link = store.goal_link(outcome.goal_id)
    assert link is not None
    assert link[0] == "d1:main:web:chat"


def test_short_operation_is_submitted_to_supervisor_for_preflight(tmp_path) -> None:
    bus = _Bus()
    store = ConversationStore(tmp_path / "conversation.sqlite3")
    adapter = RobotExecutionAdapter(
        bus,
        Topics(),
        SkillCatalog(
            (SkillSpec(name="move_base", description="move", category="base"),)
        ),
        store,
        timeout_sec=0.0,
    )

    outcome = asyncio.run(
        adapter.execute(
            ActionProposal("skill", "move_base", "向前移动", {"distance_cm": 20}),
            Envelope(robot_id="mock0"),
            "d1:main:owner",
        )
    )

    assert outcome.status == "waiting"
    assert bus.published[0][0] == Topics().short_operation_command


def test_observation_summary_requires_runtime_semantic_scene_field() -> None:
    assert (
        _trusted_observation_summary("frame=4; images=2; scene=桌上有杯子")
        == "桌上有杯子"
    )
    assert _trusted_observation_summary("frame=4; images=2; camera=available") is None
