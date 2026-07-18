from __future__ import annotations

import asyncio

from hey_robot.cognition.robot_execution_gateway import (
    RobotExecutionGateway,
    _trusted_observation_summary,
)
from hey_robot.cognition.runtime.conversation_store import ConversationStore
from hey_robot.protocol import ActionProposal, Envelope, Topics
from hey_robot.skill_os.base import SkillCatalog, SkillSpec


class _Bus:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict]] = []

    async def publish(self, topic: str, payload: dict) -> None:
        self.published.append((topic, payload))


def test_short_operation_timeout_is_terminal_failure(tmp_path) -> None:
    bus = _Bus()
    store = ConversationStore(tmp_path / "conversation.sqlite3")
    adapter = RobotExecutionGateway(
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
            ActionProposal("skill", "move_base", "move forward", {"distance_cm": 20}),
            Envelope(robot_id="mock0"),
            "d1:main:owner",
        )
    )

    assert outcome.status == "failed"
    assert outcome.retryable is False
    assert "限定时间内返回最终结果" in (outcome.user_summary or "")
    assert bus.published[0][0] == Topics().short_operation_command


def test_observation_summary_requires_runtime_semantic_scene_field() -> None:
    assert _trusted_observation_summary(
        "frame=4; images=2; scene=a cup is on the table"
    ) == ("a cup is on the table")
    assert _trusted_observation_summary("frame=4; images=2; camera=available") is None
