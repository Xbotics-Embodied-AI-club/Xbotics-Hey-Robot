from __future__ import annotations

import asyncio

from hey_robot.agents.robot_agent import RobotAgentService
from hey_robot.protocol import Envelope, UserTurn


class _Config:
    @staticmethod
    def default_robot_id(_agent_id: str) -> str:
        return "mock0"


class _Events:
    def __init__(self) -> None:
        self.items = []

    async def publish(self, event) -> None:
        self.items.append(event)


def test_turn_timeout_always_publishes_failure_reply() -> None:
    service = RobotAgentService.__new__(RobotAgentService)
    service.agent_id = "main"
    service.config = _Config()
    service.turn_timeout_sec = 0.01
    service._episode_locks = {}
    service._robot_locks = {}
    service.events = _Events()
    replies = []

    async def slow_turn(_turn) -> None:
        await asyncio.sleep(1)

    async def publish_reply(reply) -> None:
        replies.append(reply)

    async def drain(_episode_id) -> None:
        return None

    service._handle_user_turn_locked = slow_turn
    service.publish_reply = publish_reply
    service._drain_pending_turns = drain
    turn = UserTurn(
        envelope=Envelope(
            trace_id="trace-timeout",
            episode_id="episode-timeout",
            agent_id="main",
            robot_id="mock0",
            channel="web",
        ),
        text="你看到了什么",
    )

    asyncio.run(service._run_turn_pipeline(turn))

    assert len(replies) == 1
    assert replies[0].metadata["source"] == "turn_timeout"
    assert service.events.items[-1].kind == "agent.turn.end"
    assert service.events.items[-1].severity == "error"
