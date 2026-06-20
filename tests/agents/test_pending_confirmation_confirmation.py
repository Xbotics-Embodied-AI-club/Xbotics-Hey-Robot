from __future__ import annotations

from hey_robot.agents.robot_agent import RobotAgentService
from hey_robot.config import DeploymentConfig
from hey_robot.protocol import Envelope, UserTurn


def test_start_phrase_confirms_pending_capability(tmp_path) -> None:
    config = DeploymentConfig.from_dict(
        {
            "agents": {
                "main": {
                    "type": "robot_agent",
                    "robot_id": "mock0",
                    "settings": {"mode": "direct"},
                }
            },
            "robots": {"mock0": {"type": "mock"}},
        }
    )
    service = RobotAgentService(config, agent_id="main", episode_dir=tmp_path)

    decision = service._classify_pending_confirmation_turn(
        turn=UserTurn(envelope=Envelope(), text="\u5f00\u59cb\u5427")
    )

    assert decision.action == "confirm"
