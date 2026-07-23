from __future__ import annotations

from dataclasses import dataclass, field

from hey_robot.protocol import Envelope, RobotObservation, RobotStatus
from hey_robot.robot_runtime import LocalRobotClient
from hey_robot.robot_runtime.base import RobotCapabilities


@dataclass
class Runtime:
    actions: list = field(default_factory=list)

    async def capabilities(self):
        return RobotCapabilities(
            robot_id="mock0",
            driver_type="mock",
            cameras=["front"],
            metadata={"supported_skills": ["move_base"]},
        )

    async def observe(self):
        return RobotObservation(Envelope(robot_id="mock0"), frame_id=5)

    async def apply_action(self, action):
        self.actions.append(action)
        return RobotStatus(
            Envelope(robot_id="mock0"),
            frame_id=6,
            success=True,
            metrics={
                "last_skill_result": {
                    "success": True,
                    "message": "moved",
                    "distance_cm": 20,
                }
            },
        )


async def test_local_robot_client_adapts_runtime_actions() -> None:
    runtime = Runtime()
    client = LocalRobotClient({"mock0": runtime})

    capabilities = await client.capabilities("mock0")
    observation = await client.observe("mock0")
    result = await client.execute(
        "mock0",
        "move_base",
        {"direction": "forward"},
        run_id="run-1",
        expected_frame_id=4,
    )

    assert capabilities.actions[0].name == "move_base"
    assert observation.frame_id == 5
    assert result.success is True
    assert result.summary == "moved"
    assert result.data["distance_cm"] == 20
    assert runtime.actions[0].skill_id == "run-1"
    assert runtime.actions[0].metadata["expected_frame_id"] == 4
