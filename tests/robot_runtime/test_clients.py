from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from hey_robot.protocol import (
    Envelope,
    RobotObservation,
    RobotSkillAction,
    RobotStatus,
    SkillIntent,
)
from hey_robot.robot_api import RobotCapabilities
from hey_robot.robot_runtime.clients import LocalRobotClient


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

    async def emergency_stop(self, *, reason: str):
        intent = SkillIntent(
            Envelope(robot_id="mock0"),
            "emergency_stop",
            "emergency_stop",
            "skill",
            "stop_motion",
            {"emergency": True, "reason": reason},
            reason,
        )
        self.actions.append(
            RobotSkillAction(
                "stop_motion", {"emergency": True, "reason": reason}
            ).to_robot_action(intent)
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


async def test_local_robot_client_waits_for_fresh_frame_and_times_out() -> None:
    runtime = Runtime()
    frames = iter((5, 5, 6))

    async def observe():
        return RobotObservation(Envelope(robot_id="mock0"), frame_id=next(frames))

    runtime.observe = observe
    client = LocalRobotClient({"mock0": runtime})

    fresh = await client.observe("mock0", after_frame_id=5, timeout_sec=0.2)

    assert fresh.frame_id == 6
    runtime.observe = lambda: asyncio.sleep(
        0, result=RobotObservation(Envelope(robot_id="mock0"), frame_id=6)
    )
    with pytest.raises(TimeoutError, match="fresh observation timed out"):
        await client.observe("mock0", after_frame_id=6, timeout_sec=0.02)


async def test_local_robot_client_emergency_stop_uses_direct_runtime_action() -> None:
    runtime = Runtime()
    client = LocalRobotClient({"mock0": runtime})

    await client.emergency_stop("mock0", reason="operator")

    stop = RobotSkillAction.from_robot_action(runtime.actions[0])
    assert stop.name == "stop_motion"
    assert stop.arguments == {"emergency": True, "reason": "operator"}
