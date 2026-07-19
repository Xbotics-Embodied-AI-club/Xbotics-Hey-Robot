from __future__ import annotations

import asyncio

from hey_robot.config import DeploymentConfig
from hey_robot.protocol import Envelope, RobotAction
from hey_robot.robot_runtime import RobotManager
from hey_robot.robot_runtime.base import RobotDriverContext
from hey_robot.robot_runtime.embodiments import get_embodiment_profile
from hey_robot.robot_runtime.robocasa_remote.driver import RoboCasaRemoteDriver
from hey_robot.robot_runtime.robocasa_remote.protocol import (
    RemoteImage,
    RemoteObservation,
    RemoteStep,
)


class _FakeRemoteClient:
    def __init__(self) -> None:
        self.actions: list[tuple[str, list[float], int]] = []
        self.closed = False
        self.frame_id = 7

    async def health(self):
        return {"online": True, "loaded": True}

    async def create_episode(self, *, task: str, seed: int):
        assert task == "CloseFridge"
        assert seed == 1000
        return self._observation()

    async def observe(self, *, episode_id: str):
        assert episode_id == "episode-1"
        self.frame_id += 1
        return self._observation()

    async def step(
        self, *, episode_id: str, action: list[float], expected_frame_id: int
    ):
        self.actions.append((episode_id, action, expected_frame_id))
        self.frame_id += 1
        return RemoteStep(self._observation(), reward=0.25, done=False, success=False)

    async def reset(self, *, episode_id: str):
        assert episode_id == "episode-1"
        self.frame_id = 1
        return self._observation()

    async def close_episode(self, *, episode_id: str):
        assert episode_id == "episode-1"
        self.closed = True
        return True

    async def close(self):
        self.closed = True

    def _observation(self):
        return RemoteObservation(
            "episode-1",
            self.frame_id,
            [0.0] * 16,
            [
                RemoteImage("camera1", b"jpeg-1"),
                RemoteImage("camera2", b"jpeg-2"),
                RemoteImage("camera3", b"jpeg-3"),
            ],
            "CloseFridge",
        )


def _driver() -> tuple[RoboCasaRemoteDriver, _FakeRemoteClient]:
    spec = DeploymentConfig.from_dict(
        {
            "deployment": {"id": "test"},
            "robots": {
                "robocasa0": {
                    "type": "robocasa",
                    "family": "robocasa",
                    "environment": "remote",
                    "driver": "grpc",
                    "settings": {"task": "CloseFridge", "seed": 1000},
                }
            },
        }
    ).robots["robocasa0"]
    client = _FakeRemoteClient()
    return RoboCasaRemoteDriver(
        RobotDriverContext("robocasa0", spec, "test", get_embodiment_profile(spec)),
        client,
    ), client


def test_remote_driver_starts_a_single_episode_and_materializes_images() -> None:
    async def run() -> None:
        driver, _ = _driver()
        await driver.start()
        capabilities = await driver.capabilities()
        observation = await driver.observe()
        assert capabilities.action_dimensions == 12
        assert capabilities.metadata["simulator_only"] is True
        assert observation.frame_id == 8
        assert observation.proprioception == [0.0] * 16
        assert observation.assets[0].name == "camera1"
        assert observation.assets[0].data == b"jpeg-1"

    asyncio.run(run())


def test_remote_driver_steps_with_current_frame_and_rejects_bad_actions() -> None:
    async def run() -> None:
        driver, client = _driver()
        await driver.start()
        before = await driver.observe()
        status = await driver.apply_action(
            RobotAction(
                Envelope(robot_id="robocasa0"),
                [0.0] * 12,
                skill_id="vla-step-1",
                metadata={"expected_frame_id": before.frame_id},
            )
        )
        assert status.success is True
        assert client.actions == [("episode-1", [0.0] * 12, before.frame_id)]
        bad = await driver.apply_action(
            RobotAction(Envelope(robot_id="robocasa0"), [0.0] * 11)
        )
        assert bad.success is False
        assert "expected 12, got 11" in (bad.error or "")

    asyncio.run(run())


def test_remote_driver_rejects_stale_actions_and_closes_episode() -> None:
    async def run() -> None:
        driver, client = _driver()
        await driver.start()
        stale = await driver.apply_action(
            RobotAction(
                Envelope(robot_id="robocasa0"),
                [0.0] * 12,
                metadata={"expected_frame_id": 0},
            )
        )
        assert stale.success is False
        assert "stale action" in (stale.error or "")
        await driver.close()
        assert client.closed is True

    asyncio.run(run())


def test_robot_manager_builds_remote_robocasa_driver() -> None:
    config = DeploymentConfig.from_dict(
        {
            "robots": {
                "robocasa0": {
                    "type": "robocasa",
                    "family": "robocasa",
                    "environment": "remote",
                    "driver": "grpc",
                    "settings": {"target": "grpc://127.0.0.1:9093"},
                }
            }
        }
    )
    assert isinstance(RobotManager(config).require("robocasa0"), RoboCasaRemoteDriver)
