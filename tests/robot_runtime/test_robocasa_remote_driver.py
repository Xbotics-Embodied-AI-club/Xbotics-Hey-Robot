from __future__ import annotations

import asyncio
import io

import numpy as np
from PIL import Image

from hey_robot.config import DeploymentConfig
from hey_robot.robot_runtime.base import RobotDriverContext
from hey_robot.robot_runtime.embodiments import get_embodiment_profile
from hey_robot.robot_runtime.manager import RobotManager
from hey_robot.robot_runtime.robocasa_remote.driver import RoboCasaRemoteDriver
from hey_robot.robot_runtime.robocasa_remote.protocol import (
    RemoteImage,
    RemoteObservation,
)


class _Client:
    def __init__(self) -> None:
        self.observe_calls = 0
        self.closed = False

    async def health(self):
        return {"online": True, "loaded": True}

    async def observe(self):
        self.observe_calls += 1
        return RemoteObservation(
            episode_id="trial-1",
            frame_id=8,
            state=[0.0] * 16,
            images=[
                RemoteImage(camera=f"camera{index}", data=_jpeg(index))
                for index in range(1, 4)
            ],
            task="KettleBoiling",
        )

    async def close(self):
        self.closed = True


def _driver() -> tuple[RoboCasaRemoteDriver, _Client]:
    config = DeploymentConfig.from_dict(
        {
            "robots": {
                "robocasa0": {
                    "type": "robocasa",
                    "family": "robocasa",
                    "environment": "remote",
                    "driver": "grpc",
                    "embodiment_profile": "robocasa_remote",
                    "settings": {"target": "grpc://worker:9092"},
                }
            }
        }
    )
    spec = config.robots["robocasa0"]
    client = _Client()
    return (
        RoboCasaRemoteDriver(
            RobotDriverContext("robocasa0", spec, "test", get_embodiment_profile(spec)),
            client,
        ),
        client,
    )


def _jpeg(value: int) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.full((2, 2, 3), value, dtype=np.uint8)).save(
        buffer, format="JPEG"
    )
    return buffer.getvalue()


def test_driver_attaches_to_evaluator_owned_active_trial() -> None:
    async def run() -> None:
        driver, client = _driver()
        await driver.start()
        assert driver.episode_id is None
        observation = await driver.observe()
        assert client.observe_calls == 1
        assert driver.episode_id == "trial-1"
        assert driver.task == "KettleBoiling"
        assert observation.frame_id == 8
        assert observation.assets[0].data.shape == (2, 2, 3)

    asyncio.run(run())


def test_driver_close_does_not_end_evaluator_trial() -> None:
    async def run() -> None:
        driver, client = _driver()
        await driver.start()
        await driver.close()
        assert client.closed is True
        assert driver.state == "closed"

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
                    "settings": {"target": "grpc://localhost:9092"},
                }
            }
        }
    )
    assert isinstance(RobotManager(config).require("robocasa0"), RoboCasaRemoteDriver)
