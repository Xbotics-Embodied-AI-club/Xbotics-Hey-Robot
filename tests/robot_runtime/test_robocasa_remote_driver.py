from __future__ import annotations

import asyncio
import io

import numpy as np
from PIL import Image

from hey_robot.config import DeploymentConfig
from hey_robot.protocol import Envelope, RobotAction
from hey_robot.robot_runtime.base import RobotDriverContext
from hey_robot.robot_runtime.embodiments import get_embodiment_profile
from hey_robot.robot_runtime.manager import RobotManager
from hey_robot.robot_runtime.robocasa_remote.driver import RoboCasaRemoteDriver
from hey_robot.robot_runtime.robocasa_remote.protocol import (
    RemoteImage,
    RemoteObservation,
    RemoteStep,
)


class _Client:
    def __init__(self) -> None:
        self.observe_calls = 0
        self.closed = False
        self.step_calls = []
        self.begin_calls = []
        self.end_calls = []

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

    async def step(self, **kwargs):
        self.step_calls.append(kwargs)
        return RemoteStep(
            observation=RemoteObservation(
                episode_id="trial-1",
                frame_id=9,
                state=[0.0] * 16,
                images=[
                    RemoteImage(camera=f"camera{index}", data=_jpeg(index))
                    for index in range(1, 4)
                ],
                task="KettleBoiling",
            ),
            reward=0.5,
            done=False,
            metrics={"truncated": False},
        )

    async def begin_trial(self, **kwargs):
        self.begin_calls.append(kwargs)
        return RemoteObservation(
            episode_id=kwargs["trial_id"],
            frame_id=0,
            state=[0.0] * 16,
            images=[
                RemoteImage(camera=f"camera{index}", data=_jpeg(index))
                for index in range(1, 4)
            ],
            task=kwargs["task"],
        )

    async def end_trial(self, **kwargs):
        self.end_calls.append(kwargs)
        return True

    async def close(self):
        self.closed = True


def _driver(*, with_control: bool = False) -> tuple[RoboCasaRemoteDriver, _Client]:
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
            control_client=client if with_control else None,
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


def test_driver_routes_native_action_and_reset_through_runtime() -> None:
    async def run() -> None:
        driver, client = _driver(with_control=True)
        await driver.start()
        await driver.observe()
        capabilities = await driver.capabilities()
        assert capabilities.action_dimensions == 12
        assert (await driver.health()).online is True

        status = await driver.apply_action(
            RobotAction(
                envelope=Envelope(robot_id="robocasa0"),
                values=[0.0] * 12,
                skill_id="skill-1",
                metadata={
                    "expected_frame_id": 8,
                    "raw_action": [1.25] + [0.0] * 11,
                    "action_clipped": True,
                },
            )
        )
        assert status.success is True
        assert status.frame_id == 9
        assert client.step_calls[0]["expected_frame_id"] == 8
        assert client.step_calls[0]["action_clipped"] is True

        reset = await driver.reset()
        assert reset.state == "idle"
        assert driver.frame_id == 0
        assert client.end_calls == [{"reason": "robot_reset"}]
        assert client.begin_calls[0]["task"] == "CloseFridge"

    asyncio.run(run())


def test_driver_returns_structured_error_for_stale_or_invalid_action() -> None:
    async def run() -> None:
        driver, _client = _driver()
        await driver.start()
        await driver.observe()
        result = await driver.apply_action(
            RobotAction(
                envelope=Envelope(robot_id="robocasa0"),
                values=[0.0] * 12,
                skill_id="bad",
                metadata={"expected_frame_id": 7},
            )
        )
        assert result.success is False
        assert "stale action" in str(result.error)
        assert (await driver.status()).state == "error"

        reset = await driver.reset()
        assert reset.state == "error"
        assert "control plane is unavailable" in str(reset.error)

    asyncio.run(run())
