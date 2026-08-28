from __future__ import annotations

import asyncio
import io

import numpy as np
from PIL import Image

from hey_robot.config import DeploymentConfig
from hey_robot.protocol import Envelope, RobotAction, RobotSkillAction, SkillIntent
from hey_robot.robot_backends.robocasa_remote.driver import RoboCasaRemoteDriver
from hey_robot.robot_backends.robocasa_remote.protocol import (
    RemoteImage,
    RemoteObservation,
    RemoteStep,
)
from hey_robot.robot_runtime.manager import RobotManager, create_driver_context


class _Client:
    def __init__(self) -> None:
        self.observe_calls = 0
        self.closed = False
        self.option_calls = []
        self.native_calls = []
        self.localize_calls = []
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

    async def run_option(self, **kwargs):
        self.option_calls.append(kwargs)
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
            done=False,
            status="budget",
            actions_executed=8,
            chunks_executed=1,
        )

    async def step_native(self, **kwargs):
        self.native_calls.append(kwargs)
        return RemoteStep(
            observation=RemoteObservation(
                episode_id="trial-1",
                frame_id=kwargs["expected_frame_id"] + 1,
                state=[0.0] * 16,
                images=[
                    RemoteImage(camera=f"camera{index}", data=_jpeg(index))
                    for index in range(1, 4)
                ],
                task="KettleBoiling",
            ),
            done=False,
            status="native",
            actions_executed=1,
            chunks_executed=1,
        )

    async def localize_pixels(self, **kwargs):
        self.localize_calls.append(kwargs)
        return {
            "frame_id": kwargs["expected_frame_id"],
            "camera": kwargs["camera"],
            "method": "simulator_metric_depth",
            "results": [
                {
                    "pixel": kwargs["pixels"][0],
                    "world_xyz": [1.0, 2.0, 0.12],
                    "depth_m": 0.7,
                    "valid": True,
                }
            ],
            "summary": {"valid_count": 1, "total_count": 1},
        }

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
            create_driver_context("robocasa0", spec, "test"),
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


def test_driver_reattaches_when_evaluator_advances_to_next_trial() -> None:
    async def run() -> None:
        driver, client = _driver()
        await driver.start()
        await driver.observe()

        async def observe_next():
            return RemoteObservation(
                episode_id="trial-2",
                frame_id=0,
                state=[0.0] * 16,
                images=[
                    RemoteImage(camera=f"camera{index}", data=_jpeg(index))
                    for index in range(1, 4)
                ],
                task="PrepareCoffee",
            )

        client.observe = observe_next
        observation = await driver.observe()

        assert observation.frame_id == 0
        assert driver.episode_id == "trial-2"
        assert driver.task == "PrepareCoffee"
        assert driver.state == "idle"

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


def test_driver_routes_policy_option_and_reset_through_runtime() -> None:
    async def run() -> None:
        driver, client = _driver(with_control=True)
        await driver.start()
        await driver.observe()
        capabilities = await driver.capabilities()
        assert capabilities.action_dimensions == 12
        assert (await driver.health()).online is True

        status = await driver.apply_action(
            RobotSkillAction(
                "run_policy_option",
                {
                    "session_id": "trial-1",
                    "instruction": "Boil the kettle",
                    "max_actions": 8,
                },
            ).to_robot_action(
                SkillIntent(
                    envelope=Envelope(robot_id="robocasa0"),
                    skill_id="skill-1",
                    task_id="trial-1",
                    intent_kind="skill",
                    name="run_policy_option",
                    arguments={},
                    objective="run option",
                )
            )
        )
        assert status.success is True
        assert status.frame_id == 9
        assert client.option_calls[0]["instruction"] == "Boil the kettle"

        reset = await driver.reset()
        assert reset.state == "idle"
        assert driver.frame_id == 0
        assert client.end_calls == [{"reason": "robot_reset"}]
        assert client.begin_calls[0]["task"] == "CloseFridge"

    asyncio.run(run())


def test_driver_routes_native_robot_action_before_skill_parsing() -> None:
    async def run() -> None:
        driver, client = _driver()
        await driver.start()
        await driver.observe()
        values = [0.0] * 12
        values[6] = -1.0
        status = await driver.apply_action(
            RobotAction(
                envelope=Envelope(robot_id="robocasa0"),
                values=values,
                skill_id="release-1",
                task_id="trial-1",
                metadata={"action_type": "embodiment_native"},
            )
        )

        assert status.success is True
        assert status.frame_id == 9
        assert client.native_calls == [{"action": values, "expected_frame_id": 8}]

    asyncio.run(run())


def test_driver_routes_read_only_depth_localization() -> None:
    async def run() -> None:
        driver, client = _driver()
        await driver.start()
        await driver.observe()
        capabilities = await driver.capabilities()
        assert "localize_pixels" in capabilities.metadata["supported_skills"]

        status = await driver.apply_action(
            RobotSkillAction(
                "localize_pixels", {"camera": "camera1", "pixels": [[3, 4]]}
            ).to_robot_action(
                SkillIntent(
                    envelope=Envelope(robot_id="robocasa0"),
                    skill_id="localize-1",
                    task_id="trial-1",
                    intent_kind="observation",
                    name="localize_pixels",
                    arguments={},
                    objective="localize selected pixels",
                )
            )
        )

        assert status.success is True
        assert status.metrics["last_skill_result"]["method"] == (
            "simulator_metric_depth"
        )
        assert client.localize_calls == [
            {
                "camera": "camera1",
                "pixels": [[3, 4]],
                "expected_frame_id": 8,
            }
        ]

    asyncio.run(run())


def test_driver_rejects_non_option_actions() -> None:
    async def run() -> None:
        driver, _client = _driver()
        await driver.start()
        await driver.observe()
        result = await driver.apply_action(
            RobotSkillAction("move_to", {}).to_robot_action(
                SkillIntent(
                    envelope=Envelope(robot_id="robocasa0"),
                    skill_id="bad",
                    task_id="bad",
                    intent_kind="skill",
                    name="move_to",
                    arguments={},
                    objective="move",
                )
            )
        )
        assert result.success is False
        assert "unsupported RoboCasa skill" in str(result.error)
        assert (await driver.status()).state == "error"

        reset = await driver.reset()
        assert reset.state == "error"
        assert "control plane is unavailable" in str(reset.error)

    asyncio.run(run())
