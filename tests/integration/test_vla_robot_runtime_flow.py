from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from hey_robot.config import DeploymentConfig
from hey_robot.foundation.clients.models import ModelInferenceResult
from hey_robot.protocol import Envelope
from hey_robot.robot_backends.robocasa_remote.driver import RoboCasaRemoteDriver
from hey_robot.robot_backends.robocasa_remote.protocol import (
    RemoteImage,
    RemoteObservation,
    RemoteStep,
)
from hey_robot.robot_media import LocalMediaStore
from hey_robot.robot_runtime.clients import LocalRobotClient
from hey_robot.robot_runtime.manager import create_driver_context
from hey_robot.robot_runtime.runtime import RobotRuntime
from hey_robot.skills import (
    SkillCommand,
    SkillContext,
    SkillEvent,
    SkillRunner,
    load_skill_registry,
)
from hey_robot.skills.resources import ResourceManager


class _EpisodeClient:
    def __init__(self) -> None:
        self.frame_id = 0
        self.steps: list[dict[str, object]] = []

    async def health(self):
        return {"online": True, "loaded": True}

    async def observe(self):
        return self._observation()

    async def step(self, **kwargs):
        self.steps.append(kwargs)
        self.frame_id += 1
        return RemoteStep(
            observation=self._observation(),
            reward=0.0,
            done=False,
            metrics={},
        )

    async def close(self):
        return None

    def _observation(self) -> RemoteObservation:
        return RemoteObservation(
            episode_id="trial-1",
            frame_id=self.frame_id,
            state=[0.0] * 16,
            images=[
                RemoteImage(camera=f"camera{index}", data=_jpeg(index))
                for index in range(1, 4)
            ],
            task="CloseFridge",
            metadata={"policy_task": "Close the fridge door."},
        )


class _Models:
    async def infer(self, *_args: object, **_kwargs: object):
        return ModelInferenceResult(
            True,
            "policy action",
            data={
                "action_chunk": {
                    "kind": "action_chunk",
                    "action_space": "robocasa_12d",
                    "embodiment": "robocasa",
                    "horizon": 1,
                    "done": True,
                    "actions": [
                        {
                            "name": "embodiment_native_action",
                            "arguments": {
                                "values": [0.0] * 12,
                                "raw_values": [0.0] * 12,
                                "action_space": "robocasa_12d",
                                "embodiment": "robocasa",
                            },
                        }
                    ],
                }
            },
        )

    async def cancel(self, _run_id: str) -> None:
        return None


class _Events:
    async def emit(self, _event: SkillEvent) -> None:
        return None


@pytest.mark.asyncio
async def test_vla_action_reaches_real_robocasa_runtime_gate(tmp_path) -> None:
    config = DeploymentConfig.from_dict(
        {
            "robots": {
                "robocasa365": {
                    "type": "robocasa",
                    "family": "robocasa",
                    "environment": "remote",
                    "driver": "grpc",
                    "embodiment_profile": "robocasa_remote",
                    "settings": {"target": "grpc://unused"},
                }
            }
        }
    )
    spec = config.robots["robocasa365"]
    episode_client = _EpisodeClient()
    driver = RoboCasaRemoteDriver(
        create_driver_context("robocasa365", spec, "test"),
        episode_client,
    )
    runtime = RobotRuntime(driver, LocalMediaStore(tmp_path / "media"))
    await runtime.start()
    robot_client = LocalRobotClient({"robocasa365": runtime})
    registry = load_skill_registry(("hey_robot.skills.builtins",))
    runner = SkillRunner(
        registry,
        resources=ResourceManager(),
        events=_Events(),
        context_factory=lambda command: SkillContext(
            run_id=command.run_id,
            task_id=command.task_id,
            robot_id=command.robot_id,
            robot=robot_client,
            models=_Models(),
        ),
    )
    try:
        result = await runner.execute(
            SkillCommand(
                envelope=Envelope(robot_id="robocasa365", episode_id="trial-1"),
                run_id="run-1",
                task_id="task-1",
                robot_id="robocasa365",
                name="manipulate",
                arguments={"task_prompt": "Close the fridge.", "max_steps": 1},
            )
        )
    finally:
        await runtime.close()

    assert result.success is True
    assert result.data["termination_reason"] == "model_done"
    assert len(episode_client.steps) == 1
    assert episode_client.steps[0]["action"] == [0.0] * 12
    assert episode_client.steps[0]["expected_frame_id"] == 0


def _jpeg(value: int) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.full((2, 2, 3), value, dtype=np.uint8)).save(
        buffer, format="JPEG"
    )
    return buffer.getvalue()
