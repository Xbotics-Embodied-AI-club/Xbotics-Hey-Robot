from __future__ import annotations

import asyncio
from io import BytesIO

import numpy as np
from PIL import Image

from hey_robot.config import DeploymentConfig
from hey_robot.protocol import Envelope, RobotSkillAction, SkillIntent
from hey_robot.robot_runtime import RobotManager
from hey_robot.robot_runtime.base import RobotDriverContext
from hey_robot.robot_runtime.embodiments import get_embodiment_profile
from hey_robot.robot_runtime.habitat_remote.driver import HabitatRemoteDriver
from hey_robot.robot_runtime.habitat_remote.protocol import (
    HabitatAsset,
    HabitatObservation,
    HabitatSkillResult,
)


class _FakeHabitatClient:
    def __init__(self) -> None:
        self.frame_id = 1
        self.calls: list[dict] = []

    async def health(self):
        return {"online": True, "loaded": True}

    async def create_episode(self, **kwargs):
        assert kwargs["profile"] == "habitat3_social_spot_human_oracle"
        return self._observation()

    async def observe(self, *, episode_id: str):
        assert episode_id == "habitat-1"
        return self._observation()

    async def step(self, **kwargs):
        self.calls.append(kwargs)
        self.frame_id += 1
        return type("Step", (), {"observation": self._observation(), "metrics": {}})()

    async def execute_skill(self, **kwargs):
        self.calls.append(kwargs)
        self.frame_id += 2
        return HabitatSkillResult(
            observation=self._observation(),
            steps=2,
            success=True,
            metrics={"privileged": True},
            trace=[{"step": 1}],
        )

    async def cancel_skill(self, **_kwargs):
        return True

    async def reset(self, **_kwargs):
        self.frame_id += 1
        return self._observation()

    async def close_episode(self, **_kwargs):
        return True

    async def close(self):
        return None

    def _observation(self):
        return HabitatObservation(
            episode_id="habitat-1",
            frame_id=self.frame_id,
            assets=[
                HabitatAsset(
                    kind="image",
                    role="camera",
                    name="agent_0_arm_rgb",
                    data=_jpeg(),
                    content_type="image/jpeg",
                ),
                HabitatAsset(
                    kind="depth",
                    role="depth",
                    name="agent_0_arm_depth",
                    data=b"depth-png",
                    content_type="image/png",
                    metadata={"artifact_type": "depth"},
                ),
            ],
            proprioception=[1.0, 2.0],
            entities=[{"entity_id": "agent_1", "entity_type": "human"}],
        )


def _jpeg() -> bytes:
    out = BytesIO()
    Image.fromarray(np.full((4, 4, 3), 120, dtype=np.uint8)).save(out, format="JPEG")
    return out.getvalue()


def _driver() -> tuple[HabitatRemoteDriver, _FakeHabitatClient]:
    spec = DeploymentConfig.from_dict(
        {
            "deployment": {"id": "test"},
            "robots": {
                "habitat": {
                    "type": "habitat3",
                    "family": "habitat3",
                    "environment": "remote",
                    "driver": "grpc",
                }
            },
        }
    ).robots["habitat"]
    client = _FakeHabitatClient()
    return HabitatRemoteDriver(
        RobotDriverContext("habitat", spec, "test", get_embodiment_profile(spec)),
        client,
    ), client


def test_habitat_driver_executes_semantic_skill_and_preserves_depth() -> None:
    async def run() -> None:
        driver, client = _driver()
        await driver.start()
        intent = SkillIntent(
            envelope=Envelope(robot_id="habitat"),
            skill_id="skill-1",
            task_id="task-1",
            intent_kind="skill",
            name="habitat_follow_human",
            arguments={"distance_m": 2.0},
            objective="follow the person",
        )
        status = await driver.apply_action(
            RobotSkillAction("habitat_follow_human", {"max_steps": 10}).to_robot_action(
                intent
            )
        )
        observation = await driver.observe()
        assert status.success is True
        assert client.calls[0]["skill_name"] == "habitat_follow_human"
        assert client.calls[0]["expected_frame_id"] == 1
        assert observation.assets[0].data.shape == (4, 4, 3)
        assert observation.assets[1].data == b"depth-png"
        assert observation.metadata["entities"][0]["entity_id"] == "agent_1"

    asyncio.run(run())


def test_robot_manager_builds_habitat_remote_driver() -> None:
    config = DeploymentConfig.from_dict(
        {
            "robots": {
                "habitat": {
                    "type": "habitat3",
                    "family": "habitat3",
                    "environment": "remote",
                    "driver": "grpc",
                }
            }
        }
    )
    assert isinstance(RobotManager(config).require("habitat"), HabitatRemoteDriver)
