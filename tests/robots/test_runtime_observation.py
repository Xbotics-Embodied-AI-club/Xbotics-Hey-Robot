from __future__ import annotations

from hey_robot.config import DeploymentConfig
from hey_robot.media import LocalMediaStore
from hey_robot.perception import DriverObservation, ObservationAsset
from hey_robot.protocol import (
    ArtifactRef,
    Envelope,
    ImageRef,
    RobotObservation,
    RobotStatus,
    SkillIntent,
)
from hey_robot.robots import RobotManager, RobotRuntime
from hey_robot.robots.base import RobotCapabilities, RobotHealth
from hey_robot.robots.service import RobotService
from hey_robot.skills import RobotSkillAction


async def test_robot_runtime_observation_uses_pipeline(tmp_path) -> None:
    config = DeploymentConfig.from_dict({"robots": {"mock0": {"type": "mock"}}})
    runtime = RobotRuntime(
        RobotManager(config).require("mock0"), LocalMediaStore(tmp_path)
    )
    await runtime.start()

    observation = await runtime.observe()

    assert observation.images
    assert observation.artifacts == []
    assert observation.raw["perception"]["source"] == "refresh"
    assert "_images" not in observation.raw
    assert "policy_observation" not in observation.raw


async def test_robot_runtime_keeps_latest_perception_snapshot(tmp_path) -> None:
    config = DeploymentConfig.from_dict({"robots": {"mock0": {"type": "mock"}}})
    runtime = RobotRuntime(
        RobotManager(config).require("mock0"), LocalMediaStore(tmp_path)
    )
    await runtime.start()

    assert await runtime.latest_observation() is None

    observation = await runtime.observe()
    latest = await runtime.latest_observation(max_age_ms=1000)

    assert latest is not None
    assert latest.frame_id == observation.frame_id
    assert runtime.perception.latest(max_age_ms=1000) is not None


async def test_robot_runtime_handles_perception_skill_without_driver_action(
    tmp_path,
) -> None:
    config = DeploymentConfig.from_dict({"robots": {"mock0": {"type": "mock"}}})
    runtime = RobotRuntime(
        RobotManager(config).require("mock0"), LocalMediaStore(tmp_path)
    )
    await runtime.start()
    skill = SkillIntent(
        envelope=Envelope(robot_id="mock0"), skill_id="cmd1", objective="look ahead"
    )
    action = RobotSkillAction("inspect_scene", safety_level="observe").to_robot_action(
        skill
    )

    status = await runtime.apply_action(action)
    latest = await runtime.latest_observation(max_age_ms=1000)

    assert status.success is True
    assert status.skill_id == "cmd1"
    assert latest is not None
    assert latest.images
    assert status.metrics["last_skill_result"]["skill"] == "inspect_scene"
    assert status.metrics["last_skill_result"]["source"] == "refresh"


async def test_robot_runtime_observe_always_refreshes_from_driver(
    tmp_path,
) -> None:
    driver = _CountingCameraDriver("mock0")
    runtime = RobotRuntime(driver, LocalMediaStore(tmp_path / "media"))
    await runtime.start()

    observation = await runtime.observe()

    assert driver.observe_count == 1
    assert observation.images
    assert observation.raw["perception"]["source"] == "refresh"


async def test_robot_runtime_perception_skill_refreshes_camera_directly(
    tmp_path,
) -> None:
    driver = _CountingCameraDriver("mock0")
    runtime = RobotRuntime(driver, LocalMediaStore(tmp_path / "media"))
    await runtime.start()
    skill = SkillIntent(
        envelope=Envelope(robot_id="mock0"), skill_id="scan1", objective="look"
    )
    action = RobotSkillAction("inspect_scene", safety_level="observe").to_robot_action(
        skill
    )

    status = await runtime.apply_action(action)

    assert status.success is True
    assert driver.observe_count >= 1
    assert status.metrics["last_skill_result"]["skill"] == "inspect_scene"


async def test_robot_runtime_status_returns_driver_status_directly(tmp_path) -> None:
    driver = _CountingCameraDriver("mock0")
    runtime = RobotRuntime(driver, LocalMediaStore(tmp_path / "media"))
    await runtime.start()

    status = await runtime.status()

    assert status.state == "idle"
    assert "camera" not in status.metrics  # no camera_service annotation


async def test_robot_runtime_no_camera_service_params_accepted(tmp_path) -> None:
    import pytest

    config = DeploymentConfig.from_dict({"robots": {"mock0": {"type": "mock"}}})
    with pytest.raises(TypeError):
        RobotRuntime(
            RobotManager(config).require("mock0"),
            LocalMediaStore(tmp_path),
            prefer_camera_service=True,  # type: ignore[call-arg]
        )


async def test_robot_runtime_current_perception_snapshot_always_refreshes_driver(
    tmp_path,
) -> None:
    driver = _CountingCameraDriver("mock0")
    runtime = RobotRuntime(driver, LocalMediaStore(tmp_path / "media"))
    await runtime.start()

    snapshot = await runtime._current_perception_snapshot(reason="inspect_scene")

    assert snapshot is not None
    assert driver.observe_count == 1
    assert snapshot.source == "refresh"


async def test_robot_runtime_detect_marker_with_square_fallback(tmp_path) -> None:
    runtime = RobotRuntime(
        _SquareMarkerDriver("mock0", square_center_x=48), LocalMediaStore(tmp_path)
    )
    await runtime.start()
    action = RobotSkillAction("detect_marker").to_robot_action(
        SkillIntent(envelope=Envelope(robot_id="mock0"), skill_id="marker1")
    )

    status = await runtime.apply_action(action)

    assert status.success is True
    assert status.metrics["last_skill_result"]["markers"]


async def test_robot_runtime_look_around_collects_multiple_observations(
    tmp_path,
) -> None:
    driver = _CountingCameraDriver("mock0")
    runtime = RobotRuntime(driver, LocalMediaStore(tmp_path))
    await runtime.start()
    action = RobotSkillAction("look_around").to_robot_action(
        SkillIntent(envelope=Envelope(robot_id="mock0"), skill_id="look1")
    )

    status = await runtime.apply_action(action)

    assert status.success is True
    assert len(status.metrics["last_skill_result"]["observations"]) == 4
    assert [name for name, _ in driver.applied_skills] == [
        "turn_base",
        "turn_base",
        "turn_base",
    ]


def test_robot_service_does_not_publish_invalid_camera_images_to_scene_memory() -> None:
    black_frame = RobotObservation(
        envelope=Envelope(robot_id="mock0"),
        frame_id=1,
        images=[ImageRef(uri="media://local/images/mock0/black.jpg", camera="front")],
        raw={
            "perception": {
                "image_count": 1,
                "valid_image_count": 0,
                "image_quality_issues": ["black_frame"],
            }
        },
    )
    valid_frame = RobotObservation(
        envelope=Envelope(robot_id="mock0"),
        frame_id=2,
        images=[ImageRef(uri="media://local/images/mock0/frame.jpg", camera="front")],
        raw={"perception": {"image_count": 1, "valid_image_count": 1}},
    )
    artifact_only = RobotObservation(
        envelope=Envelope(robot_id="mock0"),
        frame_id=3,
        artifacts=[
            ArtifactRef(
                uri="media://local/artifacts/mock0/state.json",
                artifact_type="policy_observation",
            )
        ],
        raw={"perception": {"image_count": 0, "valid_image_count": 0}},
    )

    assert RobotService._should_publish_observation(black_frame) is False
    assert RobotService._should_publish_observation(valid_frame) is True
    assert RobotService._should_publish_observation(artifact_only) is True


class _CountingCameraDriver:
    def __init__(self, robot_id: str) -> None:
        self.robot_id = robot_id
        self.observe_count = 0
        self.applied_skills: list[tuple[str, dict]] = []

    async def start(self) -> None:
        return None

    async def capabilities(self) -> RobotCapabilities:
        return RobotCapabilities(
            robot_id=self.robot_id, driver_type="fake", cameras=["front"]
        )

    async def health(self) -> RobotHealth:
        return RobotHealth(robot_id=self.robot_id, online=True, state="idle")

    async def observe(self) -> DriverObservation:
        import numpy as np

        self.observe_count += 1
        return DriverObservation(
            envelope=Envelope(robot_id=self.robot_id),
            frame_id=self.observe_count,
            assets=[
                ObservationAsset(
                    kind="image",
                    role="camera",
                    name="front",
                    data=np.full((100, 100, 3), 255, dtype=np.uint8),
                )
            ],
        )

    async def status(self) -> RobotStatus:
        return RobotStatus(
            envelope=Envelope(robot_id=self.robot_id),
            frame_id=self.observe_count,
            state="idle",
        )

    async def apply_action(self, _action) -> RobotStatus:
        skill = RobotSkillAction.from_robot_action(_action)
        self.applied_skills.append((skill.name, dict(skill.arguments)))
        return await self.status()

    async def reset(self) -> RobotStatus:
        return await self.status()

    async def close(self) -> None:
        return None


class _SquareMarkerDriver(_CountingCameraDriver):
    def __init__(self, robot_id: str, *, square_center_x: int) -> None:
        super().__init__(robot_id)
        self.square_center_x = square_center_x

    async def observe(self) -> DriverObservation:
        import numpy as np

        self.observe_count += 1
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        x0 = max(0, min(80, self.square_center_x - 10))
        image[40:60, x0 : x0 + 20, :] = 255
        return DriverObservation(
            envelope=Envelope(robot_id=self.robot_id),
            frame_id=self.observe_count,
            assets=[
                ObservationAsset(
                    kind="image",
                    role="camera",
                    name="front",
                    data=image,
                )
            ],
        )
