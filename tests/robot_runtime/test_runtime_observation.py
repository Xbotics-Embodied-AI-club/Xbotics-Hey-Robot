from __future__ import annotations

from hey_robot.config import DeploymentConfig
from hey_robot.protocol import (
    Envelope,
    RobotObservation,
    RobotSkillAction,
    RobotStatus,
    SceneEntity,
    SceneRelation,
    SkillIntent,
)
from hey_robot.robot_api import (
    DriverObservation,
    ObservationAsset,
    RobotCapabilities,
    RobotHealth,
)
from hey_robot.robot_media import LocalMediaStore
from hey_robot.robot_runtime.manager import RobotManager
from hey_robot.robot_runtime.runtime import RobotRuntime


def _intent(skill_id: str, name: str, objective: str) -> SkillIntent:
    return SkillIntent(
        envelope=Envelope(robot_id="mock0"),
        skill_id=skill_id,
        task_id="task-test",
        intent_kind="observation"
        if name in {"inspect_scene", "look_around", "detect_marker"}
        else "skill",
        name=name,
        arguments={},
        objective=objective,
    )


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
    skill = _intent("cmd1", "inspect_scene", "look ahead")
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


async def test_robot_runtime_routes_motion_through_control_plane(tmp_path) -> None:
    config = DeploymentConfig.from_dict({"robots": {"mock0": {"type": "mock"}}})
    runtime = RobotRuntime(
        RobotManager(config).require("mock0"), LocalMediaStore(tmp_path)
    )
    await runtime.start()
    intent = _intent("move1", "move_base", "move forward")
    action = RobotSkillAction(
        "move_base", {"direction": "forward", "distance_cm": 10}
    ).to_robot_action(intent)

    status = await runtime.apply_action(action)

    assert status.success is True
    assert status.metrics["control_plane"]["buffer_size"] == 1
    assert status.metrics["control_plane"]["last_watchdog"]["skill_id"] == "move1"
    assert runtime.control_plane.action_buffer[-1].action_type == "skill"


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
    skill = _intent("scan1", "inspect_scene", "look")
    action = RobotSkillAction("inspect_scene", safety_level="observe").to_robot_action(
        skill
    )

    status = await runtime.apply_action(action)

    assert status.success is True
    assert driver.observe_count >= 1
    assert status.metrics["last_skill_result"]["skill"] == "inspect_scene"


async def test_robot_runtime_uses_scene_captioner_for_inspect_scene(tmp_path) -> None:
    captioner = _FakeSceneCaptioner()
    runtime = RobotRuntime(
        _CountingCameraDriver("mock0"),
        LocalMediaStore(tmp_path / "media"),
        scene_captioner=captioner,
    )
    await runtime.start()
    action = RobotSkillAction("inspect_scene", safety_level="observe").to_robot_action(
        _intent("caption1", "inspect_scene", "describe scene")
    )

    status = await runtime.apply_action(action)

    result = status.metrics["last_skill_result"]
    assert result["semantic_available"] is True
    assert result["summary"] == "scene=mug on table"
    assert captioner.observations[0].images


async def test_inspect_scene_preserves_structured_object_locations(tmp_path) -> None:
    class Captioner:
        async def caption(self, observation, _status, *, question=None):
            del observation
            from hey_robot.cognition.perception.scene import (
                SceneObject,
                SceneUnderstanding,
            )

            assert question == "find the kettle"
            return SceneUnderstanding(
                summary="kitchen counter",
                objects=[SceneObject("kettle", "left counter", 0.93)],
                task_relevance="the requested kettle is visible",
                confidence=0.9,
            )

    runtime = RobotRuntime(
        _CountingCameraDriver("mock0"),
        LocalMediaStore(tmp_path / "media"),
        scene_captioner=Captioner(),
    )
    await runtime.start()
    action = RobotSkillAction(
        "inspect_scene",
        {"question": "find the kettle"},
        safety_level="observe",
    ).to_robot_action(_intent("structured1", "inspect_scene", "find the kettle"))

    status = await runtime.apply_action(action)

    summary = status.metrics["last_skill_result"]["summary"]
    assert "objects=[kettle@left counter(0.93)]" in summary
    assert "task_relevance=the requested kettle is visible" in summary


async def test_inspect_scene_publishes_frame_scoped_entities(tmp_path) -> None:
    class Captioner:
        async def caption(self, observation, _status, *, question=None):
            del question
            from hey_robot.cognition.perception.scene import SceneUnderstanding

            return SceneUnderstanding(
                summary="passage visible",
                entities=[
                    SceneEntity(
                        "passage:1",
                        "passage",
                        observation.frame_id,
                        {"bearing": "front_right"},
                        [SceneRelation("leads_to", "room:kitchen")],
                    )
                ],
                confidence=0.9,
            )

    runtime = RobotRuntime(
        _CountingCameraDriver("mock0"),
        LocalMediaStore(tmp_path / "media"),
        scene_captioner=Captioner(),
    )
    await runtime.start()
    action = RobotSkillAction("inspect_scene", safety_level="observe").to_robot_action(
        _intent("entities1", "inspect_scene", "inspect doorway")
    )

    await runtime.apply_action(action)
    observation = await runtime.observe()

    assert observation.entities[0].entity_id == "passage:1"
    assert observation.entities[0].relations[0].object_id == "room:kitchen"


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
        _intent("marker1", "detect_marker", "detect marker")
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
        _intent("look1", "look_around", "look around")
    )

    status = await runtime.apply_action(action)

    assert status.success is True
    assert len(status.metrics["last_skill_result"]["observations"]) == 4
    assert [name for name, _ in driver.applied_skills] == [
        "turn_base",
        "turn_base",
        "turn_base",
    ]


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


class _FakeSceneCaptioner:
    def __init__(self) -> None:
        self.observations: list[RobotObservation] = []

    async def caption(self, observation, _status, *, question=None):
        del question
        from hey_robot.cognition.perception.scene import SceneUnderstanding

        self.observations.append(observation)
        return SceneUnderstanding(summary="mug on table", confidence=0.9)
