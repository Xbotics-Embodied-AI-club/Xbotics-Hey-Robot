from __future__ import annotations

from hey_robot.config import DeploymentConfig
from hey_robot.media import LocalMediaStore
from hey_robot.protocol import Envelope, RobotAction, SkillIntent
from hey_robot.robots import RobotManager, RobotRuntime
from hey_robot.skills import RobotSkillAction


def _runtime(tmp_path, settings: dict | None = None) -> RobotRuntime:
    config = DeploymentConfig.from_dict(
        {"robots": {"mock0": {"type": "mock", **(settings or {})}}}
    )
    return RobotRuntime(
        RobotManager(config).require("mock0"), LocalMediaStore(tmp_path)
    )


def _action(
    name: str, arguments: dict | None = None, *, skill_id: str = "cmd1"
) -> RobotAction:
    intent = SkillIntent(
        envelope=Envelope(robot_id="mock0"), skill_id=skill_id, objective=name
    )
    return RobotSkillAction(name, arguments or {}).to_robot_action(intent)


async def test_mock_capabilities_are_xlerobot_like(tmp_path) -> None:
    runtime = _runtime(tmp_path)

    snapshot = await runtime.start()

    assert snapshot.capabilities.driver_type == "mock"
    assert snapshot.capabilities.action_dimensions is None
    assert snapshot.capabilities.cameras == ["front", "left_wrist", "right_wrist"]
    assert snapshot.capabilities.metadata["body"] == "xlerobot"
    assert snapshot.capabilities.metadata["robot_family"] == "xlerobot"
    assert snapshot.capabilities.metadata["environment"] == "mock"
    assert snapshot.capabilities.metadata["embodiment_profile"] == "xlerobot_mock"
    assert snapshot.capabilities.metadata["control"] == "skill_action"
    assert "move_base" in snapshot.capabilities.metadata["supported_skills"]
    assert "inspect_scene" in snapshot.capabilities.metadata["supported_skills"]
    assert snapshot.health.metrics["readiness"]["base"]["ok"] is True
    assert snapshot.health.metrics["readiness"]["front_camera"]["ok"] is True


async def test_mock_executes_base_and_arm_skills(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    await runtime.start()

    move = await runtime.apply_action(
        _action(
            "move_base", {"direction": "forward", "distance_cm": 25.0}, skill_id="move1"
        )
    )
    turn = await runtime.apply_action(
        _action("turn_base", {"direction": "left", "angle_deg": 90.0}, skill_id="turn1")
    )
    joint = await runtime.apply_action(
        _action(
            "move_arm_joints",
            {"mode": "delta", "joints": {"wrist_roll": 15.0}},
            skill_id="arm1",
        )
    )
    grip = await runtime.apply_action(
        _action("set_gripper", {"action": "close"}, skill_id="grip1")
    )

    assert move.success is True
    assert move.metrics["base_pose"]["x_cm"] == 25.0
    assert turn.metrics["base_pose"]["yaw_deg"] == 90.0
    assert joint.metrics["arm_status"]["joint_states"]["wrist_roll"] == 15.0
    assert grip.metrics["object_held"] == "mock_object"


async def test_mock_readiness_gate_blocks_unavailable_gripper(tmp_path) -> None:
    runtime = _runtime(tmp_path, {"gripper_available": False})
    await runtime.start()

    status = await runtime.apply_action(_action("set_gripper", {"action": "open"}))

    assert status.success is False
    assert status.state == "failed"
    assert "gripper is not ready" in (status.error or "")
    assert status.metrics["last_skill_result"]["failure_mode"] == "readiness_failed"


async def test_mock_rejects_non_skill_action(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    await runtime.start()

    status = await runtime.apply_action(
        RobotAction(envelope=Envelope(robot_id="mock0"), values=[0.0], skill_id="bad")
    )

    assert status.success is False
    assert status.metrics["last_skill_result"]["failure_mode"] == "invalid_action"


async def test_mock_observation_uses_front_camera_and_xlerobot_metadata(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    await runtime.start()

    observation = await runtime.observe()

    assert observation.images
    assert {image.camera for image in observation.images} == {
        "front",
        "left_wrist",
        "right_wrist",
    }
    assert observation.proprioception
    assert observation.raw["body"] == "xlerobot"
    assert observation.raw["camera"]["frame_available"] is True
    assert set(observation.raw["cameras"]) == {"front", "left_wrist", "right_wrist"}


async def test_mock_camera_drop_pattern_emits_missing_frame_signal(tmp_path) -> None:
    runtime = _runtime(tmp_path, {"camera_drop_every_n_observe": 2})
    await runtime.start()

    first = await runtime.observe()
    second = await runtime.observe()

    assert first.images
    assert second.images == []
    assert second.raw["camera"]["frame_available"] is False
    assert second.raw["camera"]["drop_reason"] == "intermittent_drop"
    assert second.raw["cameras"]["left_wrist"]["frame_available"] is False


async def test_mock_set_gripper_can_pick_visible_object(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    await runtime.start()

    picked = await runtime.apply_action(
        _action("set_gripper", {"action": "close", "object": "cup"}, skill_id="pick1")
    )
    observation = await runtime.observe()

    assert picked.success is True
    assert picked.metrics["world"]["held_object"] == "cup"
    assert observation.raw["scene"]["held_object"] == "cup"


async def test_mock_open_gripper_releases_held_object_to_front_workspace(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    await runtime.start()

    picked = await runtime.apply_action(
        _action("set_gripper", {"action": "close", "object": "cup"}, skill_id="pick1")
    )
    released = await runtime.apply_action(
        _action("set_gripper", {"action": "open"}, skill_id="place1")
    )
    observation = await runtime.observe()

    assert picked.success is True
    assert released.success is True
    assert released.metrics["last_skill_result"]["message"] == "gripper opening set"
    assert released.metrics["world"]["objects"]["cup"]["location"] == "front_workspace"
    assert observation.raw["scene"]["held_object"] is None


async def test_mock_inspect_scene_reports_scene(tmp_path) -> None:
    runtime = _runtime(
        tmp_path,
        {
            "world": {"objects": {"cup": {"visible": False, "location": "table"}}},
            "visible_after_scans": {"cup": 1},
        },
    )
    await runtime.start()

    detected = await runtime.apply_action(
        _action("inspect_scene", {"question": "find cup"}, skill_id="scan1")
    )

    assert detected.success is True
    assert detected.metrics["last_skill_result"]["image_count"] == 3
    assert "cup" in detected.metrics["last_skill_result"]["summary"]


async def test_mock_inspect_scene_reveals_hidden_object_after_repeated_scans(
    tmp_path,
) -> None:
    runtime = _runtime(
        tmp_path,
        {
            "world": {
                "objects": {"cup": {"visible": False, "location": "front_workspace"}}
            },
            "visible_after_scans": {"cup": 2},
        },
    )
    await runtime.start()

    await runtime.apply_action(
        _action("inspect_scene", {"question": "find cup"}, skill_id="scan1")
    )
    first = await runtime.latest_observation(max_age_ms=1000)
    await runtime.apply_action(
        _action("inspect_scene", {"question": "find cup"}, skill_id="scan2")
    )
    second = await runtime.latest_observation(max_age_ms=1000)

    assert first is not None
    assert second is not None
    assert "cup" not in first.raw["scene"]["visible_objects"]
    assert "cup" in second.raw["scene"]["visible_objects"]


async def test_mock_inspect_scene_false_negative_hides_visible_object_until_retried(
    tmp_path,
) -> None:
    runtime = _runtime(
        tmp_path,
        {
            "world": {
                "objects": {"cup": {"visible": True, "location": "front_workspace"}}
            },
            "perception_false_negative_scans": {"cup": 1},
        },
    )
    await runtime.start()

    await runtime.apply_action(
        _action("inspect_scene", {"question": "find cup"}, skill_id="scan1")
    )
    first = await runtime.latest_observation(max_age_ms=1000)
    await runtime.apply_action(
        _action("inspect_scene", {"question": "find cup"}, skill_id="scan2")
    )
    second = await runtime.latest_observation(max_age_ms=1000)

    assert first is not None
    assert second is not None
    assert "cup" not in first.raw["scene"]["visible_objects"]
    assert "cup" in second.raw["scene"]["visible_objects"]


async def test_mock_scan_then_grasp_updates_world_state(tmp_path) -> None:
    runtime = _runtime(
        tmp_path,
        {
            "world": {
                "objects": {"cup": {"visible": False, "location": "front_workspace"}}
            },
            "visible_after_scans": {"cup": 1},
        },
    )
    await runtime.start()

    await runtime.apply_action(
        _action(
            "inspect_scene", {"question": "find the cup in front"}, skill_id="scan1"
        )
    )
    picked = await runtime.apply_action(
        _action("set_gripper", {"action": "close", "object": "cup"}, skill_id="pick1")
    )

    assert picked.success is True
    assert picked.metrics["world"]["held_object"] == "cup"


async def test_mock_multistep_scan_pick_release_chain_updates_world_state(
    tmp_path,
) -> None:
    runtime = _runtime(
        tmp_path,
        {
            "world": {
                "objects": {"cup": {"visible": False, "location": "front_workspace"}}
            },
            "visible_after_scans": {"cup": 1},
        },
    )
    await runtime.start()

    scanned = await runtime.apply_action(
        _action(
            "inspect_scene",
            {"question": "look for the cup in front"},
            skill_id="scan1",
        )
    )
    picked = await runtime.apply_action(
        _action("set_gripper", {"action": "close", "object": "cup"}, skill_id="pick1")
    )
    released = await runtime.apply_action(
        _action("set_gripper", {"action": "open"}, skill_id="place1")
    )
    observation = await runtime.observe()

    assert scanned.success is True
    assert picked.success is True
    assert released.success is True
    assert released.metrics["world"]["objects"]["cup"]["location"] == "front_workspace"
    assert released.metrics["world"]["held_object"] is None
    assert observation.raw["scene"]["held_object"] is None
    assert "cup" in observation.raw["scene"]["visible_objects"]


async def test_mock_scripted_failure_is_attempt_aware(tmp_path) -> None:
    runtime = _runtime(
        tmp_path,
        {
            "scripted_failures": [
                {
                    "skill": "stop_motion",
                    "attempt": 1,
                    "message": "stop command timeout",
                    "failure_mode": "timeout",
                }
            ]
        },
    )
    await runtime.start()

    first = await runtime.apply_action(_action("stop_motion", skill_id="stop1"))
    second = await runtime.apply_action(_action("stop_motion", skill_id="stop2"))

    assert first.success is False
    assert first.metrics["last_skill_result"]["failure_mode"] == "timeout"
    assert second.success is True
    assert second.metrics["last_skill_result"]["message"] == "base stopped"


async def test_mock_status_exposes_base_control_diagnostics(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    await runtime.start()

    moved = await runtime.apply_action(
        _action("move_base", {"direction": "forward", "distance_cm": 20.0})
    )
    stopped = await runtime.apply_action(
        _action("stop_motion", {"emergency": False}, skill_id="stop1")
    )

    assert moved.success is True
    assert moved.metrics["base_control"]["last_motion_report"]["kind"] == "move_base"
    assert (
        moved.metrics["base_control"]["last_motion_report"]["base_pose"]["x_cm"] == 20.0
    )
    assert stopped.success is True
    assert stopped.metrics["base_control"]["last_stop_command"]["success"] is True
    assert (
        stopped.metrics["base_control"]["last_motion_report"]["kind"] == "stop_motion"
    )


async def test_mock_transient_readiness_fault_blocks_then_recovers(tmp_path) -> None:
    runtime = _runtime(
        tmp_path,
        {"readiness_faults": [{"resource": "gripper", "until_attempt": 1}]},
    )
    await runtime.start()

    first = await runtime.apply_action(
        _action("set_gripper", {"action": "open"}, skill_id="grip1")
    )
    second = await runtime.apply_action(
        _action("set_gripper", {"action": "open"}, skill_id="grip2")
    )

    assert first.success is False
    assert first.metrics["last_skill_result"]["failure_mode"] == "readiness_failed"
    assert "gripper is not ready" in (first.error or "")
    assert second.success is True
    assert second.metrics["last_skill_result"]["message"] == "gripper opening set"
