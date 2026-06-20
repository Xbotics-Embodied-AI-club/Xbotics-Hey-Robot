from __future__ import annotations

import pytest

from hey_robot.capability.vla.schemas import VLAConfig, VLARequest, VLAResult


class TestVLAConfig:
    def test_defaults(self) -> None:
        cfg = VLAConfig()
        assert cfg.policy_runtime == "fake"
        assert cfg.arm == "right"
        assert cfg.fps == 25
        assert cfg.action_horizon == 16
        assert cfg.action_mode == "absolute_joint_position_rad"
        assert cfg.execution_time_sec == 10.0
        assert cfg.camera_names == ("front", "right_wrist")
        assert cfg.task_prompt == "Pick up the object."
        assert cfg.language_key == "language"
        assert cfg.camera_key_map == {}
        assert cfg.state_key_map == {}

    def test_rejects_invalid_arm(self) -> None:
        with pytest.raises(ValueError, match="arm must be"):
            VLAConfig(arm="head")

    def test_rejects_invalid_action_mode(self) -> None:
        with pytest.raises(ValueError, match="action_mode must be"):
            VLAConfig(action_mode="velocity_control")

    def test_accepts_valid_action_modes(self) -> None:
        for mode in (
            "absolute_joint_position_rad",
            "delta_joint_position_rad",
            "normalized_joint_position",
        ):
            cfg = VLAConfig(action_mode=mode)
            assert cfg.action_mode == mode

    def test_accepts_left_arm(self) -> None:
        cfg = VLAConfig(arm="left")
        assert cfg.arm == "left"

    def test_frozen_prevents_mutation(self) -> None:
        cfg = VLAConfig()
        with pytest.raises(AttributeError):
            cfg.fps = 30  # type: ignore[misc]

    def test_camera_key_map_via_field(self) -> None:
        cfg = VLAConfig(camera_key_map={"front": "camera1"})
        assert cfg.camera_key_map == {"front": "camera1"}

    def test_state_key_map_via_field(self) -> None:
        cfg = VLAConfig(state_key_map={"single_arm": "arm"})
        assert cfg.state_key_map == {"single_arm": "arm"}


class TestVLARequest:
    def test_minimal_construction(self) -> None:
        req = VLARequest(config=VLAConfig())
        assert req.skill_id is None
        assert req.episode_id is None
        assert req.arguments == {}

    def test_full_construction(self) -> None:
        cfg = VLAConfig(task_prompt="grasp")
        req = VLARequest(
            config=cfg,
            skill_id="skill-1",
            episode_id="ep-1",
            arguments={"object": "cube"},
        )
        assert req.config.task_prompt == "grasp"
        assert req.skill_id == "skill-1"
        assert req.episode_id == "ep-1"
        assert req.arguments == {"object": "cube"}


class TestVLAResult:
    def test_success_result(self) -> None:
        r = VLAResult(success=True, summary="done")
        assert r.success is True
        assert r.status == "completed"
        assert r.failure_mode is None

    def test_failed_result_with_metrics(self) -> None:
        r = VLAResult(
            success=False,
            summary="timeout",
            status="failed",
            failure_mode="policy_timeout",
            error="timed out",
            metrics={"duration_sec": 5.0},
        )
        assert r.success is False
        assert r.failure_mode == "policy_timeout"
        assert r.metrics["duration_sec"] == 5.0

    def test_cancelled_result(self) -> None:
        r = VLAResult(
            success=False,
            summary="cancelled",
            status="cancelled",
            failure_mode="cancelled",
        )
        assert r.status == "cancelled"
