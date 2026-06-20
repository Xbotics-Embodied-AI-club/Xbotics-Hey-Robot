from __future__ import annotations

from pathlib import Path

import pytest

from hey_robot.config import RobotSpec
from hey_robot.protocol import Envelope, SkillIntent
from hey_robot.robots import get_embodiment_profile
from hey_robot.robots.base import RobotCapabilities, RobotDriverContext, RobotHealth
from hey_robot.skills import RobotSkillAction


@pytest.fixture
def sim_context() -> RobotDriverContext:
    spec = RobotSpec(type="xlerobot_sim", enabled=True, settings={})
    return RobotDriverContext(
        robot_id="test_sim_robot",
        spec=spec,
        deployment_id="test_deployment",
        embodiment=get_embodiment_profile(spec),
    )


def _skill_action(name: str, arguments: dict[str, object]) -> object:
    return RobotSkillAction(name, arguments).to_robot_action(
        SkillIntent(envelope=Envelope(), name=name)
    )


class TestXLeRobotSimSkillAdapter:
    @staticmethod
    def _adapter() -> object:
        from hey_robot.robots.simulation.skill_adapter import XLeRobotSimSkillAdapter

        return XLeRobotSimSkillAdapter(
            embodiment=get_embodiment_profile(RobotSpec(type="xlerobot_sim"))
        )

    def test_decode_move_base_forward(self) -> None:
        adapter = self._adapter()
        cmd = adapter.decode(
            RobotSkillAction("move_base", {"distance_cm": 20, "direction": "forward"})
        )
        # Forward maps to vy (world X) so the robot heads toward the table at x=2.0.
        assert cmd.vy > 0
        assert cmd.vx == 0
        assert cmd.duration_sec > 0
        assert "forward" in cmd.message

    def test_decode_move_base_backward(self) -> None:
        adapter = self._adapter()
        cmd = adapter.decode(
            RobotSkillAction("move_base", {"distance_cm": 15, "direction": "backward"})
        )
        # Backward negates vy (world X), no lateral (vx) drift.
        assert cmd.vy < 0
        assert cmd.vx == 0
        assert cmd.duration_sec > 0
        assert "backward" in cmd.message

    def test_decode_move_base_left(self) -> None:
        adapter = self._adapter()
        cmd = adapter.decode(
            RobotSkillAction("move_base", {"distance_cm": 10, "direction": "left"})
        )
        assert cmd.vx < 0
        assert cmd.vy == 0
        assert cmd.duration_sec > 0
        assert "left" in cmd.message

    def test_decode_turn_base_left(self) -> None:
        adapter = self._adapter()
        cmd = adapter.decode(
            RobotSkillAction("turn_base", {"angle_deg": 90, "direction": "left"})
        )
        assert cmd.vw > 0
        assert cmd.duration_sec > 0

    def test_decode_stop_motion(self) -> None:
        adapter = self._adapter()
        cmd = adapter.decode(RobotSkillAction("stop_motion", {}))
        assert cmd.vx == 0
        assert cmd.vy == 0
        assert cmd.vw == 0

    def test_decode_reset_posture(self) -> None:
        adapter = self._adapter()
        cmd = adapter.decode(RobotSkillAction("reset_posture", {}))
        assert len(cmd.arm_targets) == 12  # 6 joints x 2 arms

    def test_decode_set_arm_pose(self) -> None:
        adapter = self._adapter()
        cmd = adapter.decode(RobotSkillAction("set_arm_pose", {"pose_name": "home"}))
        assert len(cmd.arm_targets) == 12

    def test_decode_unknown_pose_raises(self) -> None:
        adapter = self._adapter()
        with pytest.raises(ValueError, match="unknown named pose"):
            adapter.decode(
                RobotSkillAction("set_arm_pose", {"pose_name": "nonexistent"})
            )

    def test_vla_manipulation_uses_external_capability(self) -> None:
        from hey_robot.skills.builtin.capability import VLAManipulationSkill

        assert VLAManipulationSkill.spec.external_capability == "vla_manipulation"
        assert VLAManipulationSkill.spec.required_resources == (
            "arm",
            "gripper",
            "camera",
        )

    def test_decode_gripper_open_close(self) -> None:
        adapter = self._adapter()
        cmd_open = adapter.decode(RobotSkillAction("set_gripper", {"action": "open"}))
        assert cmd_open.jaw_left == 1.2
        cmd_close = adapter.decode(RobotSkillAction("set_gripper", {"action": "close"}))
        assert cmd_close.jaw_left == 0.0

    def test_decode_uses_embodiment_actuator_layout_and_gripper_range(self) -> None:
        from hey_robot.robots.classic import ClassicEmbodimentProfile
        from hey_robot.robots.embodiments import EmbodimentProfile
        from hey_robot.robots.simulation.skill_adapter import XLeRobotSimSkillAdapter

        embodiment = EmbodimentProfile(
            name="custom_sim_body",
            robot_family="xlerobot",
            environment="sim",
            named_poses={"home": {"shoulder_pan": 0.0, "gripper": 0.05}},
            actuator_layout={"shoulder_pan": (21, 22), "gripper": (23, 24)},
            gripper_range=(0.01, 0.05),
        )
        classic_profile = ClassicEmbodimentProfile.from_embodiment(embodiment)

        assert classic_profile is not None
        assert classic_profile.joint_actuator_pair("gripper") == (23, 24)
        assert classic_profile.gripper_open_value == pytest.approx(0.05)

        adapter = XLeRobotSimSkillAdapter(embodiment=embodiment)

        home = adapter.decode(RobotSkillAction("reset_posture", {}))
        opened = adapter.decode(RobotSkillAction("set_gripper", {"action": "open"}))
        closed = adapter.decode(RobotSkillAction("set_gripper", {"action": "close"}))

        assert home.arm_targets[21] == pytest.approx(0.0)
        assert home.arm_targets[22] == pytest.approx(0.0)
        assert home.arm_targets[23] == pytest.approx(0.05)
        assert home.arm_targets[24] == pytest.approx(0.05)
        assert opened.jaw_left == pytest.approx(0.05)
        assert closed.jaw_left == pytest.approx(0.01)

    def test_decode_unsupported_skill_raises(self) -> None:
        adapter = self._adapter()
        with pytest.raises(ValueError, match="unsupported"):
            adapter.decode(RobotSkillAction("nonexistent_skill", {}))


class TestXLeRobotSimDriver:
    def test_xlerobot_model_contains_arm_mount_geometry(self) -> None:
        import mujoco

        model = mujoco.MjModel.from_xml_path(
            str(Path("assets/robots/xlerobot/scene.xml").resolve())
        )

        for body_name in (
            "base_link",
            "Base",
            "Base_2",
            "Rotation_Pitch",
            "Rotation_Pitch_2",
            "Right_Arm_Camera",
            "Left_Arm_Camera",
        ):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name) >= 0

        for geom_name in (
            "base_link_chassis",
            "base_link_motor",
            "Right_Arm_Camera_visual",
            "Left_Arm_Camera_visual",
        ):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name) >= 0

    def test_xlerobot_model_contains_calibrated_native_cameras(self) -> None:
        import mujoco

        model = mujoco.MjModel.from_xml_path(
            str(Path("assets/robots/xlerobot/scene.xml").resolve())
        )

        expected = {
            "front": pytest.approx(91.673, abs=1e-3),
            "left_wrist": pytest.approx(74.485, abs=1e-3),
            "right_wrist": pytest.approx(74.485, abs=1e-3),
        }
        for camera_name, fovy in expected.items():
            camera_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
            )
            assert camera_id >= 0
            assert model.cam_pos[camera_id].tolist() == pytest.approx(
                [0.0, 0.04, 0.0], abs=1e-6
            )
            assert model.cam_fovy[camera_id] == fovy

        for actuator_name in ("head_pan_hold", "head_tilt_hold"):
            assert (
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                >= 0
            )

    def test_driver_instantiation(self, sim_context: RobotDriverContext) -> None:
        from hey_robot.robots.simulation.xlerobot_sim_driver import XLeRobotSimDriver

        driver = XLeRobotSimDriver(sim_context)
        assert driver.robot_id == "test_sim_robot"
        assert driver.state == "created"

    @pytest.mark.asyncio
    async def test_driver_start_and_capabilities(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robots.simulation.xlerobot_sim_driver import XLeRobotSimDriver

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()
        assert driver.state == "idle"

        caps = await driver.capabilities()
        assert isinstance(caps, RobotCapabilities)
        assert caps.driver_type == "xlerobot_sim"
        assert caps.cameras == ["front", "left_wrist", "right_wrist"]
        assert caps.metadata["embodiment_profile"] == "xlerobot_sim"

        await driver.close()

    @pytest.mark.asyncio
    async def test_driver_health(self, sim_context: RobotDriverContext) -> None:
        from hey_robot.robots.simulation.xlerobot_sim_driver import XLeRobotSimDriver

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()

        health = await driver.health()
        assert isinstance(health, RobotHealth)
        assert health.online is True
        assert health.state == "idle"

        await driver.close()

    @pytest.mark.asyncio
    async def test_driver_observe(self, sim_context: RobotDriverContext) -> None:
        from hey_robot.robots.simulation.xlerobot_sim_driver import XLeRobotSimDriver

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()

        obs = await driver.observe()
        assert obs.frame_id == 1
        assert len(obs.assets) == 3
        assert obs.assets[0].kind == "image"
        assert {asset.name for asset in obs.assets} == {
            "front",
            "left_wrist",
            "right_wrist",
        }
        assert "cameras" in obs.metadata

        await driver.close()

    @pytest.mark.asyncio
    async def test_driver_status_exposes_multi_camera_shape(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robots.simulation.xlerobot_sim_driver import XLeRobotSimDriver

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()
        await driver.observe()

        status = await driver.status()

        assert set(status.metrics["cameras"]) == {"front", "left_wrist", "right_wrist"}
        assert status.metrics["readiness"]["front_camera"]["ok"] is True
        assert status.metrics["readiness"]["left_wrist_camera"]["ok"] is True
        assert status.metrics["readiness"]["right_wrist_camera"]["ok"] is True

        await driver.close()

    @pytest.mark.asyncio
    async def test_driver_apply_move_base_action(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robots.simulation.xlerobot_sim_driver import XLeRobotSimDriver

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()

        skill = RobotSkillAction(
            "move_base", {"distance_cm": 10, "direction": "forward"}
        )
        action = skill.to_robot_action(
            SkillIntent(envelope=Envelope(), name="move_base")
        )

        status = await driver.apply_action(action)
        assert status.success is True
        assert status.state == "skill_completed"

        await driver.close()

    @pytest.mark.asyncio
    async def test_driver_reports_idle_after_completed_action_on_status(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robots.simulation.xlerobot_sim_driver import XLeRobotSimDriver

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()
        action = RobotSkillAction("stop_motion", {}).to_robot_action(
            SkillIntent(envelope=Envelope(), name="stop_motion")
        )

        action_status = await driver.apply_action(action)
        heartbeat_status = await driver.status()

        assert action_status.state == "skill_completed"
        assert heartbeat_status.state == "idle"
        await driver.close()

    @pytest.mark.asyncio
    async def test_driver_reports_emergency_stop_readiness(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robots.simulation.xlerobot_sim_driver import XLeRobotSimDriver

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()
        action = RobotSkillAction("stop_motion", {"emergency": True}).to_robot_action(
            SkillIntent(envelope=Envelope(), name="stop_motion")
        )

        status = await driver.apply_action(action)

        assert status.metrics["readiness"]["emergency_stop"] is True
        assert status.metrics["startup_diagnostics"]["safety"]["emergency_stop"] is True
        await driver.close()

    @pytest.mark.asyncio
    async def test_driver_reset(self, sim_context: RobotDriverContext) -> None:
        from hey_robot.robots.simulation.xlerobot_sim_driver import XLeRobotSimDriver

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()
        await driver.reset()
        assert driver.state == "idle"
        await driver.close()

    @pytest.mark.asyncio
    async def test_driver_close(self, sim_context: RobotDriverContext) -> None:
        from hey_robot.robots.simulation.xlerobot_sim_driver import XLeRobotSimDriver

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()
        await driver.close()
        assert driver.state == "closed"

    @pytest.mark.asyncio
    async def test_driver_stop_motion_zeros_base_motion(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robots.simulation.xlerobot_sim_driver import XLeRobotSimDriver

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()

        await driver.apply_action(
            _skill_action("move_base", {"distance_cm": 10, "direction": "forward"})
        )
        await driver.apply_action(_skill_action("stop_motion", {}))

        assert driver.data is not None
        assert driver.data.qvel[0] == pytest.approx(0.0, abs=1e-9)
        assert driver.data.qvel[1] == pytest.approx(0.0, abs=1e-9)
        assert driver.data.qvel[2] == pytest.approx(0.0, abs=1e-9)
        assert driver.data.ctrl[15] == pytest.approx(0.0, abs=1e-9)
        assert driver.data.ctrl[16] == pytest.approx(0.0, abs=1e-9)
        assert driver.data.ctrl[17] == pytest.approx(0.0, abs=1e-9)

        await driver.close()


class TestXLeRobotSimE2EFlow:
    """End-to-end: skill action to sim execution to observation to verification."""

    @pytest.fixture
    def sim_context(self) -> RobotDriverContext:
        spec = RobotSpec(type="xlerobot_sim", enabled=True, settings={})
        return RobotDriverContext(
            robot_id="e2e_sim",
            spec=spec,
            deployment_id="e2e_deploy",
            embodiment=get_embodiment_profile(spec),
        )

    @pytest.mark.asyncio
    async def test_flow_move_base_forward_changes_position(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robots.simulation.xlerobot_sim_driver import XLeRobotSimDriver

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()

        obs_before = await driver.observe()
        x_before = obs_before.metadata["base_pose"]["x_cm"]

        skill = RobotSkillAction(
            "move_base", {"distance_cm": 10, "direction": "forward"}
        )
        action = skill.to_robot_action(
            SkillIntent(envelope=Envelope(), name="move_base")
        )
        status = await driver.apply_action(action)
        assert status.success is True

        obs_after = await driver.observe()
        x_after = obs_after.metadata["base_pose"]["x_cm"]

        # Forward uses vy, which maps to world X at yaw=0 (toward the table).
        assert x_after > x_before + 1.0, (
            f"forward should increase world x: {x_before:.2f} -> {x_after:.2f}"
        )
        await driver.close()

    @pytest.mark.asyncio
    async def test_flow_turn_base_changes_yaw(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robots.simulation.xlerobot_sim_driver import XLeRobotSimDriver

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()

        obs_before = await driver.observe()
        yaw_before = obs_before.metadata["base_pose"]["yaw_deg"]

        skill = RobotSkillAction("turn_base", {"angle_deg": 45, "direction": "left"})
        action = skill.to_robot_action(
            SkillIntent(envelope=Envelope(), name="turn_base")
        )
        status = await driver.apply_action(action)
        assert status.success is True

        obs_after = await driver.observe()
        yaw_after = obs_after.metadata["base_pose"]["yaw_deg"]

        assert yaw_after > yaw_before, (
            f"robot should have turned left: {yaw_before} -> {yaw_after}"
        )
        await driver.close()

    @pytest.mark.asyncio
    async def test_flow_move_arm_joints_changes_arm_state(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robots.simulation.xlerobot_sim_driver import XLeRobotSimDriver

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()

        skill = RobotSkillAction(
            "move_arm_joints",
            {"joints": {"shoulder_lift": 0.5, "elbow_flex": -0.3}, "mode": "absolute"},
        )
        action = skill.to_robot_action(
            SkillIntent(envelope=Envelope(), name="move_arm_joints")
        )
        status = await driver.apply_action(action)
        assert status.success is True

        obs = await driver.observe()
        arm = obs.metadata["arm_status"]
        assert arm["joint_states"]["shoulder_lift"] == pytest.approx(0.5, abs=0.01)
        assert arm["joint_states"]["elbow_flex"] == pytest.approx(-0.3, abs=0.01)
        await driver.close()

    @pytest.mark.asyncio
    async def test_flow_reset_posture_resets_joints(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robots.simulation.xlerobot_sim_driver import XLeRobotSimDriver

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()

        # Move away from home
        skill = RobotSkillAction(
            "move_arm_joints",
            {"joints": {"shoulder_lift": 0.2}, "mode": "absolute"},
        )
        action = skill.to_robot_action(
            SkillIntent(envelope=Envelope(), name="move_arm_joints")
        )
        await driver.apply_action(action)

        # Home
        home_skill = RobotSkillAction("reset_posture", {})
        home_action = home_skill.to_robot_action(
            SkillIntent(envelope=Envelope(), name="reset_posture")
        )
        status = await driver.apply_action(home_action)
        assert status.success is True

        obs = await driver.observe()
        arm = obs.metadata["arm_status"]
        assert arm["joint_states"]["shoulder_lift"] == pytest.approx(0.8, abs=0.01)
        assert arm["joint_states"]["elbow_flex"] == pytest.approx(0.7, abs=0.01)
        assert arm["joint_states"]["wrist_flex"] == pytest.approx(-0.6, abs=0.01)
        await driver.close()

    @pytest.mark.asyncio
    async def test_flow_gripper_open_close_cycle(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robots.simulation.xlerobot_sim_driver import XLeRobotSimDriver

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()

        close_skill = RobotSkillAction("set_gripper", {"action": "close"})
        close_action = close_skill.to_robot_action(
            SkillIntent(envelope=Envelope(), name="set_gripper")
        )
        await driver.apply_action(close_action)

        obs_closed = await driver.observe()
        pct_closed = obs_closed.metadata["arm_status"]["gripper_opening_pct"]
        assert pct_closed == pytest.approx(0.0, abs=0.5)

        open_skill = RobotSkillAction("set_gripper", {"action": "open"})
        open_action = open_skill.to_robot_action(
            SkillIntent(envelope=Envelope(), name="set_gripper")
        )
        await driver.apply_action(open_action)

        obs_open = await driver.observe()
        pct_open = obs_open.metadata["arm_status"]["gripper_opening_pct"]
        assert pct_open > 80.0

        await driver.close()

    @pytest.mark.asyncio
    async def test_flow_gripper_cycle_does_not_move_arm_joints(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robots.simulation.xlerobot_sim_driver import XLeRobotSimDriver

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()

        gripper_indices = set(driver.adapter.gripper_actuator_indices() or ())
        arm_indices = set(driver.adapter.arm_actuator_indices("left")) | set(
            driver.adapter.arm_actuator_indices("right")
        )
        locked_indices = sorted(arm_indices - gripper_indices)
        before = {
            idx: driver._joint_position_for_actuator(idx) for idx in locked_indices
        }

        open_skill = RobotSkillAction("set_gripper", {"action": "open"})
        open_action = open_skill.to_robot_action(
            SkillIntent(envelope=Envelope(), name="set_gripper")
        )
        await driver.apply_action(open_action)

        close_skill = RobotSkillAction("set_gripper", {"action": "close"})
        close_action = close_skill.to_robot_action(
            SkillIntent(envelope=Envelope(), name="set_gripper")
        )
        await driver.apply_action(close_action)

        after = {
            idx: driver._joint_position_for_actuator(idx) for idx in locked_indices
        }
        assert after == pytest.approx(before, abs=1e-6)

        await driver.close()

    @pytest.mark.asyncio
    async def test_flow_observe_returns_image_with_correct_shape(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robots.simulation.xlerobot_sim_driver import XLeRobotSimDriver

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()

        obs = await driver.observe()
        image = obs.assets[0].data
        assert image.shape == (480, 640, 3)
        assert image.dtype == "uint8"
        await driver.close()

    @pytest.mark.asyncio
    async def test_flow_multi_skill_sequence(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robots.simulation.xlerobot_sim_driver import XLeRobotSimDriver

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()

        obs0 = await driver.observe()
        x0 = obs0.metadata["base_pose"]["x_cm"]

        # Forward 5cm: at yaw 0, forward (vy) maps to world X.
        fwd = RobotSkillAction("move_base", {"distance_cm": 5, "direction": "forward"})
        await driver.apply_action(
            fwd.to_robot_action(SkillIntent(envelope=Envelope(), name="move_base"))
        )
        obs1 = await driver.observe()
        assert obs1.metadata["base_pose"]["x_cm"] > x0 + 1.0, (
            "forward should increase world x at yaw 0"
        )

        # Turn right 90 degrees.
        yaw_before = obs1.metadata["base_pose"]["yaw_deg"]
        turn = RobotSkillAction("turn_base", {"angle_deg": 90, "direction": "right"})
        await driver.apply_action(
            turn.to_robot_action(SkillIntent(envelope=Envelope(), name="turn_base"))
        )
        obs2 = await driver.observe()
        assert obs2.metadata["base_pose"]["yaw_deg"] < yaw_before, (
            "right turn should decrease yaw"
        )

        # Forward again: at yaw -90°, forward (vy) maps to -world Y.
        y_before = obs2.metadata["base_pose"]["y_cm"]
        fwd2 = RobotSkillAction("move_base", {"distance_cm": 5, "direction": "forward"})
        await driver.apply_action(
            fwd2.to_robot_action(SkillIntent(envelope=Envelope(), name="move_base"))
        )
        obs3 = await driver.observe()
        assert obs3.metadata["base_pose"]["y_cm"] < y_before - 1.0, (
            "second forward should decrease world y after right turn"
        )
        await driver.close()

    @pytest.mark.asyncio
    async def test_flow_reset_clears_state(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robots.simulation.xlerobot_sim_driver import XLeRobotSimDriver

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()

        skill = RobotSkillAction(
            "move_base", {"distance_cm": 20, "direction": "forward"}
        )
        action = skill.to_robot_action(
            SkillIntent(envelope=Envelope(), name="move_base")
        )
        await driver.apply_action(action)

        await driver.reset()
        obs = await driver.observe()
        assert obs.metadata["base_pose"]["x_cm"] == pytest.approx(0.0, abs=0.1)
        await driver.close()

    @pytest.mark.asyncio
    async def test_flow_arm_pose_does_not_cause_base_drift_after_motion(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robots.simulation.xlerobot_sim_driver import XLeRobotSimDriver

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()

        await driver.apply_action(
            _skill_action("move_base", {"distance_cm": 20, "direction": "forward"})
        )
        await driver.apply_action(
            _skill_action("turn_base", {"angle_deg": 45, "direction": "left"})
        )
        await driver.apply_action(
            _skill_action("move_base", {"distance_cm": 15, "direction": "forward"})
        )
        pose_before = driver._base_pose()

        status = await driver.apply_action(
            _skill_action("set_arm_pose", {"pose_name": "pregrasp"})
        )
        assert status.success is True
        pose_after = driver._base_pose()

        assert pose_after["x_cm"] == pytest.approx(pose_before["x_cm"], abs=0.2)
        assert pose_after["y_cm"] == pytest.approx(pose_before["y_cm"], abs=0.2)
        assert pose_after["yaw_deg"] == pytest.approx(pose_before["yaw_deg"], abs=0.2)
        await driver.close()

    @pytest.mark.asyncio
    async def test_flow_gripper_cycle_does_not_cause_base_drift_after_motion(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robots.simulation.xlerobot_sim_driver import XLeRobotSimDriver

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()

        await driver.apply_action(
            _skill_action("move_base", {"distance_cm": 20, "direction": "forward"})
        )
        await driver.apply_action(
            _skill_action("turn_base", {"angle_deg": 45, "direction": "left"})
        )
        await driver.apply_action(
            _skill_action("move_base", {"distance_cm": 15, "direction": "forward"})
        )
        anchor_pose = driver._base_pose()

        close_status = await driver.apply_action(
            _skill_action("set_gripper", {"action": "close"})
        )
        assert close_status.success is True
        pose_after_close = driver._base_pose()

        open_status = await driver.apply_action(
            _skill_action("set_gripper", {"action": "open"})
        )
        assert open_status.success is True
        pose_after_open = driver._base_pose()

        for pose in (pose_after_close, pose_after_open):
            assert pose["x_cm"] == pytest.approx(anchor_pose["x_cm"], abs=0.2)
            assert pose["y_cm"] == pytest.approx(anchor_pose["y_cm"], abs=0.2)
            assert pose["yaw_deg"] == pytest.approx(anchor_pose["yaw_deg"], abs=0.2)
        await driver.close()

    @pytest.mark.asyncio
    async def test_flow_reset_clears_pose_and_base_velocity(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robots.simulation.xlerobot_sim_driver import XLeRobotSimDriver

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()

        await driver.apply_action(
            _skill_action("move_base", {"distance_cm": 20, "direction": "forward"})
        )
        await driver.apply_action(
            _skill_action("turn_base", {"angle_deg": 45, "direction": "left"})
        )

        await driver.reset()

        assert driver.data is not None
        obs = await driver.observe()
        pose = obs.metadata["base_pose"]
        assert pose["x_cm"] == pytest.approx(0.0, abs=0.1)
        assert pose["y_cm"] == pytest.approx(0.0, abs=0.1)
        assert pose["yaw_deg"] == pytest.approx(0.0, abs=0.2)
        assert driver.data.qvel[0] == pytest.approx(0.0, abs=1e-9)
        assert driver.data.qvel[1] == pytest.approx(0.0, abs=1e-9)
        assert driver.data.qvel[2] == pytest.approx(0.0, abs=1e-9)
        await driver.close()

    @pytest.mark.asyncio
    async def test_flow_scene_render_contains_robot_geometry(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robots.simulation.xlerobot_sim_driver import XLeRobotSimDriver

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()

        obs = await driver.observe()
        image = obs.assets[0].data
        # A blank or failed render tends to have near-zero variance.
        assert float(image.std()) > 5.0
        # The repaired scene should expose visible robot/cart geometry.
        assert int(image[:, :, 2].max()) >= 100
        await driver.close()

    @pytest.mark.asyncio
    async def test_front_camera_sees_table_target(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robots.simulation.xlerobot_sim_driver import XLeRobotSimDriver

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()

        obs = await driver.observe()
        front = next(asset.data for asset in obs.assets if asset.name == "front")
        red_target_pixels = (
            (front[:, :, 0] > 120) & (front[:, :, 1] < 100) & (front[:, :, 2] < 100)
        )
        table_pixels = (
            (front[:, :, 0] > 70)
            & (front[:, :, 0] < 150)
            & (front[:, :, 1] > 30)
            & (front[:, :, 1] < 100)
            & (front[:, :, 2] < 90)
        )

        assert int(red_target_pixels.sum()) >= 20
        assert int(table_pixels.sum()) >= 1000
        await driver.close()
