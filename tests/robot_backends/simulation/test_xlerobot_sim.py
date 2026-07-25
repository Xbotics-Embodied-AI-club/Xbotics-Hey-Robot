from __future__ import annotations

import asyncio
import importlib.util
import threading
from pathlib import Path

import pytest

from hey_robot.config import RobotSpec
from hey_robot.protocol import Envelope, RobotSkillAction, SkillIntent
from hey_robot.robot_api import (
    RobotCapabilities,
    RobotDriverContext,
    RobotHealth,
)
from hey_robot.robot_runtime.embodiments import get_embodiment_profile
from hey_robot.robot_runtime.manager import create_driver_context


@pytest.fixture
def sim_context() -> RobotDriverContext:
    spec = RobotSpec(type="xlerobot_sim", enabled=True, settings={})
    return create_driver_context("test_sim_robot", spec, "test_deployment")


def _skill_action(name: str, arguments: dict[str, object]) -> object:
    return RobotSkillAction(name, arguments).to_robot_action(_intent(name))


def _intent(name: str) -> SkillIntent:
    return SkillIntent(
        envelope=Envelope(robot_id="test_sim_robot"),
        skill_id=f"test-{name}",
        task_id="task-test",
        intent_kind="skill",
        name=name,
        arguments={},
        objective=f"test {name}",
    )


class TestXLeRobotSimSkillAdapter:
    @staticmethod
    def _adapter() -> object:
        from hey_robot.robot_backends.simulation.skill_adapter import (
            XLeRobotSimSkillAdapter,
        )

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
        assert cmd.vx > 0
        assert cmd.vy == 0
        assert cmd.duration_sec > 0
        assert "left" in cmd.message

    def test_decode_move_base_right(self) -> None:
        adapter = self._adapter()
        cmd = adapter.decode(
            RobotSkillAction("move_base", {"distance_cm": 10, "direction": "right"})
        )
        assert cmd.vx < 0
        assert cmd.vy == 0
        assert cmd.duration_sec > 0
        assert "right" in cmd.message

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

    def test_manipulate_uses_required_model_service(self) -> None:
        from hey_robot.skills.builtins.vla import MANIPULATE

        assert MANIPULATE.required_models == ("manipulate",)
        # A native VLA action can drive base, both arms, and grippers together.
        # The semantic skill therefore owns the whole actuator boundary.
        assert MANIPULATE.resources == ("robot_control", "camera")

    def test_decode_gripper_open_close(self) -> None:
        adapter = self._adapter()
        cmd_open = adapter.decode(RobotSkillAction("set_gripper", {"action": "open"}))
        assert cmd_open.jaw_left == 1.7
        cmd_close = adapter.decode(RobotSkillAction("set_gripper", {"action": "close"}))
        assert cmd_close.jaw_left == 0.0

    def test_decode_uses_embodiment_actuator_layout_and_gripper_range(self) -> None:
        from hey_robot.robot_api import EmbodimentProfile
        from hey_robot.robot_api.classic_profile import ClassicEmbodimentProfile
        from hey_robot.robot_backends.simulation.skill_adapter import (
            XLeRobotSimSkillAdapter,
        )

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
    pytestmark = pytest.mark.skipif(
        importlib.util.find_spec("mujoco") is None,
        reason="requires optional sim dependency: mujoco",
    )

    def test_xlerobot_model_contains_arm_mount_geometry(self) -> None:
        import mujoco

        model = mujoco.MjModel.from_xml_path(
            str(Path("assets/scenes/home_scene.xml").resolve())
        )

        for body_name in (
            "base_link",
            "Base",
            "Rotation_Pitch",
            "Right_Arm_Camera",
        ):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name) >= 0

        for geom_name in (
            "base_link_chassis",
            "base_link_motor",
            "Right_Arm_Camera_visual",
        ):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name) >= 0

    def test_xlerobot_model_contains_calibrated_native_cameras(self) -> None:
        import mujoco

        model = mujoco.MjModel.from_xml_path(
            str(Path("assets/scenes/home_scene.xml").resolve())
        )

        expected = {
            "front": pytest.approx(91.673, abs=1e-3),
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
        from hey_robot.robot_backends.simulation.xlerobot_sim_driver import (
            XLeRobotSimDriver,
        )

        driver = XLeRobotSimDriver(sim_context)
        assert driver.robot_id == "test_sim_robot"
        assert driver.state == "created"

    @pytest.mark.asyncio
    async def test_cancelled_simulation_work_waits_until_writer_stops(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robot_backends.simulation.xlerobot_sim_driver import (
            XLeRobotSimDriver,
        )

        driver = XLeRobotSimDriver(sim_context)
        started = threading.Event()
        stopped = threading.Event()

        def work(stop_event: threading.Event) -> bool:
            started.set()
            stop_event.wait(timeout=1.0)
            stopped.set()
            return False

        task = asyncio.create_task(driver._run_simulation_work(work))
        assert await asyncio.to_thread(started.wait, 1.0)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert stopped.is_set()
        assert driver._active_motion_stop is None
        assert driver.state == "idle"

    @pytest.mark.asyncio
    async def test_driver_start_and_capabilities(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robot_backends.simulation.xlerobot_sim_driver import (
            XLeRobotSimDriver,
        )

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()
        assert driver.state == "idle"

        caps = await driver.capabilities()
        assert isinstance(caps, RobotCapabilities)
        assert caps.driver_type == "xlerobot_sim"
        assert caps.cameras == ["front", "right_wrist"]
        assert caps.metadata["embodiment_profile"] == "xlerobot_sim"

        await driver.close()

    @pytest.mark.asyncio
    async def test_driver_health(self, sim_context: RobotDriverContext) -> None:
        from hey_robot.robot_backends.simulation.xlerobot_sim_driver import (
            XLeRobotSimDriver,
        )

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()

        health = await driver.health()
        assert isinstance(health, RobotHealth)
        assert health.online is True
        assert health.state == "idle"

        await driver.close()

    @pytest.mark.asyncio
    async def test_driver_observe(self, sim_context: RobotDriverContext) -> None:
        from hey_robot.robot_backends.simulation.xlerobot_sim_driver import (
            XLeRobotSimDriver,
        )

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()

        obs = await driver.observe()
        assert obs.frame_id == 1
        assert len(obs.assets) >= 1
        assert obs.assets[0].kind == "image"
        assert "cameras" in obs.metadata

        await driver.close()

    @pytest.mark.asyncio
    async def test_driver_status_exposes_multi_camera_shape(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robot_backends.simulation.xlerobot_sim_driver import (
            XLeRobotSimDriver,
        )

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()
        await driver.observe()

        status = await driver.status()

        assert set(status.metrics["cameras"]) == {"front", "right_wrist"}
        assert status.metrics["readiness"]["front_camera"]["ok"] is True
        assert status.metrics["readiness"]["right_wrist_camera"]["ok"] is True

        await driver.close()

    @pytest.mark.asyncio
    async def test_driver_apply_move_base_action(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robot_backends.simulation.xlerobot_sim_driver import (
            XLeRobotSimDriver,
        )

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()

        skill = RobotSkillAction(
            "move_base", {"distance_cm": 10, "direction": "forward"}
        )
        action = skill.to_robot_action(_intent("move_base"))

        status = await driver.apply_action(action)
        assert status.success is True
        assert status.state == "idle"
        assert status.metrics["base_pose"] == driver._base_pose()

        await driver.close()

    @pytest.mark.asyncio
    async def test_driver_reports_idle_after_completed_action_on_status(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robot_backends.simulation.xlerobot_sim_driver import (
            XLeRobotSimDriver,
        )

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()
        action = RobotSkillAction("stop_motion", {}).to_robot_action(
            _intent("stop_motion")
        )

        action_status = await driver.apply_action(action)
        heartbeat_status = await driver.status()

        assert action_status.state == "idle"
        assert heartbeat_status.state == "idle"
        await driver.close()

    @pytest.mark.asyncio
    async def test_driver_reports_emergency_stop_readiness(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robot_backends.simulation.xlerobot_sim_driver import (
            XLeRobotSimDriver,
        )

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()
        action = RobotSkillAction("stop_motion", {"emergency": True}).to_robot_action(
            _intent("stop_motion")
        )

        status = await driver.apply_action(action)

        assert status.metrics["readiness"]["emergency_stop"] is True
        assert status.metrics["startup_diagnostics"]["safety"]["emergency_stop"] is True
        await driver.close()

    @pytest.mark.asyncio
    async def test_driver_reset(self, sim_context: RobotDriverContext) -> None:
        from hey_robot.robot_backends.simulation.xlerobot_sim_driver import (
            XLeRobotSimDriver,
        )

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()
        await driver.reset()
        assert driver.state == "idle"
        await driver.close()

    @pytest.mark.asyncio
    async def test_driver_close(self, sim_context: RobotDriverContext) -> None:
        from hey_robot.robot_backends.simulation.xlerobot_sim_driver import (
            XLeRobotSimDriver,
        )

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()
        await driver.close()
        assert driver.state == "closed"

    @pytest.mark.asyncio
    async def test_driver_stop_motion_zeros_base_motion(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robot_backends.simulation.xlerobot_sim_driver import (
            XLeRobotSimDriver,
        )

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
        import mujoco

        for lock_name in ("base_x_lock", "base_y_lock", "base_yaw_lock"):
            lock_id = mujoco.mj_name2id(
                driver.model, mujoco.mjtObj.mjOBJ_ACTUATOR, lock_name
            )
            if lock_id >= 0:
                assert driver.data.ctrl[lock_id] == pytest.approx(0.0, abs=1e-9)

        await driver.close()


class TestXLeRobotSimE2EFlow:
    """End-to-end: skill action to sim execution to observation to verification."""

    pytestmark = pytest.mark.skipif(
        importlib.util.find_spec("mujoco") is None,
        reason="requires optional sim dependency: mujoco",
    )

    @pytest.fixture
    def sim_context(self) -> RobotDriverContext:
        spec = RobotSpec(type="xlerobot_sim", enabled=True, settings={})
        return create_driver_context("e2e_sim", spec, "e2e_deploy")

    @pytest.mark.asyncio
    async def test_flow_move_base_action_returns_success(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robot_backends.simulation.xlerobot_sim_driver import (
            XLeRobotSimDriver,
        )

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()

        obs_before = await driver.observe()
        x_before = obs_before.metadata["base_pose"]["x_cm"]

        skill = RobotSkillAction(
            "move_base", {"distance_cm": 10, "direction": "forward"}
        )
        action = skill.to_robot_action(_intent("move_base"))
        status = await driver.apply_action(action)
        assert status.success is True

        # The default home scene allows the base to move.
        obs_after = await driver.observe()
        x_after = obs_after.metadata["base_pose"]["x_cm"]
        assert x_after == pytest.approx(x_before + 10.0, abs=1.0)

        await driver.close()

    @pytest.mark.asyncio
    async def test_flow_turn_base_action_returns_success(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robot_backends.simulation.xlerobot_sim_driver import (
            XLeRobotSimDriver,
        )

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()

        skill = RobotSkillAction("turn_base", {"angle_deg": 45, "direction": "left"})
        action = skill.to_robot_action(_intent("turn_base"))
        status = await driver.apply_action(action)
        assert status.success is True

        await driver.close()

    @pytest.mark.asyncio
    async def test_flow_move_arm_joints_changes_arm_state(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robot_backends.simulation.xlerobot_sim_driver import (
            XLeRobotSimDriver,
        )

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()

        skill = RobotSkillAction(
            "move_arm_joints",
            {"joints": {"shoulder_lift": 0.5, "elbow_flex": -0.3}, "mode": "absolute"},
        )
        action = skill.to_robot_action(_intent("move_arm_joints"))
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
        from hey_robot.robot_backends.simulation.xlerobot_sim_driver import (
            XLeRobotSimDriver,
        )

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()

        # Move away from home
        skill = RobotSkillAction(
            "move_arm_joints",
            {"joints": {"shoulder_lift": 0.2}, "mode": "absolute"},
        )
        action = skill.to_robot_action(_intent("move_arm_joints"))
        await driver.apply_action(action)

        # Home
        home_skill = RobotSkillAction("reset_posture", {})
        home_action = home_skill.to_robot_action(_intent("reset_posture"))
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
        from hey_robot.robot_backends.simulation.xlerobot_sim_driver import (
            XLeRobotSimDriver,
        )

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()

        close_skill = RobotSkillAction("set_gripper", {"action": "close"})
        close_action = close_skill.to_robot_action(_intent("set_gripper"))
        await driver.apply_action(close_action)

        obs_closed = await driver.observe()
        pct_closed = obs_closed.metadata["arm_status"]["gripper_opening_pct"]
        assert pct_closed < 10.0

        open_skill = RobotSkillAction("set_gripper", {"action": "open"})
        open_action = open_skill.to_robot_action(_intent("set_gripper"))
        await driver.apply_action(open_action)

        obs_open = await driver.observe()
        pct_open = obs_open.metadata["arm_status"]["gripper_opening_pct"]
        assert pct_open > 80.0

        await driver.close()

    @pytest.mark.asyncio
    async def test_flow_gripper_cycle_does_not_move_arm_joints(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robot_backends.simulation.xlerobot_sim_driver import (
            XLeRobotSimDriver,
        )

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
        open_action = open_skill.to_robot_action(_intent("set_gripper"))
        await driver.apply_action(open_action)

        close_skill = RobotSkillAction("set_gripper", {"action": "close"})
        close_action = close_skill.to_robot_action(_intent("set_gripper"))
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
        from hey_robot.robot_backends.simulation.xlerobot_sim_driver import (
            XLeRobotSimDriver,
        )

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
        from hey_robot.robot_backends.simulation.xlerobot_sim_driver import (
            XLeRobotSimDriver,
        )

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()

        # Base is locked in the single-arm scene; verify sequenced skills
        # complete without errors.
        actions = [
            ("move_base", {"distance_cm": 5, "direction": "forward"}),
            ("turn_base", {"angle_deg": 90, "direction": "right"}),
            ("set_gripper", {"action": "open"}),
            ("set_gripper", {"action": "close"}),
            ("reset_posture", {}),
        ]
        for name, args in actions:
            skill = RobotSkillAction(name, args)
            action = skill.to_robot_action(_intent(name))
            status = await driver.apply_action(action)
            assert status.success, f"skill {name} should succeed"

        await driver.close()

    @pytest.mark.asyncio
    async def test_flow_reset_clears_state(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robot_backends.simulation.xlerobot_sim_driver import (
            XLeRobotSimDriver,
        )

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()
        initial_obs = await driver.observe()
        initial_x_cm = initial_obs.metadata["base_pose"]["x_cm"]

        skill = RobotSkillAction(
            "move_base", {"distance_cm": 20, "direction": "forward"}
        )
        action = skill.to_robot_action(_intent("move_base"))
        await driver.apply_action(action)

        await driver.reset()
        obs = await driver.observe()
        assert obs.metadata["base_pose"]["x_cm"] == pytest.approx(initial_x_cm, abs=0.1)
        await driver.close()

    @pytest.mark.asyncio
    async def test_flow_arm_pose_does_not_cause_base_drift_after_motion(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robot_backends.simulation.xlerobot_sim_driver import (
            XLeRobotSimDriver,
        )

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
        from hey_robot.robot_backends.simulation.xlerobot_sim_driver import (
            XLeRobotSimDriver,
        )

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
    async def test_flow_reset_returns_robot_to_idle(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robot_backends.simulation.xlerobot_sim_driver import (
            XLeRobotSimDriver,
        )

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()

        # Base is locked at origin in the single-arm scene;
        # verify reset completes and returns robot to idle state.
        await driver.reset()
        assert driver.state == "idle"

        assert driver.data is not None
        assert driver.data.qvel[0] == pytest.approx(0.0, abs=1e-9)
        assert driver.data.qvel[1] == pytest.approx(0.0, abs=1e-9)
        assert driver.data.qvel[2] == pytest.approx(0.0, abs=1e-9)
        await driver.close()

    @pytest.mark.asyncio
    async def test_flow_scene_render_contains_robot_geometry(
        self, sim_context: RobotDriverContext
    ) -> None:
        from hey_robot.robot_backends.simulation.xlerobot_sim_driver import (
            XLeRobotSimDriver,
        )

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
        from hey_robot.robot_backends.simulation.xlerobot_sim_driver import (
            XLeRobotSimDriver,
        )

        driver = XLeRobotSimDriver(sim_context)
        await driver.start()

        obs = await driver.observe()
        front = next(asset.data for asset in obs.assets if asset.name == "front")
        # Camera renders a non-blank image with visible content.
        assert front.shape == (480, 640, 3)
        assert float(front.std()) > 5.0
        await driver.close()
