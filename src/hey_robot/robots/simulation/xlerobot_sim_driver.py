from __future__ import annotations

import asyncio
import contextlib
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from hey_robot.logging import HeyRobotLogger

if TYPE_CHECKING:
    from hey_robot.capability.vla.io_adapter import VLAIOAdapter
from hey_robot.perception import DriverObservation, ObservationAsset
from hey_robot.protocol import Envelope, RobotAction, RobotStatus
from hey_robot.robots.base import RobotCapabilities, RobotDriverContext, RobotHealth
from hey_robot.robots.simulation.skill_adapter import XLeRobotSimSkillAdapter
from hey_robot.skills import RobotSkillAction, RobotSkillResult
from hey_robot.skills.contracts import SkillContractRuntime

logger = HeyRobotLogger(name="xlerobot_sim")
_ROBOT_BODY = "base_link"
_DEFAULT_SIM_CAMERA_LAYOUT: dict[str, dict[str, Any]] = {
    "front": {
        "camera_name": "front",
        "prefer_native": True,
        "body": "head_tilt_link",
        "distance": 2.2,
        "azimuth": 180.0,
        "elevation": -10.0,
        "lookat": [0.0, 0.0, 0.0],
    },
    "left_wrist": {
        "camera_name": "left_wrist",
        "prefer_native": True,
        "body": "Left_Arm_Camera",
        "distance": 0.35,
        "azimuth": 180.0,
        "elevation": -15.0,
        "lookat": [0.0, 0.0, 0.0],
    },
    "right_wrist": {
        "camera_name": "right_wrist",
        "prefer_native": True,
        "body": "Right_Arm_Camera",
        "distance": 0.35,
        "azimuth": 180.0,
        "elevation": -15.0,
        "lookat": [0.0, 0.0, 0.0],
    },
}

# SDK joint names (in the canonical order used by arm_status / arm_joints).
_ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def _resolve_mjcf_path(settings: dict[str, Any]) -> Path:
    raw = settings.get("mjcf_path") or settings.get("mjcf")
    if raw:
        p = Path(raw)
        if p.is_absolute():
            return p
        return Path.cwd() / p
    return Path.cwd() / "assets" / "robots" / "xlerobot" / "scene.xml"


class XLeRobotSimDriver:
    """MuJoCo simulation driver implementing the RobotDriver protocol."""

    def __init__(self, context: RobotDriverContext) -> None:
        self.context = context
        self.robot_id = context.robot_id
        self.settings = dict(context.spec.settings or {})

        self._linear_speed = float(self.settings.get("linear_speed", 0.2))
        self._angular_speed = float(self.settings.get("angular_speed", 0.45))
        self._control_hz = float(self.settings.get("control_hz", 2.0))
        self._render_width = int(self.settings.get("render_width", 640))
        self._render_height = int(self.settings.get("render_height", 480))
        self._camera_names = self._resolve_camera_names()
        self._default_camera = (
            context.embodiment.default_camera
            if context.embodiment and context.embodiment.default_camera
            else "front"
        )
        if self._default_camera not in self._camera_names:
            self._camera_names.insert(0, self._default_camera)

        self.adapter = XLeRobotSimSkillAdapter(
            linear_speed=self._linear_speed,
            angular_speed=self._angular_speed,
            embodiment=context.embodiment,
        )
        self.contracts = SkillContractRuntime(context.skill_catalog)

        self.model: Any = None
        self.data: Any = None
        self.renderer: Any = None
        self._scene_camera: Any = None
        self._scene_cameras: dict[str, Any] = {}
        self._viewer: Any = None
        viewer_cfg = self.settings.get("viewer", {}) or {}
        self._viewer_enabled = bool(viewer_cfg.get("enabled", False))

        self.state = "created"
        self._emergency_stop_active = False
        self.frame_id = 0
        self.last_error: str | None = None
        self.last_skill_result: RobotSkillResult | None = None
        self.last_camera: dict[str, Any] = {"frame_available": False, "frame_id": None}
        self.last_cameras_status: dict[str, dict[str, Any]] = {}
        self.last_arm_status: dict[str, Any] = {}
        self.last_battery: dict[str, Any] = {
            "status": "normal",
            "voltage": 12.0,
            "percentage": 85.0,
        }
        self.startup_diagnostics: dict[str, Any] = {}

        self._last_rendered_frame: np.ndarray | None = None

    # RobotDriver protocol

    async def start(self) -> None:
        import mujoco
        import mujoco.viewer

        mjcf_path = str(_resolve_mjcf_path(self.settings))
        logger.info(f"{self.robot_id} loading MuJoCo model from {mjcf_path}")
        self.model = await asyncio.to_thread(mujoco.MjModel.from_xml_path, mjcf_path)
        self.data = await asyncio.to_thread(mujoco.MjData, self.model)

        rest = self.adapter.arm_rest_positions()
        for idx, pos in rest.items():
            self.data.ctrl[idx] = pos
            self._set_actuator_joint_position(idx, pos)
        self._hold_head_camera()

        await asyncio.to_thread(mujoco.mj_forward, self.model, self.data)

        # Renderer must be created on the calling thread (owns the GL context).
        self.renderer = mujoco.Renderer(
            self.model, self._render_height, self._render_width
        )

        self._scene_cameras = {
            name: self._build_scene_camera(name) for name in self._camera_names
        }
        self._scene_camera = self._scene_cameras.get(self._default_camera)

        self._update_arm_status()
        self.startup_diagnostics = self._build_diagnostics()
        self.state = "idle"

        if self._viewer_enabled:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            logger.info(f"{self.robot_id} MuJoCo viewer opened")
        self.last_error = None
        self.frame_id = 0
        self._last_rendered_frame = None
        logger.info(f"{self.robot_id} MuJoCo sim ready state={self.state}")

    async def capabilities(self) -> RobotCapabilities:
        return RobotCapabilities(
            robot_id=self.robot_id,
            driver_type="xlerobot_sim",
            action_dimensions=None,
            control_hz=self._control_hz,
            cameras=list(self._camera_names),
            observation_modalities=["image", "arm_state", "status"],
            supports_reset=True,
            supports_interrupt=False,
            metadata={
                "body": "xlerobot",
                "robot_family": self.context.spec.robot_family,
                "environment": self.context.spec.robot_environment,
                "driver_kind": self.context.spec.driver_kind,
                "embodiment_profile": (
                    self.context.embodiment.name if self.context.embodiment else None
                ),
                "control": "skill_action",
                "runtime": "mujoco_simulation",
                "default_camera": self._default_camera,
                "cameras": list(self._camera_names),
                "safety": dict(self.settings.get("safety", {}) or {}),
            },
        )

    async def health(self) -> RobotHealth:
        return RobotHealth(
            robot_id=self.robot_id,
            online=self.state != "closed",
            state=self.state,
            frame_id=self.frame_id,
            error=self.last_error,
            metrics={
                "driver": "xlerobot_sim",
                "runtime": "mujoco_simulation",
                "startup_diagnostics": self._build_diagnostics(),
                "last_skill_result": self.last_skill_result.to_dict()
                if self.last_skill_result
                else None,
                "battery": self.last_battery,
                "readiness": self.readiness(),
            },
        )

    async def observe(self) -> DriverObservation:
        self.frame_id += 1
        self._update_arm_status()

        frames = self._render_frames()
        image = frames.get(self._default_camera)
        self._last_rendered_frame = image
        self.last_cameras_status = {
            name: {
                "ok": frame is not None,
                "owner": "simulation",
                "frame_available": frame is not None,
                "frame_id": self.frame_id,
                "image_shape": list(frame.shape) if frame is not None else None,
            }
            for name, frame in frames.items()
        }
        self.last_camera = dict(self.last_cameras_status.get(self._default_camera, {}))

        assets: list[ObservationAsset] = []
        for name, frame in frames.items():
            if frame is None:
                continue
            assets.append(
                ObservationAsset(
                    kind="image",
                    role="camera",
                    name=name,
                    data=frame,
                    metadata={"driver": "xlerobot_sim", "camera_role": name},
                )
            )

        return DriverObservation(
            envelope=self._envelope(),
            frame_id=self.frame_id,
            assets=assets,
            proprioception=self._proprioception(),
            metadata={
                "driver": "xlerobot_sim",
                "body": "xlerobot",
                "robot_family": self.context.spec.robot_family,
                "environment": self.context.spec.robot_environment,
                "embodiment_profile": (
                    self.context.embodiment.name if self.context.embodiment else None
                ),
                "state": self.state,
                "camera": self.last_camera,
                "cameras": self.last_cameras_status,
                "arm_status": self.last_arm_status,
                "battery": self.last_battery,
                "base_pose": self._base_pose(),
                "startup_diagnostics": self._build_diagnostics(),
                "last_skill_result": self.last_skill_result.to_dict()
                if self.last_skill_result
                else None,
                "readiness": self.readiness(),
            },
        )

    async def status(self) -> RobotStatus:
        return RobotStatus(
            envelope=self._envelope(),
            frame_id=self.frame_id,
            state=self.state,
            success=None,
            error=self.last_error,
            metrics={
                "driver": "xlerobot_sim",
                "runtime": "mujoco_simulation",
                "startup_diagnostics": self._build_diagnostics(),
                "camera": self.last_camera,
                "cameras": self.last_cameras_status,
                "arm_status": self.last_arm_status,
                "battery": self.last_battery,
                "last_skill_result": self.last_skill_result.to_dict()
                if self.last_skill_result
                else None,
                "readiness": self.readiness(),
            },
        )

    async def apply_action(self, action: RobotAction) -> RobotStatus:
        try:
            skill = RobotSkillAction.from_robot_action(action)
        except ValueError as exc:
            result = RobotSkillResult(
                False, str(exc), {"failure_mode": "invalid_action"}
            )
            self.last_skill_result = result
            self.state = "failed"
            self.last_error = result.message
            return self._status_for_action(action, success=False)

        _, decision = self.contracts.validate_action(
            skill,
            robot_type="xlerobot",
            status=await self.status(),
            readiness=self.readiness(),
        )
        if not decision.allowed:
            result = RobotSkillResult(
                False,
                decision.reason,
                {
                    "skill": skill.to_dict(),
                    "failure_mode": decision.failure_mode,
                    "contract_decision": decision.metadata,
                },
            )
            self.last_skill_result = result
            self.state = "failed"
            self.last_error = result.message
            return self._status_for_action(action, success=False)

        try:
            cmd = self.adapter.decode(skill)
        except ValueError as exc:
            result = RobotSkillResult(
                False,
                str(exc),
                {"skill": skill.to_dict(), "failure_mode": "unknown_skill"},
            )
            self.last_skill_result = result
            self.state = "failed"
            self.last_error = result.message
            return self._status_for_action(action, success=False)

        is_gripper_command = cmd.jaw_left is not None or cmd.jaw_right is not None
        if is_gripper_command:
            logger.info(
                f"{self.robot_id} gripper_debug phase=decoded "
                f"skill={cmd.skill_name} args={skill.arguments} "
                f"target_left={cmd.jaw_left} target_right={cmd.jaw_right} "
                f"{self._gripper_debug_state()}"
            )

        if cmd.arm_targets:
            self._stop_base_motion()
            if cmd.delta_mode:
                for idx, delta in cmd.arm_targets.items():
                    self.data.ctrl[idx] = float(self.data.ctrl[idx]) + delta
            else:
                for idx, target in cmd.arm_targets.items():
                    self.data.ctrl[idx] = target

        if cmd.jaw_left is not None:
            self._stop_base_motion()
            gripper_indices = self.adapter.gripper_actuator_indices()
            if gripper_indices is not None:
                self.data.ctrl[gripper_indices[0]] = cmd.jaw_left
                logger.info(
                    f"{self.robot_id} gripper_debug phase=write_left "
                    f"actuator={gripper_indices[0]} target={cmd.jaw_left} "
                    f"{self._gripper_debug_state()}"
                )
            else:
                logger.warning(
                    f"{self.robot_id} gripper_debug phase=write_left "
                    "missing_gripper_indices"
                )
        if cmd.jaw_right is not None:
            self._stop_base_motion()
            gripper_indices = self.adapter.gripper_actuator_indices()
            if gripper_indices is not None:
                self.data.ctrl[gripper_indices[1]] = cmd.jaw_right
                logger.info(
                    f"{self.robot_id} gripper_debug phase=write_right "
                    f"actuator={gripper_indices[1]} target={cmd.jaw_right} "
                    f"{self._gripper_debug_state()}"
                )
            else:
                logger.warning(
                    f"{self.robot_id} gripper_debug phase=write_right "
                    "missing_gripper_indices"
                )

        if cmd.duration_sec > 0:
            self._emergency_stop_active = False
            timestep = self.model.opt.timestep
            steps = max(1, int(cmd.duration_sec / timestep))
            await asyncio.to_thread(self._step_velocity, steps, cmd.vx, cmd.vy, cmd.vw)
            self._stop_base_motion()
        elif cmd.skill_name == "stop_motion":
            self._stop_base_motion()
            self._emergency_stop_active = bool(skill.arguments.get("emergency", False))
        elif cmd.arm_targets or cmd.jaw_left is not None or cmd.jaw_right is not None:
            self._emergency_stop_active = False
            base_qpos = self.data.qpos[:3].copy()
            # Pure gripper commands need longer settle time to show visible jaw motion,
            # while arm commands remain short to avoid destabilizing the current model.
            settle_sec = (
                0.35
                if not cmd.arm_targets
                and (cmd.jaw_left is not None or cmd.jaw_right is not None)
                else 0.1
            )
            settle_steps = max(1, int(settle_sec / self.model.opt.timestep))
            if is_gripper_command:
                logger.info(
                    f"{self.robot_id} gripper_debug phase=before_settle "
                    f"settle_sec={settle_sec:.3f} settle_steps={settle_steps} "
                    f"{self._gripper_debug_state()}"
                )
            hold_ctrl = {
                idx: float(self.data.ctrl[idx])
                for idx in self._arm_hold_actuator_indices(cmd.arm_targets)
                if 0 <= idx < len(self.data.ctrl)
            }
            lock_qpos = (
                self._non_gripper_arm_joint_positions()
                if is_gripper_command and not cmd.arm_targets
                else None
            )
            drive_qpos = (
                self._commanded_gripper_joint_positions()
                if is_gripper_command and not cmd.arm_targets
                else None
            )
            await asyncio.to_thread(
                self._step_n,
                settle_steps,
                hold_ctrl=hold_ctrl,
                lock_qpos=lock_qpos,
                drive_qpos=drive_qpos,
            )
            if is_gripper_command:
                logger.info(
                    f"{self.robot_id} gripper_debug phase=after_settle_before_base_restore "
                    f"{self._gripper_debug_state()}"
                )
            self.data.qpos[:3] = base_qpos
            import mujoco

            mujoco.mj_forward(self.model, self.data)
            self._stop_base_motion()
            if is_gripper_command:
                logger.info(
                    f"{self.robot_id} gripper_debug phase=after_base_restore "
                    f"{self._gripper_debug_state()}"
                )

        result = RobotSkillResult(True, cmd.message, {"skill": skill.to_dict()})
        self.last_skill_result = result
        self.state = "skill_completed"
        self.last_error = None
        self._update_arm_status()
        if is_gripper_command:
            logger.info(
                f"{self.robot_id} gripper_debug phase=final_status "
                f"gripper_opening_pct={self.last_arm_status.get('gripper_opening_pct')} "
                f"{self._gripper_debug_state()}"
            )
        status = self._status_for_action(action, success=True)
        self.state = "idle"
        return status

    async def reset(self) -> RobotStatus:
        import mujoco

        await asyncio.to_thread(mujoco.mj_resetData, self.model, self.data)
        rest = self.adapter.arm_rest_positions()
        for idx, pos in rest.items():
            self.data.ctrl[idx] = pos
            self._set_actuator_joint_position(idx, pos)
        self._hold_head_camera()
        self._stop_base_motion()
        await asyncio.to_thread(mujoco.mj_forward, self.model, self.data)
        self.state = "idle"
        self.last_error = None
        self.last_skill_result = RobotSkillResult(True, "sim reset", {"skill": "reset"})
        self.frame_id = 0
        self._last_rendered_frame = None
        self._update_arm_status()
        return await self.status()

    async def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
        self._viewer = None
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = None
        self._scene_camera = None
        self._scene_cameras = {}
        self.data = None
        self.model = None
        self.state = "closed"

    # Simulation helpers

    # ---- public VLA API ----

    def render_camera_frames(self, camera_names: list[str]) -> dict[str, np.ndarray]:
        """Render frames from named cameras for VLA observation."""
        previous = self._scene_camera
        frames: dict[str, np.ndarray | None] = {}
        try:
            for name in camera_names:
                camera = self._scene_cameras.get(name)
                if camera is None:
                    camera = self._build_scene_camera(name)
                    self._scene_cameras[name] = camera
                self._scene_camera = camera
                frames[name] = self._render_frame()
            return {n: f for n, f in frames.items() if f is not None}
        finally:
            self._scene_camera = previous

    def read_arm_state(self, arm: str) -> dict[str, float]:
        """Read arm joint positions for a specific side as a name->rad dict."""
        indices = self.adapter.arm_actuator_indices(arm)
        joint_order = self.adapter.arm_joint_order()
        result: dict[str, float] = {}
        for i, name in enumerate(joint_order):
            if i < len(indices) and indices[i] < len(self.data.ctrl):
                result[name] = float(self.data.ctrl[indices[i]])
            else:
                result[name] = 0.0
        return result

    def read_arm_state_vector(self, arm: str) -> np.ndarray:
        """Read arm joint positions as a 6D ndarray in canonical joint order."""
        state = self.read_arm_state(arm)
        joint_order = self.adapter.arm_joint_order()
        return np.array(
            [state.get(name, 0.0) for name in joint_order], dtype=np.float64
        )

    def write_arm_targets(self, arm: str, targets_rad: np.ndarray) -> None:
        """Write actuator targets for one arm side with ctrlrange clamping."""
        if self.model is None or self.data is None:
            return
        indices = self.adapter.arm_actuator_indices(arm)
        self._stop_base_motion()
        for i, idx in enumerate(indices):
            if i >= len(targets_rad):
                break
            lo = float(self.model.actuator_ctrlrange[idx][0])
            hi = float(self.model.actuator_ctrlrange[idx][1])
            self.data.ctrl[idx] = float(np.clip(targets_rad[i], lo, hi))
        self._update_arm_status()

    def step_control(self, dt: float) -> None:
        """Advance MuJoCo by dt seconds, keeping base still."""
        import mujoco

        if self.model is None or self.data is None:
            return
        timestep = self.model.opt.timestep
        steps = max(1, int(dt / timestep))
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            self._stop_base_motion()
            self._sync_viewer()

    def vla_readiness(self) -> dict[str, Any]:
        """Return VLA-specific readiness for capability gating."""
        base = self.readiness()
        base["vla"] = {
            "ok": self.state not in {"closed", "failed"},
            "sim_driver": self.state,
            "cameras_available": sorted(self._camera_names),
        }
        return base

    def create_vla_io_adapter(self, **settings: Any) -> VLAIOAdapter:
        """Return a :class:`SimVLAIOAdapter` wired to this simulation driver."""
        from hey_robot.robots.simulation.sim_vla_io_adapter import SimVLAIOAdapter

        return SimVLAIOAdapter(
            self,
            arm=str(settings.get("arm", "right")),
        )

    # ---- internal simulation helpers ----

    def _render_frame(self) -> np.ndarray | None:
        if self.renderer is None or self._scene_camera is None:
            return None
        try:
            self.renderer.update_scene(self.data, camera=self._scene_camera)
            pixels = self.renderer.render()
            return np.array(pixels, dtype=np.uint8)
        except Exception:
            return None

    def _render_frames(self) -> dict[str, np.ndarray | None]:
        previous = self._scene_camera
        frames: dict[str, np.ndarray | None] = {}
        for name, camera in self._scene_cameras.items():
            self._scene_camera = camera
            frames[name] = self._render_frame()
        self._scene_camera = previous
        return frames

    def _step_velocity(self, steps: int, vx: float, vy: float, vw: float) -> None:
        import mujoco

        # Official XLeRobot exposes world-frame root joints. Convert the public
        # body-frame command to root X/Y using the official yaw joint only.
        phi = float(self.data.qpos[2])
        qvel_x = math.cos(phi) * vy - math.sin(phi) * vx
        qvel_y = math.sin(phi) * vy + math.cos(phi) * vx

        for _ in range(steps):
            mujoco.mj_step1(self.model, self.data)
            self.data.qvel[0] = qvel_x
            self.data.qvel[1] = qvel_y
            self.data.qvel[2] = vw
            self.data.qacc[0] = 0.0
            self.data.qacc[1] = 0.0
            self.data.qacc[2] = 0.0
            mujoco.mj_step2(self.model, self.data)
            self._sync_viewer()
        self._stop_base_motion()

    def _stop_base_motion(self) -> None:
        if self.data is None:
            return
        # Zero both actuator targets and simulated base state so later arm/gripper
        # settle steps do not integrate residual chassis motion.
        self.data.ctrl[15] = 0.0
        self.data.ctrl[16] = 0.0
        self.data.ctrl[17] = 0.0
        self.data.qvel[0] = 0.0
        self.data.qvel[1] = 0.0
        self.data.qvel[2] = 0.0
        self.data.qacc[0] = 0.0
        self.data.qacc[1] = 0.0
        self.data.qacc[2] = 0.0

    def _hold_head_camera(self) -> None:
        import mujoco

        if self.model is None or self.data is None:
            return
        for actuator_name in ("head_pan_hold", "head_tilt_hold"):
            actuator_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            if actuator_id >= 0:
                self.data.ctrl[actuator_id] = 0.0

    def _sync_viewer(self) -> None:
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def _step_n(
        self,
        n: int,
        hold_ctrl: dict[int, float] | None = None,
        lock_qpos: dict[int, float] | None = None,
        drive_qpos: dict[int, float] | None = None,
    ) -> None:
        import mujoco

        for _ in range(n):
            if hold_ctrl:
                for idx, value in hold_ctrl.items():
                    self.data.ctrl[idx] = value
            if lock_qpos:
                for qpos_addr, value in lock_qpos.items():
                    self.data.qpos[qpos_addr] = value
                    if qpos_addr < len(self.data.qvel):
                        self.data.qvel[qpos_addr] = 0.0
            if drive_qpos:
                for qpos_addr, value in drive_qpos.items():
                    self.data.qpos[qpos_addr] = value
                    if qpos_addr < len(self.data.qvel):
                        self.data.qvel[qpos_addr] = 0.0
            mujoco.mj_step(self.model, self.data)
            self._sync_viewer()
            if hold_ctrl:
                for idx, value in hold_ctrl.items():
                    self.data.ctrl[idx] = value
            if lock_qpos:
                for qpos_addr, value in lock_qpos.items():
                    self.data.qpos[qpos_addr] = value
                    if qpos_addr < len(self.data.qvel):
                        self.data.qvel[qpos_addr] = 0.0
            if drive_qpos:
                for qpos_addr, value in drive_qpos.items():
                    self.data.qpos[qpos_addr] = value
                    if qpos_addr < len(self.data.qvel):
                        self.data.qvel[qpos_addr] = 0.0

    def _arm_hold_actuator_indices(self, command_targets: dict[int, float]) -> set[int]:
        indices: set[int] = set(command_targets)
        with contextlib.suppress(Exception):
            indices.update(self.adapter.arm_actuator_indices("left"))
            indices.update(self.adapter.arm_actuator_indices("right"))
        gripper_indices = self.adapter.gripper_actuator_indices()
        if gripper_indices is not None:
            indices.update(gripper_indices)
        return indices

    def _set_actuator_joint_position(self, actuator_idx: int, value: float) -> None:
        if self.model is None or self.data is None:
            return
        qpos_addr = self._actuator_joint_qpos_addr(actuator_idx)
        if qpos_addr is None:
            return
        self.data.qpos[qpos_addr] = float(value)

    def _non_gripper_arm_joint_positions(self) -> dict[int, float]:
        if self.model is None or self.data is None:
            return {}
        gripper_indices = set(self.adapter.gripper_actuator_indices() or ())
        actuator_indices: set[int] = set()
        with contextlib.suppress(Exception):
            actuator_indices.update(self.adapter.arm_actuator_indices("left"))
            actuator_indices.update(self.adapter.arm_actuator_indices("right"))
        locked: dict[int, float] = {}
        for idx in actuator_indices - gripper_indices:
            qpos_addr = self._actuator_joint_qpos_addr(idx)
            if qpos_addr is not None:
                locked[qpos_addr] = float(self.data.qpos[qpos_addr])
        return locked

    def _commanded_gripper_joint_positions(self) -> dict[int, float]:
        if self.model is None or self.data is None:
            return {}
        positions: dict[int, float] = {}
        for idx in self.adapter.gripper_actuator_indices() or ():
            qpos_addr = self._actuator_joint_qpos_addr(idx)
            if qpos_addr is not None:
                positions[qpos_addr] = self._clamped_actuator_ctrl(idx)
        return positions

    def _clamped_actuator_ctrl(self, actuator_idx: int) -> float:
        value = float(self.data.ctrl[actuator_idx])
        joint_id = int(self.model.actuator_trnid[actuator_idx][0])
        if 0 <= joint_id < self.model.njnt:
            lo, hi = self.model.jnt_range[joint_id]
            if hi > lo:
                return float(np.clip(value, float(lo), float(hi)))
        return value

    def _actuator_joint_qpos_addr(self, actuator_idx: int) -> int | None:
        if self.model is None:
            return None
        if actuator_idx < 0 or actuator_idx >= self.model.nu:
            return None
        joint_id = int(self.model.actuator_trnid[actuator_idx][0])
        if joint_id < 0 or joint_id >= self.model.njnt:
            return None
        qpos_addr = int(self.model.jnt_qposadr[joint_id])
        if qpos_addr < 0 or qpos_addr >= self.model.nq:
            return None
        return qpos_addr

    def _update_arm_status(self) -> None:
        if self.data is None:
            self.last_arm_status = {}
            return
        joint_states: dict[str, float] = {}
        for name in _ARM_JOINT_NAMES:
            indices = self.adapter.joint_to_actuators(name)
            if indices is None:
                continue
            left_idx = indices[0]
            if left_idx < len(self.data.ctrl):
                joint_states[name] = float(self.data.ctrl[left_idx])
            else:
                joint_states[name] = 0.0
        gripper_indices = self.adapter.gripper_actuator_indices()
        jaw_l = (
            self._joint_position_for_actuator(gripper_indices[0])
            if gripper_indices is not None and gripper_indices[0] < len(self.data.ctrl)
            else 0.0
        )
        gripper_open_value = self.adapter.gripper_open_value
        gripper_pct = (
            jaw_l / gripper_open_value * 100.0 if gripper_open_value > 0 else 0.0
        )
        gripper_pct = max(0.0, min(100.0, gripper_pct))
        self.last_arm_status = {
            "success": True,
            "enabled": True,
            "initialized": True,
            "message": "sim arm ready",
            "joint_states": joint_states,
            "joint_count": 6,
            "gripper_opening_pct": gripper_pct,
        }

    def _proprioception(self) -> list[float]:
        if self.data is None:
            return []
        qpos = self.data.qpos
        qvel = self.data.qvel
        values: list[float] = [
            float(qpos[0]) if self.model.nq > 0 else 0.0,
            float(qpos[1]) if self.model.nq > 1 else 0.0,
            float(qpos[2]) if self.model.nq > 2 else 0.0,
            float(qvel[0]) if self.model.nv > 0 else 0.0,
            float(qvel[1]) if self.model.nv > 1 else 0.0,
            float(qvel[2]) if self.model.nv > 2 else 0.0,
        ]
        for name in _ARM_JOINT_NAMES:
            indices = self.adapter.joint_to_actuators(name)
            if indices is not None:
                left_idx = indices[0]
                values.append(
                    float(self.data.ctrl[left_idx])
                    if left_idx < len(self.data.ctrl)
                    else 0.0
                )
        return values

    def _joint_position_for_actuator(self, actuator_idx: int) -> float:
        if self.model is None or self.data is None:
            return 0.0
        if actuator_idx < 0 or actuator_idx >= self.model.nu:
            return 0.0
        joint_id = int(self.model.actuator_trnid[actuator_idx][0])
        if joint_id < 0 or joint_id >= self.model.njnt:
            return float(self.data.ctrl[actuator_idx])
        qpos_addr = int(self.model.jnt_qposadr[joint_id])
        if qpos_addr < 0 or qpos_addr >= self.model.nq:
            return float(self.data.ctrl[actuator_idx])
        return float(self.data.qpos[qpos_addr])

    def _gripper_debug_state(self) -> str:
        if self.model is None or self.data is None:
            return "model_ready=False"
        indices = self.adapter.gripper_actuator_indices()
        if indices is None:
            return "gripper_indices=None"
        parts = [f"gripper_indices={indices}"]
        for side, actuator_idx in (("left", indices[0]), ("right", indices[1])):
            ctrl = (
                float(self.data.ctrl[actuator_idx])
                if 0 <= actuator_idx < len(self.data.ctrl)
                else None
            )
            joint_id = (
                int(self.model.actuator_trnid[actuator_idx][0])
                if 0 <= actuator_idx < self.model.nu
                else -1
            )
            joint_name = self._mujoco_name("joint", joint_id)
            qpos_addr = (
                int(self.model.jnt_qposadr[joint_id])
                if 0 <= joint_id < self.model.njnt
                else -1
            )
            qpos = (
                float(self.data.qpos[qpos_addr])
                if 0 <= qpos_addr < self.model.nq
                else None
            )
            joint_range = (
                tuple(float(v) for v in self.model.jnt_range[joint_id])
                if 0 <= joint_id < self.model.njnt
                else None
            )
            parts.append(
                f"{side}={{actuator:{actuator_idx},joint:{joint_name},"
                f"qpos_addr:{qpos_addr},ctrl:{ctrl},qpos:{qpos},range:{joint_range}}}"
            )
        return " ".join(parts)

    def _mujoco_name(self, obj_type: str, obj_id: int) -> str | None:
        if self.model is None or obj_id < 0:
            return None
        import mujoco

        obj = {
            "joint": mujoco.mjtObj.mjOBJ_JOINT,
            "actuator": mujoco.mjtObj.mjOBJ_ACTUATOR,
        }[obj_type]
        name = mujoco.mj_id2name(self.model, obj, obj_id)
        return name if isinstance(name, str) else None

    def _base_pose(self) -> dict[str, float]:
        if self.data is None:
            return {"x_cm": 0.0, "y_cm": 0.0, "yaw_deg": 0.0}
        import mujoco

        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, _ROBOT_BODY)
        xpos = self.data.xpos[body_id]
        xmat = self.data.xmat[body_id]
        # Extract yaw from rotation matrix: atan2(xmat[3], xmat[0])
        # xmat is 3x3 stored flat: [xx, xy, xz, yx, yy, yz, zx, zy, zz]
        yaw = math.atan2(float(xmat[3]), float(xmat[0]))
        return {
            "x_cm": float(xpos[0]) * 100.0,
            "y_cm": float(xpos[1]) * 100.0,
            "yaw_deg": math.degrees(yaw),
        }

    def readiness(self) -> dict[str, Any]:
        readiness: dict[str, Any] = {
            "robot": self.state != "closed",
            "battery": self.last_battery,
            "emergency_stop": self._emergency_stop_active,
        }
        resources = (
            self.context.embodiment.readiness_resources
            if self.context.embodiment and self.context.embodiment.readiness_resources
            else ("base", "arm", "gripper", "camera")
        )
        for resource in resources:
            readiness[resource] = {"ok": True}
        for camera_name, status in self.last_cameras_status.items():
            readiness.setdefault(
                f"{camera_name}_camera",
                {"ok": bool(status.get("ok")), "owner": "simulation"},
            )
        return readiness

    def _build_diagnostics(self) -> dict[str, Any]:
        return {
            "bus": {"ok": True, "port": "sim", "baudrate": 0, "message": "sim"},
            "servo_bus": {
                "ok": True,
                "configured_ids": list(range(1, 19)),
                "servos": [],
            },
            "base": {
                "ok": True,
                "response": {"success": True, "message": "sim base ready"},
            },
            "arm": {
                "ok": True,
                "joint_count": 6,
                "response": {"success": True, "message": "sim arm ready"},
                "status_response": self.last_arm_status,
            },
            "camera": {
                "ok": True,
                "frame_available": True,
                "frame_id": self.frame_id,
                "owner": "simulation",
            },
            "cameras": {
                name: {
                    "ok": True,
                    "frame_available": True,
                    "frame_id": self.frame_id,
                    "owner": "simulation",
                }
                for name in self._camera_names
            },
            "battery": self.last_battery,
            "safety": {"emergency_stop": self._emergency_stop_active},
        }

    def _status_for_action(self, action: RobotAction, *, success: bool) -> RobotStatus:
        return RobotStatus(
            envelope=self._envelope(),
            frame_id=self.frame_id,
            state=self.state,
            skill_id=action.skill_id,
            success=success,
            error=None if success else self.last_error,
            metrics={
                "driver": "xlerobot_sim",
                "runtime": "mujoco_simulation",
                "startup_diagnostics": self._build_diagnostics(),
                "camera": self.last_camera,
                "cameras": self.last_cameras_status,
                "arm_status": self.last_arm_status,
                "battery": self.last_battery,
                "last_skill_result": self.last_skill_result.to_dict()
                if self.last_skill_result
                else None,
                "readiness": self.readiness(),
            },
        )

    def _envelope(self) -> Envelope:
        return Envelope(
            robot_id=self.robot_id,
            deployment_id=self.context.deployment_id,
            trace_id=f"xlerobot_sim_{self.robot_id}_{int(time.time() * 1000)}",
        )

    def _resolve_camera_names(self) -> list[str]:
        if self.context.embodiment is not None:
            raw = self.context.embodiment.camera_layout.get("cameras")
            if isinstance(raw, (list, tuple)):
                names = [str(item) for item in raw if str(item).strip()]
                if names:
                    return names
        return ["front"]

    def _build_scene_camera(self, name: str):
        import mujoco

        layout = dict(_DEFAULT_SIM_CAMERA_LAYOUT.get(name, {}))
        body_name = str(layout.get("body") or _ROBOT_BODY)
        camera_name = str(layout.get("camera_name") or name)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        prefer_native = bool(layout.get("prefer_native", True))
        if prefer_native:
            camera_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
            )
            if camera_id >= 0:
                camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
                camera.fixedcamid = camera_id
                return camera
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        camera.distance = float(layout.get("distance", 0.8))
        camera.azimuth = float(layout.get("azimuth", 180.0))
        camera.elevation = float(layout.get("elevation", -20.0))
        camera.lookat = np.array(layout.get("lookat", [0.0, 0.0, 0.0]), dtype=float)
        return camera
