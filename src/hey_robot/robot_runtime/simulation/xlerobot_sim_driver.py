from __future__ import annotations

import asyncio
import contextlib
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from hey_robot.contracts import SkillContractRuntime
from hey_robot.logging import HeyRobotLogger
from hey_robot.protocol import (
    Envelope,
    RobotAction,
    RobotSkillAction,
    RobotSkillResult,
    RobotStatus,
)
from hey_robot.robot_runtime.base import (
    RobotCapabilities,
    RobotDriverContext,
    RobotHealth,
)
from hey_robot.robot_runtime.observations import DriverObservation, ObservationAsset
from hey_robot.robot_runtime.simulation.mujoco_logging import (
    configure_mujoco_warning_logging,
)
from hey_robot.robot_runtime.simulation.skill_adapter import XLeRobotSimSkillAdapter

logger = HeyRobotLogger(name="xlerobot_sim")
_ROBOT_BODY = "base_link"
_DEFAULT_HEAD_PAN = 0.0
_DEFAULT_HEAD_TILT = 0.25
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

# SDK 关节名（arm_status / arm_joints 使用的标准顺序）。
_ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


_EGL_CONTEXT: Any = None
"""Singleton EGL GL context for headless rendering."""


class _DockSessionAdapter:
    """通过已验证的 dock-kernel session API 暴露 XLeRobotSimDriver。"""

    def __init__(self, driver: XLeRobotSimDriver) -> None:
        self.driver = driver
        self.model = driver.model
        self.data = driver.data
        self._base_qpos: dict[int, float] = {}
        self._base_dofs: list[int] = []
        self._right_arm_qpos: dict[int, float] = {}
        self._right_arm_dofs: list[int] = []
        self.lock_base()

    @property
    def connected(self) -> bool:
        return self.model is not None and self.data is not None

    @property
    def viewer(self) -> Any:
        return self.driver._viewer

    def require_connected(self) -> None:
        if not self.connected:
            raise RuntimeError("XLeRobot simulation is not connected")

    def id_for(self, object_type: Any, name: str) -> int:
        import mujoco

        object_id = int(mujoco.mj_name2id(self.model, object_type, name))
        if object_id < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return object_id

    def lock_base(self) -> None:
        """为左臂操作快照底盘和非活动右臂状态。"""
        import mujoco

        self._base_qpos.clear()
        self._base_dofs.clear()
        for name in (
            "root_x_axis_joint",
            "root_y_axis_joint",
            "root_z_rotation_joint",
        ):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_addr = int(self.model.jnt_qposadr[joint_id])
            dof_addr = int(self.model.jnt_dofadr[joint_id])
            self._base_qpos[qpos_addr] = float(self.data.qpos[qpos_addr])
            self._base_dofs.append(dof_addr)

        self._right_arm_qpos.clear()
        self._right_arm_dofs.clear()
        for actuator_idx in self.driver.adapter.arm_actuator_indices("right"):
            joint_id = int(self.model.actuator_trnid[actuator_idx][0])
            qpos_addr = int(self.model.jnt_qposadr[joint_id])
            dof_addr = int(self.model.jnt_dofadr[joint_id])
            self._right_arm_qpos[qpos_addr] = float(self.data.qpos[qpos_addr])
            self._right_arm_dofs.append(dof_addr)

    def clamp_base(self) -> None:
        for qpos_addr, value in self._base_qpos.items():
            self.data.qpos[qpos_addr] = value
        for dof_addr in self._base_dofs:
            self.data.qvel[dof_addr] = 0.0
        for qpos_addr, value in self._right_arm_qpos.items():
            self.data.qpos[qpos_addr] = value
        for dof_addr in self._right_arm_dofs:
            self.data.qvel[dof_addr] = 0.0


def _configure_mujoco_gl_backend() -> str | None:
    configured = os.environ.get("MUJOCO_GL")
    if configured:
        return configured
    system = platform.system()
    if system == "Windows":
        os.environ["MUJOCO_GL"] = "wgl"
        return "wgl"
    if system == "Linux" and not os.environ.get("DISPLAY"):
        os.environ["MUJOCO_GL"] = "egl"
        return "egl"
    return None


def _needs_egl_context() -> bool:
    return platform.system() == "Linux" and os.environ.get("MUJOCO_GL") == "egl"


def _ensure_egl_context(width: int = 640, height: int = 480) -> None:
    global _EGL_CONTEXT
    if _EGL_CONTEXT is not None:
        return
    import mujoco.egl

    _EGL_CONTEXT = mujoco.egl.GLContext(width, height)
    _EGL_CONTEXT.make_current()


def _resolve_mjcf_path(settings: dict[str, Any]) -> Path:
    raw = settings.get("mjcf_path") or settings.get("mjcf")
    if raw:
        p = Path(raw)
        if p.is_absolute():
            return p
        return Path.cwd() / p
    return Path.cwd() / "assets" / "scenes" / "home_scene.xml"


class XLeRobotSimDriver:
    """实现 RobotDriver 协议的 MuJoCo 仿真驱动。"""

    def __init__(self, context: RobotDriverContext) -> None:
        self.context = context
        self.robot_id = context.robot_id
        self.settings = dict(context.spec.settings or {})
        # 在测试或调用方绕过 ``start``、直接通过该驱动构造 MjModel/MjData 前完成配置。
        with contextlib.suppress(ImportError):
            configure_mujoco_warning_logging(context.deployment_id)

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
        self._dock_session: _DockSessionAdapter | None = None
        self._dock_arm: Any = None
        self._dock_gripper: Any = None
        self._dock_arm_side = "left"
        self._dock_manipulation_active = False

    # RobotDriver 协议

    async def start(self) -> None:
        _configure_mujoco_gl_backend()
        import mujoco
        import mujoco.viewer

        configure_mujoco_warning_logging(
            self.context.deployment_id, mujoco_module=mujoco
        )
        mjcf_path = str(_resolve_mjcf_path(self.settings))
        logger.info(f"{self.robot_id} loading MuJoCo model from {mjcf_path}")
        # 在 headless EGL 部署中，MuJoCo 模型加载不能安全地放到任意 executor 线程。
        # 场景加载时间很短，并且必须和数据、renderer 初始化共享驱动持有 GL 的线程。
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        logger.info(f"{self.robot_id} MuJoCo model loaded; creating simulation data")
        self.data = mujoco.MjData(self.model)

        rest = self.adapter.arm_rest_positions()
        num_ctrl = len(self.data.ctrl)
        for idx, pos in rest.items():
            if idx < num_ctrl:
                self.data.ctrl[idx] = pos
                self._set_actuator_joint_position(idx, pos)
        self._hold_head_camera()

        mujoco.mj_forward(self.model, self.data)
        self._initialize_dock_manipulation()
        logger.info(f"{self.robot_id} simulation data initialized")

        # Linux headless 渲染在创建 Renderer 前需要 EGL context。
        # Windows 使用 WGL；如果缺少 EGL.dll，导入 mujoco.egl 会失败。
        if _needs_egl_context():
            logger.info(f"{self.robot_id} initializing EGL context")
            _ensure_egl_context(self._render_width, self._render_height)

        # Renderer 必须在调用线程创建，因为该线程持有 GL context。
        logger.info(f"{self.robot_id} creating MuJoCo renderer")
        self.renderer = mujoco.Renderer(
            self.model, self._render_height, self._render_width
        )
        logger.info(f"{self.robot_id} MuJoCo renderer ready")

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
            state=self._protocol_state(),  # type: ignore[arg-type]
            location_id=self._location_id(),
            motion_state="moving" if self.state == "executing" else "idle",
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

        # MuJoCo model/data 和 passive viewer 都归这个线程所有。
        # 即使操作本身很短，把这些 primitive 放到 asyncio worker pool 中运行也可能
        # 让原生 viewer/context 死锁。
        dock_result = self._execute_dock_primitive(skill)
        if dock_result is not None:
            self.last_skill_result = dock_result
            self.state = "skill_completed" if dock_result.success else "failed"
            self.last_error = None if dock_result.success else dock_result.message
            self._update_arm_status()
            status = self._status_for_action(action, success=dock_result.success)
            self.state = "idle" if dock_result.success else "failed"
            return status

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

        num_ctrl = len(self.data.ctrl)
        if cmd.arm_targets:
            self._stop_base_motion()
            if cmd.delta_mode:
                for idx, delta in cmd.arm_targets.items():
                    if idx < num_ctrl:
                        self.data.ctrl[idx] = float(self.data.ctrl[idx]) + delta
            else:
                for idx, target in cmd.arm_targets.items():
                    if idx < num_ctrl:
                        self.data.ctrl[idx] = target

        if cmd.jaw_left is not None:
            self._stop_base_motion()
            gripper_indices = self.adapter.gripper_actuator_indices()
            if gripper_indices is not None and gripper_indices[0] < num_ctrl:
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
            if gripper_indices is not None and gripper_indices[1] < num_ctrl:
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
            # 纯夹爪命令需要更长 settle 时间才能看到明显夹爪运动；
            # 机械臂命令保持较短，以免当前模型不稳定。
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
        num_ctrl = len(self.data.ctrl)
        for idx, pos in rest.items():
            if idx < num_ctrl:
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
        self._dock_session = None
        self._dock_arm = None
        self._dock_gripper = None
        self.data = None
        self.model = None
        self.state = "closed"

    # 仿真辅助方法

    async def stream_camera_frames(
        self, *, timeout_ms: int = 100
    ) -> dict[str, dict[str, Any]]:
        """从所有场景相机流式输出渲染帧。

        返回结构与 XLeRobotDriver.stream_camera_frames 一致，使 NATS 相机流循环也能用于仿真。
        """
        del timeout_ms
        rendered = self._render_frames()
        return {
            name: {"frame_id": self.frame_id, "image": img}
            for name, img in rendered.items()
            if img is not None
        }

    # ---- 公共 VLA API ----

    def render_camera_frames(self, camera_names: list[str]) -> dict[str, np.ndarray]:
        """从指定相机渲染用于 VLA 观测的图像帧。"""
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
        """读取指定手臂侧的关节位置，返回 name->rad 字典。"""
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
        """按标准关节顺序读取 6D ndarray 形式的手臂关节位置。"""
        state = self.read_arm_state(arm)
        joint_order = self.adapter.arm_joint_order()
        return np.array(
            [state.get(name, 0.0) for name in joint_order], dtype=np.float64
        )

    def write_arm_targets(self, arm: str, targets_rad: np.ndarray) -> None:
        """为指定手臂侧写入 actuator 目标，并按 ctrlrange 裁剪。"""
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
        """让 MuJoCo 前进 dt 秒，同时保持底盘静止。"""
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
        """返回用于能力门控的 VLA 专用就绪状态。"""
        base = self.readiness()
        base["vla"] = {
            "ok": self.state not in {"closed", "failed"},
            "sim_driver": self.state,
            "cameras_available": sorted(self._camera_names),
        }
        return base

    # ---- 内部仿真辅助方法 ----

    def _initialize_dock_manipulation(self) -> None:
        """当活动场景包含 wand 时绑定 dock kernel。"""
        import mujoco

        if (
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wand") < 0
            or mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grip_weld")
            < 0
        ):
            return
        from hey_robot.robot_runtime.simulation.dock_manipulation.arm import (
            So101MobileArmKernel,
        )
        from hey_robot.robot_runtime.simulation.dock_manipulation.gripper import (
            WandGripperKernel,
        )

        self._dock_session = _DockSessionAdapter(self)
        use_physical_left = (
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw_tip_2")
            >= 0
        )
        if use_physical_left:
            self._dock_arm_side = "right"
            self._dock_arm = So101MobileArmKernel(
                self._dock_session,
                joint_names=(
                    "Rotation_2",
                    "Pitch_2",
                    "Elbow_2",
                    "Wrist_Pitch_2",
                    "Wrist_Roll_2",
                ),
                actuator_names=(
                    "Rotation_R",
                    "Pitch_R",
                    "Elbow_R",
                    "Wrist_Pitch_R",
                    "Wrist_Roll_R",
                ),
                ee_body_name="Fixed_Jaw_tip_2",
            )
            jaw_actuator_name = "Jaw_R"
        else:
            self._dock_arm_side = "left"
            self._dock_arm = So101MobileArmKernel(self._dock_session)
            jaw_actuator_name = "Jaw_L"
        self._dock_arm.bind()
        self._dock_gripper = WandGripperKernel(
            self._dock_session,
            self._dock_arm,
            jaw_actuator_name=jaw_actuator_name,
        )
        self._dock_gripper.bind()

    def _execute_dock_primitive(
        self, skill: RobotSkillAction
    ) -> RobotSkillResult | None:
        """在 home 场景中执行 dock 操作 primitive。"""
        if self._dock_session is None:
            return None
        name = skill.name
        if name not in {
            "arm_get_state",
            "arm_solve_position_ik",
            "sim_locate_object",
            "sim_get_object_state",
        }:
            if name == "move_arm_joints":
                joints = skill.arguments.get("joints")
                if not isinstance(joints, dict) or not any(
                    key in joints
                    for key in ("Rotation", "Pitch", "Elbow", "Wrist_Pitch")
                ):
                    return None
            elif name != "set_gripper":
                return None

        self._dock_session.lock_base()
        if name == "arm_get_state":
            return RobotSkillResult(
                True,
                "left arm state read",
                {"joint_positions": self._dock_arm.get_joint_positions()},
            )
        if name == "arm_solve_position_ik":
            self._dock_manipulation_active = True
            target_local = self._xyz(skill.arguments.get("target_xyz"))
            target_world = self._base_to_world(target_local)
            target_axis_local = None
            target_axis_world = None
            if skill.arguments.get("target_axis") is not None:
                target_axis_local = self._xyz(skill.arguments.get("target_axis"))
                target_axis_world = self._base_vector_to_world(target_axis_local)
            seed_value = skill.arguments.get("current_joints")
            seed = (
                [float(value) for value in seed_value]
                if isinstance(seed_value, (list, tuple))
                and len(seed_value) == self._dock_arm.dof
                else None
            )
            solution = self._dock_arm.ik(
                target_world,
                seed,
                target_axis=target_axis_world,
            )
            return RobotSkillResult(
                True,
                "left-arm IK solved" if solution is not None else "IK unreachable",
                {
                    "operation_success": solution is not None,
                    "failure_mode": None if solution is not None else "ik_unreachable",
                    "target_xyz": list(target_local),
                    "target_axis": (
                        list(target_axis_local)
                        if target_axis_local is not None
                        else None
                    ),
                    "joint_positions": solution,
                },
            )
        if name == "move_arm_joints":
            self._dock_manipulation_active = True
            joints = dict(skill.arguments.get("joints") or {})
            positions = [
                float(joints[joint_name])
                for joint_name in (
                    "Rotation",
                    "Pitch",
                    "Elbow",
                    "Wrist_Pitch",
                    "Wrist_Roll",
                )
            ]
            self._move_dock_left_arm(
                positions, float(skill.arguments.get("duration", 3.0))
            )
            return RobotSkillResult(
                True,
                "left arm joints moved",
                {"joint_positions": self._dock_arm.get_joint_positions()},
            )
        if name == "set_gripper":
            if not self._dock_manipulation_active:
                return None
            command = str(skill.arguments.get("action") or "").lower()
            if command == "open":
                self._set_dock_left_gripper(opened=True)
            elif command == "close":
                self._set_dock_left_gripper(opened=False)
            else:
                opening = float(skill.arguments.get("opening_pct", 0.0))
                self._set_dock_left_gripper(opened=opening >= 50.0)
            return RobotSkillResult(
                True,
                f"left gripper {command or 'set'}",
                {
                    "held_object": self._dock_gripper.held_object,
                    "welds": self._dock_gripper.weld_states(),
                },
            )
        if name == "sim_locate_object":
            query = str(skill.arguments.get("query") or "").strip().lower()
            if query not in {"wand", "棒", "玩具棒", "toy"}:
                return RobotSkillResult(
                    True,
                    f"object not found: {query}",
                    {
                        "operation_success": False,
                        "failure_mode": "object_not_found",
                    },
                )
            point = self._wand_grasp_position_base()
            axis = self._wand_grasp_axis_base()
            count = max(1, int(skill.arguments.get("sample_count", 1)))
            return RobotSkillResult(
                True,
                "located wand",
                {
                    "operation_success": True,
                    "object_name": "wand",
                    "samples": [list(point) for _ in range(count)],
                    "grasp_axis": list(axis),
                    "source": "mujoco_oracle_base_frame",
                },
            )
        if name == "sim_get_object_state":
            objects = {
                "wand": list(self._body_position_base("wand")),
                "wand_dock": list(self._body_position_base("wand_dock")),
            }
            return RobotSkillResult(
                True,
                "dock object state read",
                {
                    "operation_success": True,
                    "objects": objects,
                    "dock_target": list(self._dock_target_base()),
                    "held_object": self._dock_gripper.held_object,
                    "welds": self._dock_gripper.weld_states(),
                },
            )
        return None

    def _move_dock_left_arm(self, positions: list[float], duration: float) -> None:
        """使用确定性的运动学插值移动左臂。"""
        if self._dock_session is None:
            return
        import mujoco

        actuator_ids = self.adapter.arm_actuator_indices(self._dock_arm_side)[:5]
        if len(positions) != len(actuator_ids):
            raise ValueError("left arm requires five joint positions")
        starts = [
            self._joint_position_for_actuator(actuator_id)
            for actuator_id in actuator_ids
        ]
        steps = max(2, int(max(0.0, duration) * 60.0))
        frame_period = duration / steps if self._viewer is not None else 0.0
        for step in range(steps):
            alpha = (step + 1) / steps
            self._dock_session.clamp_base()
            for actuator_id, start, target in zip(
                actuator_ids, starts, positions, strict=True
            ):
                value = start + alpha * (float(target) - start)
                self.data.ctrl[actuator_id] = value
                joint_id = int(self.model.actuator_trnid[actuator_id][0])
                qpos_addr = int(self.model.jnt_qposadr[joint_id])
                dof_addr = int(self.model.jnt_dofadr[joint_id])
                self.data.qpos[qpos_addr] = value
                self.data.qvel[dof_addr] = 0.0
            mujoco.mj_kinematics(self.model, self.data)
            self._sync_active_dock_welds()
            self._sync_viewer()
            if frame_period > 0.0:
                time.sleep(frame_period)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        self._sync_active_dock_welds()

    def _set_dock_left_gripper(self, *, opened: bool) -> None:
        """以运动学方式设置 dock 夹爪，并更新抓取 weld。"""
        import mujoco

        from hey_robot.robot_runtime.simulation.dock_manipulation.gripper import (
            JAW_CLOSED,
            JAW_OPEN,
        )

        if opened:
            self._dock_gripper._release_all()
        actuator_id = int(self._dock_gripper._actuator_id)
        joint_id = int(self.model.actuator_trnid[actuator_id][0])
        qpos_addr = int(self.model.jnt_qposadr[joint_id])
        dof_addr = int(self.model.jnt_dofadr[joint_id])
        target = JAW_OPEN if opened else JAW_CLOSED
        self.data.ctrl[actuator_id] = target
        self.data.qpos[qpos_addr] = target
        self.data.qvel[dof_addr] = 0.0
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        self._dock_gripper._is_open = opened
        if opened:
            self._dock_gripper._held_object = None
            self._dock_manipulation_active = False
            self._sync_active_dock_welds()
        else:
            self._activate_dock_grip_weld()

    def _activate_dock_grip_weld(self) -> None:
        """当 wand 在可达范围内时，将其附着到活动 dock 夹爪。"""
        import mujoco

        grasp_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "wand_grasp"
        )
        ee_position = self._dock_arm.ee_position()
        grasp_position = np.asarray(self.data.site_xpos[grasp_site_id], dtype=float)
        if float(np.linalg.norm(ee_position - grasp_position)) >= 0.06:
            self._dock_gripper._held_object = None
            return

        grip_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grip_weld"
        )
        dock_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "dock_weld"
        )
        self.data.eq_active[grip_id] = 1
        if dock_id >= 0:
            self.data.eq_active[dock_id] = 0

        body1_id = int(self.model.eq_obj1id[grip_id])
        body2_id = int(self.model.eq_obj2id[grip_id])
        position1 = self.data.xpos[body1_id]
        rotation1 = self.data.xmat[body1_id].reshape(3, 3)
        position2 = self.data.xpos[body2_id]
        rotation2 = self.data.xmat[body2_id].reshape(3, 3)
        relative_position = rotation1.T @ (position2 - position1)
        relative_rotation = rotation1.T @ rotation2
        relative_quaternion = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(relative_quaternion, relative_rotation.reshape(-1))
        self.model.eq_data[grip_id, :3] = 0.0
        self.model.eq_data[grip_id, 3:6] = relative_position
        self.model.eq_data[grip_id, 6:10] = relative_quaternion
        self._dock_gripper._held_object = "wand"
        self._sync_active_dock_welds()

    def _sync_active_dock_welds(self) -> None:
        """在运动学运动期间，为自由 dock 物体应用活动 weld 位姿。"""
        import mujoco

        for equality_name in ("dock_weld", "grip_weld"):
            equality_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_name
            )
            if equality_id < 0 or not bool(self.data.eq_active[equality_id]):
                continue
            body1_id = int(self.model.eq_obj1id[equality_id])
            body2_id = int(self.model.eq_obj2id[equality_id])
            joint_start = int(self.model.body_jntadr[body2_id])
            if joint_start < 0:
                continue
            joint_id = joint_start
            if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            qpos_addr = int(self.model.jnt_qposadr[joint_id])
            dof_addr = int(self.model.jnt_dofadr[joint_id])
            relative_position = np.asarray(
                self.model.eq_data[equality_id, 3:6], dtype=float
            )
            relative_quaternion = np.asarray(
                self.model.eq_data[equality_id, 6:10], dtype=float
            )
            if float(np.linalg.norm(relative_quaternion)) <= 0.0:
                relative_quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            rotation1 = self.data.xmat[body1_id].reshape(3, 3)
            position = self.data.xpos[body1_id] + rotation1 @ relative_position
            quaternion = np.zeros(4, dtype=float)
            mujoco.mju_mulQuat(
                quaternion,
                np.asarray(self.data.xquat[body1_id], dtype=float),
                relative_quaternion,
            )
            self.data.qpos[qpos_addr : qpos_addr + 3] = position
            self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = quaternion
            self.data.qvel[dof_addr : dof_addr + 6] = 0.0
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)

    @staticmethod
    def _xyz(value: Any) -> tuple[float, float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError("target_xyz must contain three numbers")
        return (float(value[0]), float(value[1]), float(value[2]))

    def _base_to_world(
        self, point: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        import mujoco

        root_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "root")
        root = self.data.xpos[root_id]
        yaw = float(self.data.qpos[2])
        cosine, sine = math.cos(yaw), math.sin(yaw)
        return (
            float(root[0] + cosine * point[0] - sine * point[1]),
            float(root[1] + sine * point[0] + cosine * point[1]),
            float(root[2] + point[2]),
        )

    def _base_vector_to_world(
        self, vector: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        yaw = float(self.data.qpos[2])
        cosine, sine = math.cos(yaw), math.sin(yaw)
        return (
            float(cosine * vector[0] - sine * vector[1]),
            float(sine * vector[0] + cosine * vector[1]),
            float(vector[2]),
        )

    def _world_to_base(self, point: np.ndarray) -> tuple[float, float, float]:
        import mujoco

        root_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "root")
        delta = np.asarray(point, dtype=float) - self.data.xpos[root_id]
        yaw = float(self.data.qpos[2])
        cosine, sine = math.cos(yaw), math.sin(yaw)
        return (
            float(cosine * delta[0] + sine * delta[1]),
            float(-sine * delta[0] + cosine * delta[1]),
            float(delta[2]),
        )

    def _world_vector_to_base(self, vector: np.ndarray) -> tuple[float, float, float]:
        yaw = float(self.data.qpos[2])
        cosine, sine = math.cos(yaw), math.sin(yaw)
        return (
            float(cosine * vector[0] + sine * vector[1]),
            float(-sine * vector[0] + cosine * vector[1]),
            float(vector[2]),
        )

    def _body_position_base(self, name: str) -> tuple[float, float, float]:
        import mujoco

        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self._world_to_base(self.data.xpos[body_id])

    def _wand_grasp_position_base(self) -> tuple[float, float, float]:
        import mujoco

        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "wand_grasp")
        return self._world_to_base(self.data.site_xpos[site_id])

    def _wand_grasp_axis_base(self) -> tuple[float, float, float]:
        import mujoco

        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "wand_grasp")
        ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "wand_ball")
        axis_world = np.asarray(self.data.geom_xpos[ball_id], dtype=float) - np.asarray(
            self.data.site_xpos[site_id], dtype=float
        )
        norm = float(np.linalg.norm(axis_world))
        if norm <= 0.0:
            return (1.0, 0.0, 0.0)
        return self._world_vector_to_base(axis_world / norm)

    def _dock_target_base(self) -> tuple[float, float, float]:
        """返回可恢复 wand 停靠位姿的夹爪目标。"""
        import mujoco

        dock_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wand_dock")
        dock = self._world_to_base(self.data.xpos[dock_id])
        return (dock[0], dock[1], dock[2] + 0.255)

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

        # 官方 XLeRobot 暴露的是世界坐标系 root joints。
        # 这里只使用官方 yaw joint，将公开的机体坐标系命令转换为 root X/Y。
        phi = float(self.data.qpos[2])
        qvel_x = math.cos(phi) * vy - math.sin(phi) * vx
        qvel_y = math.sin(phi) * vy + math.cos(phi) * vx
        arm_locks = self._arm_joint_locks()
        arm_ctrl = {
            actuator_idx: float(self.data.ctrl[actuator_idx])
            for actuator_idx, _, _, _ in arm_locks
        }

        for _ in range(steps):
            self._restore_arm_locks(arm_locks, arm_ctrl)
            mujoco.mj_step1(self.model, self.data)
            self.data.qvel[0] = qvel_x
            self.data.qvel[1] = qvel_y
            self.data.qvel[2] = vw
            self.data.qacc[0] = 0.0
            self.data.qacc[1] = 0.0
            self.data.qacc[2] = 0.0
            mujoco.mj_step2(self.model, self.data)
            # 该仿真器中的底盘运动是运动学方式。双臂也保持运动学方式，
            # 避免底盘瞬移向动态关节注入冲量。
            self._restore_arm_locks(arm_locks, arm_ctrl)
            mujoco.mj_kinematics(self.model, self.data)
            self._sync_viewer()
        self._stop_base_motion()

    def _arm_joint_locks(self) -> list[tuple[int, int, int, float]]:
        """快照双臂的 actuator、qpos 和 dof 地址。"""
        locks: list[tuple[int, int, int, float]] = []
        actuator_indices: set[int] = set()
        for arm in ("left", "right"):
            actuator_indices.update(self.adapter.arm_actuator_indices(arm))
        num_actuator = int(self.model.nu)
        for actuator_idx in sorted(actuator_indices):
            if actuator_idx >= num_actuator:
                continue
            joint_id = int(self.model.actuator_trnid[actuator_idx][0])
            if joint_id < 0 or joint_id >= self.model.njnt:
                continue
            qpos_addr = int(self.model.jnt_qposadr[joint_id])
            dof_addr = int(self.model.jnt_dofadr[joint_id])
            locks.append(
                (
                    actuator_idx,
                    qpos_addr,
                    dof_addr,
                    float(self.data.qpos[qpos_addr]),
                )
            )
        return locks

    def _restore_arm_locks(
        self,
        locks: list[tuple[int, int, int, float]],
        controls: dict[int, float],
    ) -> None:
        """恢复机械臂目标，并消除底盘引入的关节运动。"""
        for actuator_idx, qpos_addr, dof_addr, position in locks:
            self.data.ctrl[actuator_idx] = controls[actuator_idx]
            self.data.qpos[qpos_addr] = position
            self.data.qvel[dof_addr] = 0.0
            self.data.qacc[dof_addr] = 0.0

    def _stop_base_motion(self) -> None:
        if self.data is None:
            return
        import mujoco

        # 同时清零 actuator target 和仿真底盘状态，避免后续机械臂/夹爪 settle 步骤
        # 积分残余底盘运动。
        for lock_name in ("base_x_lock", "base_y_lock", "base_yaw_lock"):
            lock_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, lock_name
            )
            if lock_id >= 0:
                self.data.ctrl[lock_id] = 0.0
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
        targets = {
            "head_pan_hold": ("head_pan_joint", _DEFAULT_HEAD_PAN),
            "head_tilt_hold": ("head_tilt_joint", _DEFAULT_HEAD_TILT),
        }
        for actuator_name, (joint_name, target) in targets.items():
            actuator_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            if actuator_id >= 0:
                self.data.ctrl[actuator_id] = target
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id >= 0:
                qpos_addr = int(self.model.jnt_qposadr[joint_id])
                dof_addr = int(self.model.jnt_dofadr[joint_id])
                self.data.qpos[qpos_addr] = target
                self.data.qvel[dof_addr] = 0.0

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
        num_ctrl = len(self.data.ctrl)
        for name in _ARM_JOINT_NAMES:
            indices = self.adapter.joint_to_actuators(name)
            if indices is None:
                continue
            # 选择第一个在范围内的 actuator 索引。
            valid = next((idx for idx in indices if idx < num_ctrl), None)
            joint_states[name] = (
                float(self.data.ctrl[valid]) if valid is not None else 0.0
            )
        gripper_indices = self.adapter.gripper_actuator_indices()
        jaw_l = 0.0
        if gripper_indices is not None:
            valid_grip = next((idx for idx in gripper_indices if idx < num_ctrl), None)
            if valid_grip is not None:
                jaw_l = self._joint_position_for_actuator(valid_grip)
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
        # 从旋转矩阵中提取 yaw：atan2(xmat[3], xmat[0])
        # xmat 是按 flat 形式存储的 3x3：[xx, xy, xz, yx, yy, yz, zx, zy, zz]
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
            state=self._protocol_state(),  # type: ignore[arg-type]
            location_id=self._location_id(),
            motion_state="idle" if success else "unknown",
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

    def _protocol_state(self) -> str:
        if self.state in {"idle", "executing", "offline", "unknown"}:
            return self.state
        if self.state in {"failed", "closed"}:
            return "offline" if self.state == "closed" else "error"
        # created、skill_completed 等内部生命周期状态不是 transport 状态；
        # 它们不表示物理动作仍在运行。
        return "idle"

    def _location_id(self) -> str | None:
        pose = self._base_pose()
        x = pose["x_cm"] / 100.0
        y = pose["y_cm"] / 100.0
        if 0.0 <= x <= 6.0 and 0.0 <= y <= 5.0:
            return "room:living_room"
        if 0.0 <= x <= 6.0 and 5.0 < y <= 10.0:
            return "room:dining_room"
        if 14.0 <= x <= 20.0 and 0.0 <= y <= 5.0:
            return "room:kitchen"
        return None

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
        track_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if track_body < 0:
            track_body = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, _ROBOT_BODY
            )
        camera.trackbodyid = track_body
        camera.distance = float(layout.get("distance", 0.8))
        camera.azimuth = float(layout.get("azimuth", 180.0))
        camera.elevation = float(layout.get("elevation", -20.0))
        camera.lookat = np.array(layout.get("lookat", [0.0, 0.0, 0.0]), dtype=float)
        return camera
