from __future__ import annotations

import asyncio
import contextlib
import functools
import threading
import time
from typing import Any

import numpy as np

from hey_robot.logging import HeyRobotLogger
from hey_robot.protocol import (
    Envelope,
    RobotAction,
    RobotSkillAction,
    RobotSkillResult,
    RobotStatus,
)
from hey_robot.robot_api import (
    DriverObservation,
    ObservationAsset,
    RobotCapabilities,
    RobotDriverContext,
    RobotHealth,
)
from hey_robot.robot_backends.simulation.mujoco_logging import (
    configure_mujoco_warning_logging,
)
from hey_robot.robot_backends.simulation.mujoco_platform import (
    configure_mujoco_gl_backend,
    ensure_render_context,
    resolve_mjcf_path,
)
from hey_robot.robot_backends.simulation.skill_adapter import XLeRobotSimSkillAdapter
from hey_robot.robot_backends.simulation.xlerobot_sim_dock import DockManipulation
from hey_robot.robot_backends.simulation.xlerobot_sim_kernel import XLeRobotSimKernel
from hey_robot.robot_runtime.skill_gate import SkillAdmissionGate

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


class XLeRobotSimDriver:
    """实现 RobotDriver 协议的 MuJoCo 仿真驱动。"""

    def __init__(self, context: RobotDriverContext) -> None:
        self.context = context
        self.robot_id = context.robot_id
        self.settings = dict(context.settings or {})
        # 在测试或调用方绕过 ``start``、直接通过该驱动构造 MjModel/MjData 前完成配置。
        configure_mujoco_gl_backend()
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
        self.contracts = SkillAdmissionGate(context.action_specs)
        self._kernel = XLeRobotSimKernel(
            robot_id=self.robot_id,
            adapter=self.adapter,
            camera_layout=_DEFAULT_SIM_CAMERA_LAYOUT,
        )
        viewer_cfg = self.settings.get("viewer", {}) or {}
        self._viewer_enabled = bool(viewer_cfg.get("enabled", False))

        self.state = "created"
        self._action_lock = asyncio.Lock()
        self._active_motion_stop: threading.Event | None = None
        self.frame_id = 0
        self.last_error: str | None = None
        self.last_skill_result: RobotSkillResult | None = None
        self.last_camera: dict[str, Any] = {"frame_available": False, "frame_id": None}
        self.last_cameras_status: dict[str, dict[str, Any]] = {}
        self.last_battery: dict[str, Any] = {
            "status": "normal",
            "voltage": 12.0,
            "percentage": 85.0,
        }
        self.startup_diagnostics: dict[str, Any] = {}

        self._last_rendered_frame: np.ndarray | None = None
        self._dock = DockManipulation(self)

    # Resource compatibility properties. The kernel remains their sole owner.
    @property
    def model(self) -> Any:
        return self._kernel.model

    @model.setter
    def model(self, value: Any) -> None:
        self._kernel.model = value

    @property
    def data(self) -> Any:
        return self._kernel.data

    @data.setter
    def data(self, value: Any) -> None:
        self._kernel.data = value

    @property
    def renderer(self) -> Any:
        return self._kernel.renderer

    @renderer.setter
    def renderer(self, value: Any) -> None:
        self._kernel.renderer = value

    @property
    def _viewer(self) -> Any:
        return self._kernel.viewer

    @_viewer.setter
    def _viewer(self, value: Any) -> None:
        self._kernel.viewer = value

    @property
    def _scene_camera(self) -> Any:
        return self._kernel.scene_camera

    @_scene_camera.setter
    def _scene_camera(self, value: Any) -> None:
        self._kernel.scene_camera = value

    @property
    def _scene_cameras(self) -> dict[str, Any]:
        return self._kernel.scene_cameras

    @_scene_cameras.setter
    def _scene_cameras(self, value: dict[str, Any]) -> None:
        self._kernel.scene_cameras = value

    @property
    def _data_lock(self) -> threading.RLock:
        return self._kernel.data_lock

    @property
    def _emergency_stop_active(self) -> bool:
        return self._kernel.emergency_stop_active

    @_emergency_stop_active.setter
    def _emergency_stop_active(self, value: bool) -> None:
        self._kernel.emergency_stop_active = value

    @property
    def last_arm_status(self) -> dict[str, Any]:
        return self._kernel.last_arm_status

    # RobotDriver 协议

    async def start(self) -> None:
        configure_mujoco_gl_backend()
        import mujoco
        import mujoco.viewer

        configure_mujoco_warning_logging(
            self.context.deployment_id, mujoco_module=mujoco
        )
        mjcf_path = str(resolve_mjcf_path(self.settings))
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
                self._kernel.set_actuator_joint_position(idx, pos)
        self._kernel.hold_head_camera()

        mujoco.mj_forward(self.model, self.data)
        self._initialize_dock_manipulation()
        logger.info(f"{self.robot_id} simulation data initialized")

        # Linux headless 渲染在创建 Renderer 前需要 EGL context。
        # Windows 使用 WGL；如果缺少 EGL.dll，导入 mujoco.egl 会失败。
        logger.info(f"{self.robot_id} ensuring MuJoCo render context")
        ensure_render_context(self._render_width, self._render_height)

        # Renderer 必须在调用线程创建，因为该线程持有 GL context。
        logger.info(f"{self.robot_id} creating MuJoCo renderer")
        self.renderer = mujoco.Renderer(
            self.model,
            self._render_height,
            self._render_width,
        )
        logger.info(f"{self.robot_id} MuJoCo renderer ready")

        self._scene_cameras = {
            name: self._kernel.build_scene_camera(name) for name in self._camera_names
        }
        self._scene_camera = self._scene_cameras.get(self._default_camera)

        self._kernel.update_arm_status()
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
            supports_interrupt=True,
            metadata={
                "body": "xlerobot",
                "robot_family": self.context.robot_family,
                "environment": self.context.environment,
                "driver_kind": self.context.driver_kind,
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
        self._kernel.update_arm_status()

        frames = self._kernel.render_frames()
        with self._data_lock:
            self._kernel.sync_viewer()
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
            proprioception=self._kernel.proprioception(),
            metadata={
                "driver": "xlerobot_sim",
                "body": "xlerobot",
                "robot_family": self.context.robot_family,
                "environment": self.context.environment,
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
                "base_pose": self._base_pose(),
                "last_skill_result": self.last_skill_result.to_dict()
                if self.last_skill_result
                else None,
                "readiness": self.readiness(),
            },
        )

    async def apply_action(self, action: RobotAction) -> RobotStatus:
        """Serialize actions and let stop requests preempt active simulation work."""
        with contextlib.suppress(ValueError):
            if RobotSkillAction.from_robot_action(action).name == "stop_motion":
                self._request_motion_stop()
        async with self._action_lock:
            return await self._apply_action(action)

    async def _apply_action(self, action: RobotAction) -> RobotStatus:
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
            self._kernel.update_arm_status()
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
                f"{self._kernel.gripper_debug_state()}"
            )

        num_ctrl = len(self.data.ctrl)
        if cmd.arm_targets:
            self._kernel.stop_base_motion()
            if cmd.delta_mode:
                for idx, delta in cmd.arm_targets.items():
                    if idx < num_ctrl:
                        self.data.ctrl[idx] = float(self.data.ctrl[idx]) + delta
            else:
                for idx, target in cmd.arm_targets.items():
                    if idx < num_ctrl:
                        self.data.ctrl[idx] = target

        if cmd.jaw_left is not None:
            self._kernel.stop_base_motion()
            gripper_indices = self.adapter.gripper_actuator_indices()
            if gripper_indices is not None and gripper_indices[0] < num_ctrl:
                self.data.ctrl[gripper_indices[0]] = cmd.jaw_left
                logger.info(
                    f"{self.robot_id} gripper_debug phase=write_left "
                    f"actuator={gripper_indices[0]} target={cmd.jaw_left} "
                    f"{self._kernel.gripper_debug_state()}"
                )
            else:
                logger.warning(
                    f"{self.robot_id} gripper_debug phase=write_left "
                    "missing_gripper_indices"
                )
        if cmd.jaw_right is not None:
            self._kernel.stop_base_motion()
            gripper_indices = self.adapter.gripper_actuator_indices()
            if gripper_indices is not None and gripper_indices[1] < num_ctrl:
                self.data.ctrl[gripper_indices[1]] = cmd.jaw_right
                logger.info(
                    f"{self.robot_id} gripper_debug phase=write_right "
                    f"actuator={gripper_indices[1]} target={cmd.jaw_right} "
                    f"{self._kernel.gripper_debug_state()}"
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
            self.state = "executing"
            completed = await self._run_simulation_work(
                self._step_velocity, steps, cmd.vx, cmd.vy, cmd.vw
            )
            self._kernel.stop_base_motion()
            if not completed:
                result = RobotSkillResult(
                    False,
                    f"{cmd.skill_name} interrupted and stopped",
                    {
                        "skill": skill.to_dict(),
                        "failure_mode": "interrupted",
                        "base_pose": self._base_pose(),
                        "stop_confirmed": True,
                    },
                )
                self.last_skill_result = result
                self.state = "idle"
                self.last_error = result.message
                return self._status_for_action(action, success=False)
        elif cmd.skill_name == "stop_motion":
            self._kernel.stop_base_motion()
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
                    f"{self._kernel.gripper_debug_state()}"
                )
            hold_ctrl = {
                idx: float(self.data.ctrl[idx])
                for idx in self._kernel.arm_hold_actuator_indices(cmd.arm_targets)
                if 0 <= idx < len(self.data.ctrl)
            }
            lock_qpos = (
                self._kernel.non_gripper_arm_joint_positions()
                if is_gripper_command and not cmd.arm_targets
                else None
            )
            drive_qpos = (
                self._kernel.commanded_gripper_joint_positions()
                if is_gripper_command and not cmd.arm_targets
                else None
            )
            self.state = "executing"
            completed = await self._run_simulation_work(
                self._kernel.step,
                settle_steps,
                hold_ctrl=hold_ctrl,
                lock_qpos=lock_qpos,
                drive_qpos=drive_qpos,
            )
            if not completed:
                result = RobotSkillResult(
                    False,
                    f"{cmd.skill_name} interrupted and stopped",
                    {
                        "skill": skill.to_dict(),
                        "failure_mode": "interrupted",
                        "base_pose": self._base_pose(),
                        "stop_confirmed": True,
                    },
                )
                self.last_skill_result = result
                self.state = "idle"
                self.last_error = result.message
                return self._status_for_action(action, success=False)
            if is_gripper_command:
                logger.info(
                    f"{self.robot_id} gripper_debug phase=after_settle_before_base_restore "
                    f"{self._kernel.gripper_debug_state()}"
                )
            self.data.qpos[:3] = base_qpos
            import mujoco

            mujoco.mj_forward(self.model, self.data)
            self._kernel.stop_base_motion()
            if is_gripper_command:
                logger.info(
                    f"{self.robot_id} gripper_debug phase=after_base_restore "
                    f"{self._kernel.gripper_debug_state()}"
                )

        result = RobotSkillResult(
            True,
            cmd.message,
            {
                "skill": skill.to_dict(),
                "base_pose": self._base_pose(),
            },
        )
        self.last_skill_result = result
        self.state = "skill_completed"
        self.last_error = None
        self._kernel.update_arm_status()
        if is_gripper_command:
            logger.info(
                f"{self.robot_id} gripper_debug phase=final_status "
                f"gripper_opening_pct={self.last_arm_status.get('gripper_opening_pct')} "
                f"{self._kernel.gripper_debug_state()}"
            )
        status = self._status_for_action(action, success=True)
        self.state = "idle"
        return status

    async def reset(self) -> RobotStatus:
        self._request_motion_stop()
        async with self._action_lock:
            return await self._reset()

    async def _reset(self) -> RobotStatus:
        import mujoco

        with self._data_lock:
            mujoco.mj_resetData(self.model, self.data)
            rest = self.adapter.arm_rest_positions()
            num_ctrl = len(self.data.ctrl)
            for idx, pos in rest.items():
                if idx < num_ctrl:
                    self.data.ctrl[idx] = pos
                    self._kernel.set_actuator_joint_position(idx, pos)
            self._kernel.hold_head_camera()
            self._kernel.stop_base_motion()
            mujoco.mj_forward(self.model, self.data)
        self.state = "idle"
        self.last_error = None
        self.last_skill_result = RobotSkillResult(True, "sim reset", {"skill": "reset"})
        self.frame_id = 0
        self._last_rendered_frame = None
        self._kernel.update_arm_status()
        return await self.status()

    async def close(self) -> None:
        self._request_motion_stop()
        async with self._action_lock:
            with self._data_lock:
                if self._viewer is not None:
                    self._viewer.close()
                self._viewer = None
                if self.renderer is not None:
                    self.renderer.close()
                self.renderer = None
                self._scene_camera = None
                self._scene_cameras = {}
                self._dock = DockManipulation(self)
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
        rendered = self._kernel.render_frames()
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
                    camera = self._kernel.build_scene_camera(name)
                    self._scene_cameras[name] = camera
                self._scene_camera = camera
                frames[name] = self._kernel.render_frame()
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
        self._kernel.stop_base_motion()
        for i, idx in enumerate(indices):
            if i >= len(targets_rad):
                break
            lo = float(self.model.actuator_ctrlrange[idx][0])
            hi = float(self.model.actuator_ctrlrange[idx][1])
            self.data.ctrl[idx] = float(np.clip(targets_rad[i], lo, hi))
        self._kernel.update_arm_status()

    def step_control(self, dt: float) -> None:
        """让 MuJoCo 前进 dt 秒，同时保持底盘静止。"""
        import mujoco

        if self.model is None or self.data is None:
            return
        timestep = self.model.opt.timestep
        steps = max(1, int(dt / timestep))
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            self._kernel.stop_base_motion()
            self._kernel.sync_viewer()

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

    def _request_motion_stop(self) -> None:
        active = self._active_motion_stop
        if active is not None:
            active.set()

    async def _run_simulation_work(
        self, function: Any, *args: Any, **kwargs: Any
    ) -> bool:
        """Run cancellable MuJoCo work and do not return before its thread stops."""
        stop_event = threading.Event()
        self._active_motion_stop = stop_event
        work = functools.partial(function, *args, stop_event, **kwargs)
        worker = asyncio.create_task(asyncio.to_thread(work))
        try:
            return bool(await asyncio.shield(worker))
        except asyncio.CancelledError:
            stop_event.set()
            # ``to_thread`` cannot be force-cancelled. Waiting here keeps the
            # Skill resource lease until the physical/simulation writer is quiet.
            try:
                await asyncio.shield(worker)
            except Exception as exc:
                logger.error(
                    f"{self.robot_id} cancelled simulation writer failed: {exc}"
                )
            finally:
                self.state = "idle"
                self.last_error = "action cancelled after motion stopped"
            raise
        finally:
            if self._active_motion_stop is stop_event:
                self._active_motion_stop = None

    def _initialize_dock_manipulation(self) -> None:
        """Compatibility entry point for tests; dock state is owned by its component."""
        self._dock.initialize()

    def _execute_dock_primitive(
        self, skill: RobotSkillAction
    ) -> RobotSkillResult | None:
        return self._dock.execute(skill)

    @property
    def _dock_session(self) -> Any:
        return self._dock.session

    @property
    def _dock_arm(self) -> Any:
        return self._dock.arm

    @property
    def _dock_gripper(self) -> Any:
        return self._dock.gripper

    @property
    def _dock_arm_side(self) -> str:
        return self._dock.arm_side

    # Compatibility seams for callers that historically probed driver internals.
    def _step_velocity(
        self,
        steps: int,
        vx: float,
        vy: float,
        vw: float,
        stop_event: threading.Event | None = None,
    ) -> bool:
        return self._kernel.step_velocity(steps, vx, vy, vw, stop_event)

    def _joint_position_for_actuator(self, actuator_idx: int) -> float:
        return self._kernel.joint_position_for_actuator(actuator_idx)

    def _base_pose(self) -> dict[str, float]:
        return self._kernel.base_pose()

    def _sync_viewer(self) -> None:
        self._kernel.sync_viewer()

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
        stop_confirmed = bool(
            self.last_skill_result
            and self.last_skill_result.data.get("stop_confirmed", False)
        )
        return RobotStatus(
            envelope=self._envelope(),
            frame_id=self.frame_id,
            state=self._protocol_state(),  # type: ignore[arg-type]
            location_id=self._location_id(),
            motion_state=(
                "idle" if success else "stopped" if stop_confirmed else "unknown"
            ),
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
                "base_pose": self._base_pose(),
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
