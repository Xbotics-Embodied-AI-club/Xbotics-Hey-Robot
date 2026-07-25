"""MuJoCo state kernel for the XLeRobot simulation driver.

This is the sole owner of model/data/render/viewer resources and their lock. It
contains no RobotDriver protocol or skill admission policy.
"""

from __future__ import annotations

import contextlib
import math
import threading
from typing import Any

import numpy as np

from hey_robot.logging import HeyRobotLogger

logger = HeyRobotLogger(name="xlerobot_sim")
_ROBOT_BODY = "base_link"
_DEFAULT_HEAD_PAN = 0.0
_DEFAULT_HEAD_TILT = 0.25
_ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class XLeRobotSimKernel:
    """Own and mutate the MuJoCo simulation state."""

    def __init__(
        self,
        *,
        robot_id: str,
        adapter: Any,
        camera_layout: dict[str, dict[str, Any]],
    ) -> None:
        self.robot_id = robot_id
        self.adapter = adapter
        self.camera_layout = camera_layout
        self.model: Any = None
        self.data: Any = None
        self.renderer: Any = None
        self.scene_camera: Any = None
        self.scene_cameras: dict[str, Any] = {}
        self.viewer: Any = None
        self.data_lock = threading.RLock()
        self.emergency_stop_active = False
        self.last_arm_status: dict[str, Any] = {}

    def render_frame(self) -> np.ndarray | None:
        if self.renderer is None or self.scene_camera is None:
            return None
        try:
            with self.data_lock:
                self.renderer.update_scene(self.data, camera=self.scene_camera)
                pixels = self.renderer.render()
                return np.array(pixels, dtype=np.uint8)
        except Exception as exc:
            logger.warning(f"{self.robot_id} simulation render failed: {exc}")
            return None

    def render_frames(self) -> dict[str, np.ndarray | None]:
        with self.data_lock:
            previous = self.scene_camera
            frames: dict[str, np.ndarray | None] = {}
            for name, camera in self.scene_cameras.items():
                self.scene_camera = camera
                frames[name] = self.render_frame()
            self.scene_camera = previous
            return frames

    def step_velocity(
        self,
        steps: int,
        vx: float,
        vy: float,
        vw: float,
        stop_event: threading.Event | None = None,
    ) -> bool:
        import mujoco

        stop_event = stop_event or threading.Event()

        # 官方 XLeRobot 暴露的是世界坐标系 root joints。
        # 这里只使用官方 yaw joint，将公开的机体坐标系命令转换为 root X/Y。
        with self.data_lock:
            phi = float(self.data.qpos[2])
            arm_locks = self._arm_joint_locks()
            arm_ctrl = {
                actuator_idx: float(self.data.ctrl[actuator_idx])
                for actuator_idx, _, _, _ in arm_locks
            }
        qvel_x = math.cos(phi) * vy - math.sin(phi) * vx
        qvel_y = math.sin(phi) * vy + math.cos(phi) * vx

        for _ in range(steps):
            if stop_event.is_set() or self.emergency_stop_active:
                with self.data_lock:
                    self.stop_base_motion()
                return False
            with self.data_lock:
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
        with self.data_lock:
            self.stop_base_motion()
        return True

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

    def stop_base_motion(self) -> None:
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

    def hold_head_camera(self) -> None:
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

    def sync_viewer(self) -> None:
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    def step(
        self,
        n: int,
        stop_event: threading.Event | None = None,
        hold_ctrl: dict[int, float] | None = None,
        lock_qpos: dict[int, float] | None = None,
        drive_qpos: dict[int, float] | None = None,
    ) -> bool:
        import mujoco

        stop_event = stop_event or threading.Event()

        for _ in range(n):
            if stop_event.is_set() or self.emergency_stop_active:
                return False
            with self.data_lock:
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
        return True

    def arm_hold_actuator_indices(self, command_targets: dict[int, float]) -> set[int]:
        indices: set[int] = set(command_targets)
        with contextlib.suppress(Exception):
            indices.update(self.adapter.arm_actuator_indices("left"))
            indices.update(self.adapter.arm_actuator_indices("right"))
        gripper_indices = self.adapter.gripper_actuator_indices()
        if gripper_indices is not None:
            indices.update(gripper_indices)
        return indices

    def set_actuator_joint_position(self, actuator_idx: int, value: float) -> None:
        if self.model is None or self.data is None:
            return
        qpos_addr = self._actuator_joint_qpos_addr(actuator_idx)
        if qpos_addr is None:
            return
        self.data.qpos[qpos_addr] = float(value)

    def non_gripper_arm_joint_positions(self) -> dict[int, float]:
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

    def commanded_gripper_joint_positions(self) -> dict[int, float]:
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

    def update_arm_status(self) -> None:
        with self.data_lock:
            self.update_arm_status_unlocked()

    def update_arm_status_unlocked(self) -> None:
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
                jaw_l = self.joint_position_for_actuator(valid_grip)
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

    def proprioception(self) -> list[float]:
        with self.data_lock:
            return self.proprioception_unlocked()

    def proprioception_unlocked(self) -> list[float]:
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

    def joint_position_for_actuator(self, actuator_idx: int) -> float:
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

    def gripper_debug_state(self) -> str:
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

    def base_pose(self) -> dict[str, float]:
        with self.data_lock:
            return self.base_pose_unlocked()

    def base_pose_unlocked(self) -> dict[str, float]:
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

    def build_scene_camera(self, name: str):
        import mujoco

        layout = dict(self.camera_layout.get(name, {}))
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
