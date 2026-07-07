# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
# Modified for Xbotics Hey Robot: shared dock arm kernel.
from __future__ import annotations

import time
from typing import Any, Protocol

import numpy as np


class DockSession(Protocol):
    """Protocol for the dock session used by arm and gripper kernels."""

    model: Any
    data: Any

    @property
    def connected(self) -> bool: ...
    @property
    def viewer(self) -> Any: ...
    def require_connected(self) -> None: ...
    def id_for(self, object_type: Any, name: str) -> int: ...
    def lock_base(self) -> None: ...
    def clamp_base(self) -> None: ...


ARM_JOINT_NAMES: tuple[str, ...] = (
    "Rotation",
    "Pitch",
    "Elbow",
    "Wrist_Pitch",
    "Wrist_Roll",
)
ARM_ACTUATOR_NAMES: tuple[str, ...] = (
    "Rotation_L",
    "Pitch_L",
    "Elbow_L",
    "Wrist_Pitch_L",
    "Wrist_Roll_L",
)
EE_BODY_NAME = "Fixed_Jaw_tip"

IK_MAX_ITER = 100
IK_TOL = 1e-3
IK_STEP_SIZE = 0.5
IK_DAMPING = 1e-4


class So101MobileArmKernel:
    """5-DOF arm kernel for XLeRobot mobile manipulation."""

    def __init__(
        self,
        session: DockSession,
        *,
        joint_names: tuple[str, ...] = ARM_JOINT_NAMES,
        actuator_names: tuple[str, ...] = ARM_ACTUATOR_NAMES,
        ee_body_name: str = EE_BODY_NAME,
    ) -> None:
        self.session = session
        self._joint_names = joint_names
        self._actuator_names = actuator_names
        self._ee_body_name = ee_body_name
        self._joint_ids: list[int] = []
        self._actuator_ids: list[int] = []
        self._ee_body_id = -1

    @property
    def dof(self) -> int:
        return 5

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._joint_names

    def bind(self) -> None:
        self.session.require_connected()
        import mujoco

        self._joint_ids = [
            self.session.id_for(mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self._joint_names
        ]
        self._actuator_ids = [
            self.session.id_for(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in self._actuator_names
        ]
        self._ee_body_id = self.session.id_for(
            mujoco.mjtObj.mjOBJ_BODY, self._ee_body_name
        )

    def get_joint_positions(self) -> list[float]:
        self.session.require_connected()
        return [
            float(self.session.data.joint(name).qpos[0]) for name in self._joint_names
        ]

    def move_joints(self, positions: list[float], duration: float = 3.0) -> bool:
        self.session.require_connected()
        if len(positions) != self.dof:
            raise ValueError(
                f"expected {self.dof} joint positions, got {len(positions)}"
            )
        if not np.all(np.isfinite(np.asarray(positions, dtype=float))):
            raise ValueError("joint targets must be finite")

        model = self.session.model
        data = self.session.data
        import mujoco

        # Re-lock mobile base at origin before arm motion
        self.session.lock_base()

        dt = float(model.opt.timestep)
        steps = max(1, int(float(duration) / dt))
        sync_interval = max(1, int(1.0 / 60.0 / dt))
        start = [float(data.ctrl[actuator]) for actuator in self._actuator_ids]
        wall_start = time.monotonic()

        for index in range(steps):
            alpha = (index + 1) / steps
            for actuator, initial, target in zip(
                self._actuator_ids, start, positions, strict=True
            ):
                data.ctrl[actuator] = initial + alpha * (target - initial)
            self.session.clamp_base()
            mujoco.mj_step(model, data)
            if self.session.viewer is not None and index % sync_interval == 0:
                self.session.viewer.sync()
                elapsed_sim = (index + 1) * dt
                remaining = elapsed_sim - (time.monotonic() - wall_start)
                if remaining > 0:
                    time.sleep(remaining)
        # Snap actuator targets to actual joint positions to prevent
        # residual-force drift in subsequent gripper motions.
        for actuator_id, name in zip(
            self._actuator_ids, self._joint_names, strict=True
        ):
            data.ctrl[actuator_id] = float(data.joint(name).qpos[0])
        return True

    def stop(self) -> None:
        import mujoco

        positions = self.get_joint_positions()
        for actuator, position in zip(self._actuator_ids, positions, strict=True):
            self.session.data.ctrl[actuator] = position
        for _ in range(max(1, int(0.05 / self.session.model.opt.timestep))):
            self.session.clamp_base()
            mujoco.mj_step(self.session.model, self.session.data)
        if self.session.viewer is not None:
            self.session.viewer.sync()

    def fk(self, joint_positions: list[float]) -> tuple[list[float], list[list[float]]]:
        if len(joint_positions) != self.dof:
            raise ValueError(f"expected {self.dof} joints")
        import mujoco

        data = self.session.data
        old_qpos = data.qpos.copy()
        old_qvel = data.qvel.copy()
        old_ctrl = data.ctrl.copy()
        try:
            for name, position in zip(self._joint_names, joint_positions, strict=True):
                data.joint(name).qpos[0] = float(position)
            mujoco.mj_forward(self.session.model, data)
            position = data.xpos[self._ee_body_id].copy().tolist()
            rotation = data.xmat[self._ee_body_id].reshape(3, 3).copy().tolist()
            return position, rotation
        finally:
            data.qpos[:] = old_qpos
            data.qvel[:] = old_qvel
            data.ctrl[:] = old_ctrl
            mujoco.mj_forward(self.session.model, data)

    def ik(
        self,
        target_xyz: tuple[float, float, float],
        current_joints: list[float] | None = None,
        target_axis: tuple[float, float, float] | None = None,
        ee_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> list[float] | None:
        import mujoco

        target = np.asarray(target_xyz, dtype=np.float64)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            return None
        axis_target: np.ndarray | None = None
        axis_local = np.asarray(ee_axis, dtype=np.float64)
        axis_local_norm = float(np.linalg.norm(axis_local))
        if axis_local.shape != (3,) or axis_local_norm <= 0.0:
            return None
        axis_local = axis_local / axis_local_norm
        if target_axis is not None:
            axis_target = np.asarray(target_axis, dtype=np.float64)
            axis_target_norm = float(np.linalg.norm(axis_target))
            if (
                axis_target.shape != (3,)
                or not np.all(np.isfinite(axis_target))
                or axis_target_norm <= 0.0
            ):
                return None
            axis_target = axis_target / axis_target_norm
        # Re-lock mobile base at origin before IK
        self.session.lock_base()
        data = self.session.data
        model = self.session.model
        old_qpos = data.qpos.copy()
        old_qvel = data.qvel.copy()
        old_ctrl = data.ctrl.copy()
        seed = list(current_joints) if current_joints is not None else [0.0] * self.dof
        if len(seed) != self.dof:
            return None
        if axis_target is not None:
            best_solution: list[float] | None = None
            best_dot = -float("inf")
            lower, upper = model.jnt_range[self._joint_ids[-1]]
            for wrist_roll in np.linspace(float(lower), float(upper), 9):
                candidate_seed = list(seed)
                candidate_seed[-1] = float(wrist_roll)
                solution = self.ik(target_xyz, candidate_seed)
                if solution is None:
                    continue
                _, rotation = self.fk(solution)
                axis_current = np.asarray(rotation, dtype=np.float64) @ axis_local
                axis_current /= max(float(np.linalg.norm(axis_current)), 1e-9)
                dot = float(np.dot(axis_current, axis_target))
                if dot > best_dot:
                    best_dot = dot
                    best_solution = solution
            return best_solution
        qpos_addresses = [
            int(model.jnt_qposadr[joint_id]) for joint_id in self._joint_ids
        ]
        dof_addresses = [
            int(model.jnt_dofadr[joint_id]) for joint_id in self._joint_ids
        ]
        result: list[float] | None = None
        try:
            for address, value in zip(qpos_addresses, seed, strict=True):
                data.qpos[address] = float(value)

            for _ in range(IK_MAX_ITER):
                # IK needs transforms and Jacobians, not collision or dynamics.
                # This keeps solving fast in the complete home scene.
                mujoco.mj_kinematics(model, data)
                mujoco.mj_comPos(model, data)
                position_error = target - data.xpos[self._ee_body_id]
                axis_error = np.zeros(3, dtype=np.float64)
                axis_aligned = True
                if axis_target is not None:
                    rotation = data.xmat[self._ee_body_id].reshape(3, 3)
                    axis_current = rotation @ axis_local
                    axis_current /= max(float(np.linalg.norm(axis_current)), 1e-9)
                    axis_error = np.cross(axis_target, axis_current)
                    axis_aligned = float(np.dot(axis_current, axis_target)) > 0.70
                if float(np.linalg.norm(position_error)) < IK_TOL and axis_aligned:
                    result = [float(data.qpos[address]) for address in qpos_addresses]
                    break
                jacobian_pos = np.zeros((3, model.nv), dtype=np.float64)
                jacobian_rot = np.zeros((3, model.nv), dtype=np.float64)
                mujoco.mj_jacBody(
                    model, data, jacobian_pos, jacobian_rot, self._ee_body_id
                )
                arm_jacobian = jacobian_pos[:, dof_addresses]
                error = position_error
                if axis_target is not None:
                    axis_weight = 0.05
                    arm_jacobian = np.vstack(
                        (
                            arm_jacobian,
                            axis_weight * jacobian_rot[:, dof_addresses],
                        )
                    )
                    error = np.concatenate((position_error, axis_weight * axis_error))
                normal = arm_jacobian @ arm_jacobian.T + IK_DAMPING * np.eye(
                    arm_jacobian.shape[0]
                )
                delta = arm_jacobian.T @ np.linalg.solve(normal, error)
                for index, address in enumerate(qpos_addresses):
                    data.qpos[address] += IK_STEP_SIZE * delta[index]
                    lower, upper = model.jnt_range[self._joint_ids[index]]
                    data.qpos[address] = float(
                        np.clip(data.qpos[address], lower, upper)
                    )
            return result
        finally:
            data.qpos[:] = old_qpos
            data.qvel[:] = old_qvel
            data.ctrl[:] = old_ctrl
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)

    def ee_position(self) -> np.ndarray:
        return np.asarray(
            self.session.data.xpos[self._ee_body_id], dtype=np.float64
        ).copy()

    def get_object_positions(self) -> dict[str, list[float]]:
        import mujoco

        result: dict[str, list[float]] = {}
        model = self.session.model
        data = self.session.data
        for body_id in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name is None:
                continue
            joint_start = int(model.body_jntadr[body_id])
            if joint_start < 0:
                continue
            jt = model.jnt_type[joint_start]
            if jt == mujoco.mjtJoint.mjJNT_FREE:
                result[str(name)] = [float(value) for value in data.xpos[body_id]]
        return result

    def diagnostics(self) -> dict[str, Any]:
        return {
            "dof": self.dof,
            "joint_names": list(self.joint_names),
            "joint_ids": list(self._joint_ids),
            "actuator_ids": list(self._actuator_ids),
            "ee_body_id": self._ee_body_id,
        }
