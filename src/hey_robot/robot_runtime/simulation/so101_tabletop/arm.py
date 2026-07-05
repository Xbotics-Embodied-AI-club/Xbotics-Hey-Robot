# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
# Modified for Xbotics Hey Robot.
from __future__ import annotations

import time
from typing import Any

import numpy as np

from hey_robot.robot_runtime.simulation.so101_tabletop.session import (
    So101TabletopSession,
)

ARM_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
ARM_ACTUATOR_NAMES: tuple[str, ...] = (
    "act_shoulder_pan",
    "act_shoulder_lift",
    "act_elbow_flex",
    "act_wrist_flex",
    "act_wrist_roll",
)
EE_SITE_NAME = "ee_site"

IK_MAX_ITER = 100
IK_TOL = 1e-3
IK_STEP_SIZE = 0.5
IK_DAMPING = 1e-4


class So101ArmKernel:
    def __init__(self, session: So101TabletopSession) -> None:
        self.session = session
        self._joint_ids: list[int] = []
        self._actuator_ids: list[int] = []
        self._site_id = -1

    @property
    def dof(self) -> int:
        return 5

    @property
    def joint_names(self) -> tuple[str, ...]:
        return ARM_JOINT_NAMES

    def bind(self) -> None:
        self.session.require_connected()
        import mujoco

        self._joint_ids = [
            self.session.id_for(mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ARM_JOINT_NAMES
        ]
        self._actuator_ids = [
            self.session.id_for(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ARM_ACTUATOR_NAMES
        ]
        self._site_id = self.session.id_for(mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME)

    def get_joint_positions(self) -> list[float]:
        self.session.require_connected()
        return [
            float(self.session.data.joint(name).qpos[0]) for name in ARM_JOINT_NAMES
        ]

    def move_joints(self, positions: list[float], duration: float = 3.0) -> bool:
        self.session.require_connected()
        if len(positions) != self.dof:
            raise ValueError(
                f"expected {self.dof} SO101 joint positions, got {len(positions)}"
            )
        if not np.all(np.isfinite(np.asarray(positions, dtype=float))):
            raise ValueError("joint targets must be finite")

        model = self.session.model
        data = self.session.data
        import mujoco

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
            mujoco.mj_step(model, data)
            if self.session.viewer is not None and index % sync_interval == 0:
                self.session.viewer.sync()
                elapsed_sim = (index + 1) * dt
                remaining = elapsed_sim - (time.monotonic() - wall_start)
                if remaining > 0:
                    time.sleep(remaining)
        return True

    def stop(self) -> None:
        import mujoco

        positions = self.get_joint_positions()
        for actuator, position in zip(self._actuator_ids, positions, strict=True):
            self.session.data.ctrl[actuator] = position
        for _ in range(max(1, int(0.05 / self.session.model.opt.timestep))):
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
            for name, position in zip(ARM_JOINT_NAMES, joint_positions, strict=True):
                data.joint(name).qpos[0] = float(position)
            mujoco.mj_forward(self.session.model, data)
            position = data.site_xpos[self._site_id].copy().tolist()
            rotation = data.site_xmat[self._site_id].reshape(3, 3).copy().tolist()
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
    ) -> list[float] | None:
        import mujoco

        target = np.asarray(target_xyz, dtype=np.float64)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            return None
        data = self.session.data
        model = self.session.model
        old_qpos = data.qpos.copy()
        old_qvel = data.qvel.copy()
        old_ctrl = data.ctrl.copy()
        seed = (
            list(current_joints)
            if current_joints is not None
            else self.get_joint_positions()
        )
        if len(seed) != self.dof:
            return None
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
                mujoco.mj_forward(model, data)
                error = target - data.site_xpos[self._site_id]
                if float(np.linalg.norm(error)) < IK_TOL:
                    result = [float(data.qpos[address]) for address in qpos_addresses]
                    break
                jacobian = np.zeros((3, model.nv), dtype=np.float64)
                mujoco.mj_jacSite(model, data, jacobian, None, self._site_id)
                arm_jacobian = jacobian[:, dof_addresses]
                normal = arm_jacobian @ arm_jacobian.T + IK_DAMPING * np.eye(3)
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
            mujoco.mj_forward(model, data)

    def ee_position(self) -> np.ndarray:
        return np.asarray(
            self.session.data.site_xpos[self._site_id], dtype=np.float64
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
            if model.jnt_type[joint_start] == mujoco.mjtJoint.mjJNT_FREE:
                result[str(name)] = [float(value) for value in data.xpos[body_id]]
        return result

    def diagnostics(self) -> dict[str, Any]:
        return {
            "dof": self.dof,
            "joint_names": list(self.joint_names),
            "joint_ids": list(self._joint_ids),
            "actuator_ids": list(self._actuator_ids),
            "ee_site_id": self._site_id,
        }
