# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
# Modified for Xbotics Hey Robot: shared dock gripper kernel.
from __future__ import annotations

import numpy as np

from hey_robot.robot_runtime.simulation.dock_manipulation.arm import (
    DockSession,
    So101MobileArmKernel,
)

JAW_OPEN = 1.7
JAW_CLOSED = 0.0
JAW_ACTUATOR_NAME = "Jaw_L"
GRASP_RADIUS = 0.06
SETTLE_STEPS = 400


class WandGripperKernel:
    """Weld-based gripper for XLeRobot right jaw (Jaw_R)."""

    def __init__(
        self,
        session: DockSession,
        arm: So101MobileArmKernel,
        *,
        jaw_actuator_name: str = JAW_ACTUATOR_NAME,
    ) -> None:
        self.session = session
        self.arm = arm
        self._jaw_actuator_name = jaw_actuator_name
        self._actuator_id = -1
        self._held_object: str | None = None
        self._is_open = True

    def bind(self) -> None:
        import mujoco

        self._actuator_id = self.session.id_for(
            mujoco.mjtObj.mjOBJ_ACTUATOR, self._jaw_actuator_name
        )
        self._sync_held_from_constraints()
        self._is_open = self._held_object is None

    @property
    def held_object(self) -> str | None:
        self._sync_held_from_constraints()
        return self._held_object

    def is_holding(self, object_name: str | None = None) -> bool:
        held = self.held_object
        return held is not None and (object_name is None or held == object_name)

    def get_position(self) -> float:
        return 1.0 if self._is_open else 0.0

    def open(self) -> bool:
        self._release_all()
        self._animate(JAW_OPEN)
        self._is_open = True
        self._held_object = None
        return True

    def close(self) -> bool:
        self._animate(JAW_CLOSED)
        self._is_open = False
        self._try_grasp()
        return True

    def weld_states(self) -> dict[str, bool]:
        import mujoco

        states: dict[str, bool] = {}
        model = self.session.model
        data = self.session.data
        for equality_id in range(model.neq):
            if model.eq_type[equality_id] != mujoco.mjtEq.mjEQ_WELD:
                continue
            body_id = int(model.eq_obj2id[equality_id])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name is not None:
                states[str(name)] = bool(data.eq_active[equality_id])
        return states

    def _animate(self, target: float) -> None:
        import mujoco

        data = self.session.data
        model = self.session.model
        # Freeze arm joints (position + velocity) so only the jaw moves.
        arm_qpos_addresses = [
            int(model.jnt_qposadr[jid]) for jid in self.arm._joint_ids
        ]
        arm_dof_addresses = [int(model.jnt_dofadr[jid]) for jid in self.arm._joint_ids]
        frozen_qpos = [float(data.qpos[adr]) for adr in arm_qpos_addresses]
        start = float(data.ctrl[self._actuator_id])
        for index in range(SETTLE_STEPS):
            alpha = (index + 1) / SETTLE_STEPS
            data.ctrl[self._actuator_id] = start + alpha * (target - start)
            self.session.clamp_base()
            # Freeze arm in place — kp=50 is too weak to hold against gravity
            for adr, value in zip(arm_qpos_addresses, frozen_qpos, strict=True):
                data.qpos[adr] = float(value)
            for adr in arm_dof_addresses:
                data.qvel[adr] = 0.0
            mujoco.mj_step(model, data)

    def _try_grasp(self) -> None:
        import mujoco

        ee_position = self.arm.ee_position()
        nearest_name: str | None = None
        nearest_distance = GRASP_RADIUS
        for name, position in self.arm.get_object_positions().items():
            distance = float(
                np.linalg.norm(np.asarray(position, dtype=float) - ee_position)
            )
            if distance < nearest_distance:
                nearest_name = name
                nearest_distance = distance
        if nearest_name is None:
            self._held_object = None
            return

        model = self.session.model
        data = self.session.data
        # Find the grip weld (body1=Fixed_Jaw_2 or gripper link, body2=target)
        grip_eq_id = -1
        dock_eq_id = -1
        for equality_id in range(model.neq):
            if model.eq_type[equality_id] != mujoco.mjtEq.mjEQ_WELD:
                continue
            body2_id = int(model.eq_obj2id[equality_id])
            body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
            if body2_name != nearest_name:
                continue
            body1_id = int(model.eq_obj1id[equality_id])
            body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            if body1_name == "wand_dock":
                dock_eq_id = equality_id
            else:
                grip_eq_id = equality_id
        if grip_eq_id < 0:
            return
        # Activate grip weld, deactivate dock weld
        data.eq_active[grip_eq_id] = 1
        if dock_eq_id >= 0:
            data.eq_active[dock_eq_id] = 0
        # Compute relative transform for grip weld
        body1_id = int(model.eq_obj1id[grip_eq_id])
        body2_id = int(model.eq_obj2id[grip_eq_id])
        position1 = data.xpos[body1_id]
        rotation1 = data.xmat[body1_id].reshape(3, 3)
        position2 = data.xpos[body2_id]
        rotation2 = data.xmat[body2_id].reshape(3, 3)
        relative_position = rotation1.T @ (position2 - position1)
        relative_rotation = rotation1.T @ rotation2
        relative_quaternion = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(relative_quaternion, relative_rotation.reshape(-1))
        model.eq_data[grip_eq_id, :3] = 0.0
        model.eq_data[grip_eq_id, 3:6] = relative_position
        model.eq_data[grip_eq_id, 6:10] = relative_quaternion
        self._held_object = nearest_name
        for _ in range(50):
            self.session.clamp_base()
            mujoco.mj_step(model, data)
        if self.session.viewer is not None:
            self.session.viewer.sync()

    def _release_all(self) -> None:
        import mujoco

        for equality_id in range(self.session.model.neq):
            if self.session.model.eq_type[equality_id] == mujoco.mjtEq.mjEQ_WELD:
                body1_id = int(self.session.model.eq_obj1id[equality_id])
                body1_name = mujoco.mj_id2name(
                    self.session.model, mujoco.mjtObj.mjOBJ_BODY, body1_id
                )
                self.session.data.eq_active[equality_id] = (
                    1 if body1_name == "wand_dock" else 0
                )
        self._held_object = None

    def _sync_held_from_constraints(self) -> None:
        active = [name for name, enabled in self.weld_states().items() if enabled]
        self._held_object = active[0] if active else None
