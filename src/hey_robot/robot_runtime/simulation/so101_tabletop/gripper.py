# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
# Modified for Xbotics Hey Robot.
from __future__ import annotations

import time

import numpy as np

from hey_robot.robot_runtime.simulation.so101_tabletop.arm import So101ArmKernel
from hey_robot.robot_runtime.simulation.so101_tabletop.session import (
    So101TabletopSession,
)

JAW_OPEN = 0.8
JAW_CLOSED = -0.17
JAW_ACTUATOR_NAME = "act_jaw_visual"
GRASP_RADIUS = 0.05
SETTLE_STEPS = 400


class WeldGripperKernel:
    def __init__(self, session: So101TabletopSession, arm: So101ArmKernel) -> None:
        self.session = session
        self.arm = arm
        self._actuator_id = -1
        self._held_object: str | None = None
        self._is_open = True

    def bind(self) -> None:
        import mujoco

        self._actuator_id = self.session.id_for(
            mujoco.mjtObj.mjOBJ_ACTUATOR, JAW_ACTUATOR_NAME
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
        start = float(data.ctrl[self._actuator_id])
        dt = float(self.session.model.opt.timestep)
        sync_interval = max(1, int(1.0 / 60.0 / dt))
        wall_start = time.monotonic()
        for index in range(SETTLE_STEPS):
            alpha = (index + 1) / SETTLE_STEPS
            data.ctrl[self._actuator_id] = start + alpha * (target - start)
            mujoco.mj_step(model, data)
            if self.session.viewer is not None and index % sync_interval == 0:
                self.session.viewer.sync()
                elapsed_sim = (index + 1) * dt
                remaining = elapsed_sim - (time.monotonic() - wall_start)
                if remaining > 0:
                    time.sleep(remaining)

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
        for equality_id in range(model.neq):
            if model.eq_type[equality_id] != mujoco.mjtEq.mjEQ_WELD:
                continue
            body2_id = int(model.eq_obj2id[equality_id])
            body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
            if body2_name != nearest_name:
                continue
            body1_id = int(model.eq_obj1id[equality_id])
            position1 = data.xpos[body1_id]
            rotation1 = data.xmat[body1_id].reshape(3, 3)
            position2 = data.xpos[body2_id]
            rotation2 = data.xmat[body2_id].reshape(3, 3)
            relative_position = rotation1.T @ (position2 - position1)
            relative_rotation = rotation1.T @ rotation2
            relative_quaternion = np.zeros(4, dtype=float)
            mujoco.mju_mat2Quat(relative_quaternion, relative_rotation.reshape(-1))
            model.eq_data[equality_id, :3] = 0.0
            model.eq_data[equality_id, 3:6] = relative_position
            model.eq_data[equality_id, 6:10] = relative_quaternion
            data.eq_active[equality_id] = 1
            self._held_object = nearest_name
            for _ in range(50):
                mujoco.mj_step(model, data)
            if self.session.viewer is not None:
                self.session.viewer.sync()
            return

    def _release_all(self) -> None:
        import mujoco

        for equality_id in range(self.session.model.neq):
            if self.session.model.eq_type[equality_id] == mujoco.mjtEq.mjEQ_WELD:
                self.session.data.eq_active[equality_id] = 0
        self._held_object = None

    def _sync_held_from_constraints(self) -> None:
        active = [name for name, enabled in self.weld_states().items() if enabled]
        self._held_object = active[0] if active else None
