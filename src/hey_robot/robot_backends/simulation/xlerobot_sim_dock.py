"""Optional home-scene dock manipulation.

The generic simulation driver knows only that this component may consume a skill.
All wand, weld and dock-specific state lives here.
"""

from __future__ import annotations

import math
import time
from typing import Any, Protocol

import numpy as np

from hey_robot.protocol import RobotSkillAction, RobotSkillResult


class SimDockHost(Protocol):
    model: Any
    data: Any
    adapter: Any
    _viewer: Any

    def _joint_position_for_actuator(self, actuator_idx: int) -> float: ...

    def _sync_viewer(self) -> None: ...


class DockSession:
    """Small session surface required by the existing arm/gripper kernels."""

    def __init__(self, host: SimDockHost) -> None:
        self.host = host
        self.model = host.model
        self.data = host.data
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
        return self.host._viewer

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
        import mujoco

        self._base_qpos.clear()
        self._base_dofs.clear()
        for name in ("root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint"):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_addr = int(self.model.jnt_qposadr[joint_id])
            self._base_qpos[qpos_addr] = float(self.data.qpos[qpos_addr])
            self._base_dofs.append(int(self.model.jnt_dofadr[joint_id]))
        self._right_arm_qpos.clear()
        self._right_arm_dofs.clear()
        for actuator_idx in self.host.adapter.arm_actuator_indices("right"):
            joint_id = int(self.model.actuator_trnid[actuator_idx][0])
            qpos_addr = int(self.model.jnt_qposadr[joint_id])
            self._right_arm_qpos[qpos_addr] = float(self.data.qpos[qpos_addr])
            self._right_arm_dofs.append(int(self.model.jnt_dofadr[joint_id]))

    def clamp_base(self) -> None:
        for qpos_addr, value in self._base_qpos.items():
            self.data.qpos[qpos_addr] = value
        for dof_addr in self._base_dofs:
            self.data.qvel[dof_addr] = 0.0
        for qpos_addr, value in self._right_arm_qpos.items():
            self.data.qpos[qpos_addr] = value
        for dof_addr in self._right_arm_dofs:
            self.data.qvel[dof_addr] = 0.0


class DockManipulation:
    """Consumes only dock-specific skills; returns ``None`` for generic skills."""

    def __init__(self, host: SimDockHost) -> None:
        self.host = host
        self.session: DockSession | None = None
        self.arm: Any = None
        self.gripper: Any = None
        self.arm_side = "left"
        self.active = False

    @property
    def model(self) -> Any:
        return self.host.model

    @property
    def data(self) -> Any:
        return self.host.data

    def initialize(self) -> None:
        import mujoco

        if (
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wand") < 0
            or mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grip_weld")
            < 0
        ):
            return
        from hey_robot.robot_backends.simulation.dock_manipulation.arm import (
            So101MobileArmKernel,
        )
        from hey_robot.robot_backends.simulation.dock_manipulation.gripper import (
            WandGripperKernel,
        )

        self.session = DockSession(self.host)
        physical_left = (
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw_tip_2")
            >= 0
        )
        if physical_left:
            self.arm_side = "right"
            self.arm = So101MobileArmKernel(
                self.session,
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
            self.arm = So101MobileArmKernel(self.session)
            jaw_actuator_name = "Jaw_L"
        self.arm.bind()
        self.gripper = WandGripperKernel(
            self.session, self.arm, jaw_actuator_name=jaw_actuator_name
        )
        self.gripper.bind()

    def execute(self, skill: RobotSkillAction) -> RobotSkillResult | None:
        if self.session is None:
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
        self.session.lock_base()
        if name == "arm_get_state":
            return RobotSkillResult(
                True,
                "left arm state read",
                {"joint_positions": self.arm.get_joint_positions()},
            )
        if name == "arm_solve_position_ik":
            self.active = True
            target_local = self._xyz(skill.arguments.get("target_xyz"))
            axis_local = (
                self._xyz(skill.arguments["target_axis"])
                if skill.arguments.get("target_axis") is not None
                else None
            )
            seed_value = skill.arguments.get("current_joints")
            seed = (
                [float(value) for value in seed_value]
                if isinstance(seed_value, (list, tuple))
                and len(seed_value) == self.arm.dof
                else None
            )
            solution = self.arm.ik(
                self._base_to_world(target_local),
                seed,
                target_axis=self._base_vector_to_world(axis_local)
                if axis_local
                else None,
            )
            return RobotSkillResult(
                True,
                "left-arm IK solved" if solution is not None else "IK unreachable",
                {
                    "operation_success": solution is not None,
                    "failure_mode": None if solution is not None else "ik_unreachable",
                    "target_xyz": list(target_local),
                    "target_axis": list(axis_local) if axis_local else None,
                    "joint_positions": solution,
                },
            )
        if name == "move_arm_joints":
            self.active = True
            joints = dict(skill.arguments.get("joints") or {})
            positions = [
                float(joints[key])
                for key in ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")
            ]
            self._move_arm(positions, float(skill.arguments.get("duration", 3.0)))
            return RobotSkillResult(
                True,
                "left arm joints moved",
                {"joint_positions": self.arm.get_joint_positions()},
            )
        if name == "set_gripper":
            if not self.active:
                return None
            command = str(skill.arguments.get("action") or "").lower()
            opened = command == "open" or (
                command not in {"open", "close"}
                and float(skill.arguments.get("opening_pct", 0.0)) >= 50.0
            )
            self._set_gripper(opened=opened)
            return RobotSkillResult(
                True,
                f"left gripper {command or 'set'}",
                {
                    "held_object": self.gripper.held_object,
                    "welds": self.gripper.weld_states(),
                },
            )
        if name == "sim_locate_object":
            query = str(skill.arguments.get("query") or "").strip().lower()
            if query not in {"wand", "棒", "玩具棒", "toy"}:
                return RobotSkillResult(
                    True,
                    f"object not found: {query}",
                    {"operation_success": False, "failure_mode": "object_not_found"},
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
                "held_object": self.gripper.held_object,
                "welds": self.gripper.weld_states(),
            },
        )

    def _move_arm(self, positions: list[float], duration: float) -> None:
        import mujoco

        assert self.session is not None
        actuator_ids = self.host.adapter.arm_actuator_indices(self.arm_side)[:5]
        if len(positions) != len(actuator_ids):
            raise ValueError("left arm requires five joint positions")
        starts = [
            self.host._joint_position_for_actuator(actuator_id)
            for actuator_id in actuator_ids
        ]
        steps = max(2, int(max(0.0, duration) * 60.0))
        frame_period = duration / steps if self.host._viewer is not None else 0.0
        for step in range(steps):
            alpha = (step + 1) / steps
            self.session.clamp_base()
            for actuator_id, start, target in zip(
                actuator_ids, starts, positions, strict=True
            ):
                value = start + alpha * (float(target) - start)
                joint_id = int(self.model.actuator_trnid[actuator_id][0])
                qpos_addr = int(self.model.jnt_qposadr[joint_id])
                dof_addr = int(self.model.jnt_dofadr[joint_id])
                self.data.ctrl[actuator_id] = value
                self.data.qpos[qpos_addr] = value
                self.data.qvel[dof_addr] = 0.0
            mujoco.mj_kinematics(self.model, self.data)
            self._sync_active_welds()
            self.host._sync_viewer()
            if frame_period > 0.0:
                time.sleep(frame_period)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        self._sync_active_welds()

    def _set_gripper(self, *, opened: bool) -> None:
        import mujoco

        from hey_robot.robot_backends.simulation.dock_manipulation.gripper import (
            JAW_CLOSED,
            JAW_OPEN,
        )

        if opened:
            self.gripper._release_all()
        actuator_id = int(self.gripper._actuator_id)
        joint_id = int(self.model.actuator_trnid[actuator_id][0])
        qpos_addr = int(self.model.jnt_qposadr[joint_id])
        dof_addr = int(self.model.jnt_dofadr[joint_id])
        target = JAW_OPEN if opened else JAW_CLOSED
        self.data.ctrl[actuator_id] = target
        self.data.qpos[qpos_addr] = target
        self.data.qvel[dof_addr] = 0.0
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        self.gripper._is_open = opened
        if opened:
            self.gripper._held_object = None
            self.active = False
            self._sync_active_welds()
        else:
            self._activate_grip_weld()

    def _activate_grip_weld(self) -> None:
        import mujoco

        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "wand_grasp")
        if (
            float(np.linalg.norm(self.arm.ee_position() - self.data.site_xpos[site_id]))
            >= 0.06
        ):
            self.gripper._held_object = None
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
        rotation1 = self.data.xmat[body1_id].reshape(3, 3)
        relative_position = rotation1.T @ (
            self.data.xpos[body2_id] - self.data.xpos[body1_id]
        )
        relative_rotation = rotation1.T @ self.data.xmat[body2_id].reshape(3, 3)
        relative_quaternion = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(relative_quaternion, relative_rotation.reshape(-1))
        self.model.eq_data[grip_id, :3] = 0.0
        self.model.eq_data[grip_id, 3:6] = relative_position
        self.model.eq_data[grip_id, 6:10] = relative_quaternion
        self.gripper._held_object = "wand"
        self._sync_active_welds()

    def _sync_active_welds(self) -> None:
        import mujoco

        for name in ("dock_weld", "grip_weld"):
            equality_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_EQUALITY, name
            )
            if equality_id < 0 or not bool(self.data.eq_active[equality_id]):
                continue
            body1_id = int(self.model.eq_obj1id[equality_id])
            body2_id = int(self.model.eq_obj2id[equality_id])
            joint_id = int(self.model.body_jntadr[body2_id])
            if (
                joint_id < 0
                or self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
            ):
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
                relative_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
            rotation1 = self.data.xmat[body1_id].reshape(3, 3)
            quaternion = np.zeros(4, dtype=float)
            mujoco.mju_mulQuat(
                quaternion,
                np.asarray(self.data.xquat[body1_id], dtype=float),
                relative_quaternion,
            )
            self.data.qpos[qpos_addr : qpos_addr + 3] = (
                self.data.xpos[body1_id] + rotation1 @ relative_position
            )
            self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = quaternion
            self.data.qvel[dof_addr : dof_addr + 6] = 0.0
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)

    @staticmethod
    def _xyz(value: Any) -> tuple[float, float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError("target_xyz must contain three numbers")
        return float(value[0]), float(value[1]), float(value[2])

    def _base_to_world(
        self, point: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        import mujoco

        root_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "root")
        root, yaw = self.data.xpos[root_id], float(self.data.qpos[2])
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
        delta, yaw = (
            np.asarray(point, dtype=float) - self.data.xpos[root_id],
            float(self.data.qpos[2]),
        )
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

        return self._world_to_base(
            self.data.xpos[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            ]
        )

    def _wand_grasp_position_base(self) -> tuple[float, float, float]:
        import mujoco

        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "wand_grasp")
        return self._world_to_base(self.data.site_xpos[site_id])

    def _wand_grasp_axis_base(self) -> tuple[float, float, float]:
        import mujoco

        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "wand_grasp")
        ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "wand_ball")
        axis = np.asarray(self.data.geom_xpos[ball_id]) - np.asarray(
            self.data.site_xpos[site_id]
        )
        norm = float(np.linalg.norm(axis))
        return (
            (1.0, 0.0, 0.0) if norm <= 0.0 else self._world_vector_to_base(axis / norm)
        )

    def _dock_target_base(self) -> tuple[float, float, float]:
        dock = self._body_position_base("wand_dock")
        return dock[0], dock[1], dock[2] + 0.255
