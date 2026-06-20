from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

from hey_robot.robots.classic import (
    BaseVelocityStepPrimitive,
    ClassicEmbodimentProfile,
    ClassicPrimitiveBackend,
    ClassicSkillExecutor,
    MoveArmJointsPrimitive,
    MoveBasePrimitive,
    PerceptionPrimitive,
    ResetPosturePrimitive,
    SetArmPosePrimitive,
    SetGripperPrimitive,
    StopMotionPrimitive,
    TurnBasePrimitive,
)
from hey_robot.robots.embodiments import EmbodimentProfile
from hey_robot.skills import RobotSkillAction


@dataclass
class SimSkillCommand:
    skill_name: str
    vx: float = 0.0
    vy: float = 0.0
    vw: float = 0.0
    arm_targets: dict[int, float] = field(default_factory=dict)
    delta_mode: bool = False
    duration_sec: float = 0.0
    jaw_left: float | None = None
    jaw_right: float | None = None
    message: str = ""


_JOINT_TO_ACTUATORS: dict[str, tuple[int, int]] = {
    "shoulder_pan": (9, 3),
    "shoulder_lift": (10, 4),
    "elbow_flex": (11, 5),
    "wrist_flex": (12, 6),
    "wrist_roll": (13, 7),
    "gripper": (14, 8),
}

# Canonical joint order for VLA observation/action vectors.
_ARM_JOINT_ORDER: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# Per-side actuator index lists derived from xlerobot.xml actuator order.
# Right arm: Rotation_R(3), Pitch_R(4), Elbow_R(5), Wrist_Pitch_R(6), Wrist_Roll_R(7), Jaw_R(8)
# Left arm:  Rotation_L(9), Pitch_L(10), Elbow_L(11), Wrist_Pitch_L(12), Wrist_Roll_L(13), Jaw_L(14)
_ARM_ACTUATOR_INDICES: dict[str, list[int]] = {
    "right": [3, 4, 5, 6, 7, 8],
    "left": [9, 10, 11, 12, 13, 14],
}

_DEFAULT_SIM_NAMED_POSES: dict[str, dict[str, float]] = {
    "home": {
        "shoulder_pan": 0.0,
        "shoulder_lift": 0.8,
        "elbow_flex": 0.7,
        "wrist_flex": -0.6,
        "wrist_roll": 0.0,
        "gripper": 0.08,
    },
    "pregrasp": {
        "shoulder_pan": 0.0,
        "shoulder_lift": 0.7,
        "elbow_flex": 0.9,
        "wrist_flex": -0.5,
        "wrist_roll": 0.0,
        "gripper": 0.08,
    },
    "pregrasp_table": {
        "shoulder_pan": 0.0,
        "shoulder_lift": 0.9,
        "elbow_flex": 1.05,
        "wrist_flex": -0.75,
        "wrist_roll": 0.0,
        "gripper": 0.08,
    },
    "place": {
        "shoulder_pan": 0.17,
        "shoulder_lift": 0.65,
        "elbow_flex": 0.6,
        "wrist_flex": -0.45,
        "wrist_roll": 0.0,
        "gripper": 0.07,
    },
}


class _XLeRobotSimBackend(ClassicPrimitiveBackend[SimSkillCommand]):
    def __init__(
        self,
        linear_speed: float = 0.2,
        angular_speed: float = 0.45,
        *,
        embodiment: EmbodimentProfile | None = None,
    ) -> None:
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.embodiment = ClassicEmbodimentProfile.from_embodiment(embodiment)

    def joint_to_actuators(self, joint_name: str) -> tuple[int, int] | None:
        if self.embodiment is not None:
            pair = self.embodiment.joint_actuator_pair(joint_name)
            if pair is not None:
                return pair
        return _JOINT_TO_ACTUATORS.get(joint_name)

    @property
    def gripper_open_value(self) -> float:
        if self.embodiment is not None:
            return self.embodiment.gripper_open_value
        return 1.2

    @property
    def gripper_closed_value(self) -> float:
        if self.embodiment is not None:
            return self.embodiment.gripper_closed_value
        return 0.0

    def arm_actuator_indices(self, arm: str) -> list[int]:
        """Return the 6 actuator indices for a given arm side.

        If the embodiment defines a custom joint_actuator_pair mapping,
        derive the indices from it. Otherwise use the default mapping
        from xlerobot.xml actuator order.
        """
        arm_key = arm.strip().lower()
        if arm_key not in {"left", "right"}:
            raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
        if self.embodiment is not None:
            with contextlib.suppress(Exception):
                return [
                    self.embodiment.joint_actuator_pair(name)[  # type: ignore[index]
                        0 if arm_key == "left" else 1
                    ]
                    for name in _ARM_JOINT_ORDER
                ]
        return list(_ARM_ACTUATOR_INDICES[arm_key])

    def arm_joint_order(self) -> tuple[str, ...]:
        """Return the canonical joint name order for VLA vectors."""
        return _ARM_JOINT_ORDER

    def on_stop_motion(
        self, primitive: StopMotionPrimitive, *, skill_name: str
    ) -> SimSkillCommand:
        return SimSkillCommand(
            skill_name=skill_name,
            message="emergency stop active" if primitive.emergency else "base stopped",
        )

    def on_move_base(
        self, primitive: MoveBasePrimitive, *, skill_name: str
    ) -> SimSkillCommand:
        distance = primitive.distance_cm / 100.0
        direction = primitive.direction.strip().lower()
        duration = abs(distance) / max(self.linear_speed, 0.01)
        if direction in {"left", "right"}:
            sign = -1.0 if direction == "left" else 1.0
            return SimSkillCommand(
                skill_name=skill_name,
                vx=sign * self.linear_speed,
                duration_sec=duration,
                message=f"base moved {direction} {abs(distance) * 100:.1f}cm",
            )
        sign = -1.0 if direction == "backward" else 1.0
        # vx→world Y, vy→world X. Table is on X, so use vy for forward.
        return SimSkillCommand(
            skill_name=skill_name,
            vy=sign * self.linear_speed,
            duration_sec=duration,
            message=f"base moved {direction} {abs(distance) * 100:.1f}cm",
        )

    def on_turn_base(
        self, primitive: TurnBasePrimitive, *, skill_name: str
    ) -> SimSkillCommand:
        angle_rad = abs(primitive.angle_deg) * 3.14159 / 180.0
        sign = -1.0 if primitive.direction == "right" else 1.0
        duration = angle_rad / max(self.angular_speed, 0.01)
        return SimSkillCommand(
            skill_name=skill_name,
            vw=sign * self.angular_speed,
            duration_sec=duration,
            message=f"base turned {primitive.direction} {abs(primitive.angle_deg):.1f}deg",
        )

    def on_base_velocity_step(
        self, primitive: BaseVelocityStepPrimitive, *, skill_name: str
    ) -> SimSkillCommand:
        return SimSkillCommand(
            skill_name=skill_name,
            vy=primitive.vx,
            vx=primitive.vy,
            vw=primitive.wz,
            duration_sec=primitive.duration_ms / 1000.0,
            message="base velocity step completed",
        )

    def on_set_arm_pose(
        self, primitive: SetArmPosePrimitive, *, skill_name: str
    ) -> SimSkillCommand:
        _ = skill_name
        return self._pose_command(primitive.pose_name)

    def on_move_arm_joints(
        self, primitive: MoveArmJointsPrimitive, *, skill_name: str
    ) -> SimSkillCommand:
        arm_targets: dict[int, float] = {}
        for joint_name, value in primitive.joints.items():
            indices = self.joint_to_actuators(str(joint_name))
            if indices is None:
                continue
            left_idx, right_idx = indices
            v = float(value)
            arm_targets[left_idx] = v
            arm_targets[right_idx] = v
        return SimSkillCommand(
            skill_name=skill_name,
            arm_targets=arm_targets,
            delta_mode=primitive.delta_mode,
            message=f"joints set ({'delta' if primitive.delta_mode else 'absolute'})",
        )

    def on_set_gripper(
        self, primitive: SetGripperPrimitive, *, skill_name: str
    ) -> SimSkillCommand:
        if primitive.action == "open":
            opening = self.gripper_open_value
        elif primitive.action == "close":
            opening = self.gripper_closed_value
        else:
            opening = (
                float(primitive.opening_pct or 0.0) / 100.0 * self.gripper_open_value
            )
        return SimSkillCommand(
            skill_name=skill_name,
            jaw_left=opening,
            jaw_right=opening,
            message=f"gripper set to {opening:.3f}rad",
        )

    def on_perception(
        self, primitive: PerceptionPrimitive, *, skill_name: str
    ) -> SimSkillCommand:
        _ = skill_name
        return SimSkillCommand(
            skill_name=primitive.skill_name,
            message=f"{primitive.skill_name} handled by perception service",
        )

    def on_reset_posture(
        self, primitive: ResetPosturePrimitive, *, skill_name: str
    ) -> SimSkillCommand:
        _ = primitive
        cmd = self._pose_command("home")
        return SimSkillCommand(
            skill_name=skill_name,
            arm_targets=cmd.arm_targets,
            message="robot reset posture",
        )

    def _pose_command(self, pose_name: str) -> SimSkillCommand:
        pose = self._named_pose(pose_name)
        if pose is None:
            raise ValueError(f"unknown named pose: {pose_name}")
        arm_targets: dict[int, float] = {}
        for joint_name, value in pose.items():
            indices = self.joint_to_actuators(joint_name)
            if indices is None:
                continue
            left_idx, right_idx = indices
            arm_targets[left_idx] = float(value)
            arm_targets[right_idx] = float(value)
        return SimSkillCommand(
            skill_name="set_arm_pose",
            arm_targets=arm_targets,
            message=f"arm moved to {pose_name}",
        )

    def arm_rest_positions(self) -> dict[int, float]:
        """Return the default rest (home) positions for all arm actuators."""
        targets: dict[int, float] = {}
        home = self._named_pose("home")
        if home is None:
            return targets
        for joint_name, value in home.items():
            indices = self.joint_to_actuators(joint_name)
            if indices is None:
                continue
            left_idx, right_idx = indices
            targets[left_idx] = float(value)
            targets[right_idx] = float(value)
        return targets

    def _named_pose(self, pose_name: str) -> dict[str, float] | None:
        if self.embodiment is not None:
            pose = self.embodiment.named_pose(pose_name)
            if pose is not None:
                return pose
        pose = _DEFAULT_SIM_NAMED_POSES.get(pose_name)
        if pose is None:
            return None
        return {str(joint): float(value) for joint, value in pose.items()}


class XLeRobotSimSkillAdapter:
    def __init__(
        self,
        linear_speed: float = 0.2,
        angular_speed: float = 0.45,
        *,
        embodiment: EmbodimentProfile | None = None,
    ) -> None:
        self._backend = _XLeRobotSimBackend(
            linear_speed=linear_speed,
            angular_speed=angular_speed,
            embodiment=embodiment,
        )
        self._executor = ClassicSkillExecutor(self._backend)

    def decode(self, action: RobotSkillAction) -> SimSkillCommand:
        return self._executor.execute(action)

    def arm_rest_positions(self) -> dict[int, float]:
        return self._backend.arm_rest_positions()

    def joint_to_actuators(self, joint_name: str) -> tuple[int, int] | None:
        return self._backend.joint_to_actuators(joint_name)

    def gripper_actuator_indices(self) -> tuple[int, int] | None:
        return self.joint_to_actuators("gripper")

    def arm_actuator_indices(self, arm: str) -> list[int]:
        return self._backend.arm_actuator_indices(arm)

    def arm_joint_order(self) -> tuple[str, ...]:
        return self._backend.arm_joint_order()

    @property
    def gripper_open_value(self) -> float:
        return self._backend.gripper_open_value
