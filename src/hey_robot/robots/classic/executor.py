from __future__ import annotations

from typing import Protocol

from hey_robot.robots.classic.primitives import (
    BaseVelocityStepPrimitive,
    MoveArmJointsPrimitive,
    MoveBasePrimitive,
    PerceptionPrimitive,
    ResetPosturePrimitive,
    SetArmPosePrimitive,
    SetGripperPrimitive,
    StopMotionPrimitive,
    TurnBasePrimitive,
    decode_classic_primitive,
)
from hey_robot.skills import RobotSkillAction


class ClassicPrimitiveBackend[TResult](Protocol):
    def on_stop_motion(
        self, primitive: StopMotionPrimitive, *, skill_name: str
    ) -> TResult: ...
    def on_move_base(
        self, primitive: MoveBasePrimitive, *, skill_name: str
    ) -> TResult: ...
    def on_turn_base(
        self, primitive: TurnBasePrimitive, *, skill_name: str
    ) -> TResult: ...
    def on_base_velocity_step(
        self, primitive: BaseVelocityStepPrimitive, *, skill_name: str
    ) -> TResult: ...
    def on_set_arm_pose(
        self, primitive: SetArmPosePrimitive, *, skill_name: str
    ) -> TResult: ...
    def on_move_arm_joints(
        self, primitive: MoveArmJointsPrimitive, *, skill_name: str
    ) -> TResult: ...
    def on_set_gripper(
        self, primitive: SetGripperPrimitive, *, skill_name: str
    ) -> TResult: ...
    def on_reset_posture(
        self, primitive: ResetPosturePrimitive, *, skill_name: str
    ) -> TResult: ...
    def on_perception(
        self, primitive: PerceptionPrimitive, *, skill_name: str
    ) -> TResult: ...


class ClassicSkillExecutor[TResult]:
    def __init__(self, backend: ClassicPrimitiveBackend[TResult]) -> None:
        self.backend = backend

    def execute(self, action: RobotSkillAction) -> TResult:
        primitive = decode_classic_primitive(action)
        skill_name = action.name

        if isinstance(primitive, StopMotionPrimitive):
            return self.backend.on_stop_motion(primitive, skill_name=skill_name)
        if isinstance(primitive, MoveBasePrimitive):
            return self.backend.on_move_base(primitive, skill_name=skill_name)
        if isinstance(primitive, TurnBasePrimitive):
            return self.backend.on_turn_base(primitive, skill_name=skill_name)
        if isinstance(primitive, BaseVelocityStepPrimitive):
            return self.backend.on_base_velocity_step(primitive, skill_name=skill_name)
        if isinstance(primitive, SetArmPosePrimitive):
            return self.backend.on_set_arm_pose(primitive, skill_name=skill_name)
        if isinstance(primitive, MoveArmJointsPrimitive):
            return self.backend.on_move_arm_joints(primitive, skill_name=skill_name)
        if isinstance(primitive, SetGripperPrimitive):
            return self.backend.on_set_gripper(primitive, skill_name=skill_name)
        if isinstance(primitive, ResetPosturePrimitive):
            return self.backend.on_reset_posture(primitive, skill_name=skill_name)
        if isinstance(primitive, PerceptionPrimitive):
            return self.backend.on_perception(primitive, skill_name=skill_name)
        raise ValueError(f"unsupported classic primitive for skill: {skill_name}")
