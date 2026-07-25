from __future__ import annotations

import pytest

from hey_robot.protocol import RobotSkillAction
from hey_robot.robot_api.primitives import (
    BaseVelocityStepPrimitive,
    MoveArmJointsPrimitive,
    MoveBasePrimitive,
    SetGripperPrimitive,
    decode_classic_primitive,
)


def test_decode_classic_primitive_normalizes_move_base() -> None:
    primitive = decode_classic_primitive(
        RobotSkillAction("move_base", {"direction": "FORWARD", "distance_cm": 12})
    )

    assert primitive == MoveBasePrimitive(direction="forward", distance_cm=12.0)


def test_decode_classic_primitive_supports_base_velocity_step() -> None:
    primitive = decode_classic_primitive(
        RobotSkillAction(
            "base_velocity_step",
            {"vx": 0.2, "vy": 0.0, "wz": 0.3, "duration_ms": 250},
        )
    )

    assert primitive == BaseVelocityStepPrimitive(
        vx=0.2, vy=0.0, wz=0.3, duration_ms=250
    )


def test_decode_classic_primitive_normalizes_joint_delta_payload() -> None:
    primitive = decode_classic_primitive(
        RobotSkillAction(
            "move_arm_joints",
            {"mode": "delta", "joints": {"wrist_roll": 12, "elbow_flex": -5}},
        )
    )

    assert primitive == MoveArmJointsPrimitive(
        joints={"wrist_roll": 12.0, "elbow_flex": -5.0},
        delta_mode=True,
    )


def test_decode_classic_primitive_normalizes_gripper_pct_path() -> None:
    primitive = decode_classic_primitive(
        RobotSkillAction("set_gripper", {"opening_pct": 35})
    )

    assert primitive == SetGripperPrimitive(action=None, opening_pct=35.0)


def test_decode_classic_primitive_rejects_missing_joint_dict() -> None:
    with pytest.raises(ValueError, match=r"arguments\.joints"):
        decode_classic_primitive(RobotSkillAction("move_arm_joints", {"mode": "delta"}))


def test_decode_classic_primitive_rejects_legacy_name() -> None:
    with pytest.raises(ValueError, match="unsupported classic primitive"):
        decode_classic_primitive(
            RobotSkillAction("base_move", {"direction": "forward", "distance_cm": 1})
        )
