from __future__ import annotations

import math

import pytest

from hey_robot.robot_runtime.simulation.so101_tabletop.arm import So101ArmKernel
from hey_robot.robot_runtime.simulation.so101_tabletop.session import (
    So101TabletopSession,
)


@pytest.fixture
def arm() -> So101ArmKernel:
    session = So101TabletopSession()
    session.connect()
    kernel = So101ArmKernel(session)
    kernel.bind()
    yield kernel
    session.close()


def test_arm_contract(arm: So101ArmKernel) -> None:
    assert arm.dof == 5
    assert arm.joint_names == (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    )
    assert len(arm.get_joint_positions()) == 5


def test_fk_does_not_mutate_live_state(arm: So101ArmKernel) -> None:
    before = arm.get_joint_positions()
    position, rotation = arm.fk([0.5] * 5)
    assert len(position) == 3
    assert len(rotation) == 3
    assert arm.get_joint_positions() == pytest.approx(before, abs=1e-9)


def test_position_ik_accuracy(arm: So101ArmKernel) -> None:
    target = (0.22, 0.05, 0.08)
    solution = arm.ik(target)
    assert solution is not None
    position, _ = arm.fk(solution)
    error = math.sqrt(sum((a - b) ** 2 for a, b in zip(position, target, strict=True)))
    assert error < 0.003


def test_position_ik_rejects_unreachable_target(arm: So101ArmKernel) -> None:
    assert arm.ik((5.0, 5.0, 5.0)) is None


def test_joint_interpolation_tracks_target(arm: So101ArmKernel) -> None:
    target = [0.3, -0.3, 0.2, 0.1, -0.1]
    assert arm.move_joints(target, duration=4.0)
    assert arm.get_joint_positions() == pytest.approx(target, abs=0.05)
