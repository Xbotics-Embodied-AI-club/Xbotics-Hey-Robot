from __future__ import annotations

import pytest

from hey_robot.robot_runtime.simulation.so101_tabletop.arm import So101ArmKernel
from hey_robot.robot_runtime.simulation.so101_tabletop.gripper import (
    WeldGripperKernel,
)
from hey_robot.robot_runtime.simulation.so101_tabletop.session import (
    So101TabletopSession,
)


@pytest.fixture
def stack() -> tuple[So101TabletopSession, So101ArmKernel, WeldGripperKernel]:
    session = So101TabletopSession()
    session.connect()
    arm = So101ArmKernel(session)
    arm.bind()
    gripper = WeldGripperKernel(session, arm)
    gripper.bind()
    session.step(200)
    yield session, arm, gripper
    session.close()


def test_close_near_target_activates_target_weld(stack) -> None:
    _session, arm, gripper = stack
    position = arm.get_object_positions()["banana"]
    solution = arm.ik((position[0], position[1], position[2] + 0.02))
    assert solution is not None
    arm.move_joints(solution, duration=2.0)
    gripper.close()
    assert gripper.held_object == "banana"
    assert gripper.weld_states()["banana"] is True


def test_close_far_from_objects_does_not_hold(stack) -> None:
    _session, _arm, gripper = stack
    gripper.close()
    assert gripper.held_object is None


def test_open_releases_held_object(stack) -> None:
    _session, arm, gripper = stack
    position = arm.get_object_positions()["banana"]
    solution = arm.ik((position[0], position[1], position[2] + 0.02))
    assert solution is not None
    arm.move_joints(solution, duration=2.0)
    gripper.close()
    assert gripper.is_holding("banana")
    gripper.open()
    assert gripper.held_object is None
    assert not any(gripper.weld_states().values())
