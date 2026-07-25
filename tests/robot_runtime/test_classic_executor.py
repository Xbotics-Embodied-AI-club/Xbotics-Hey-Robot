from __future__ import annotations

from hey_robot.protocol import RobotSkillAction
from hey_robot.robot_runtime.classic_executor import (
    ClassicPrimitiveBackend,
    ClassicSkillExecutor,
)


def test_classic_skill_executor_dispatches_to_backend() -> None:
    class FakeBackend(ClassicPrimitiveBackend[str]):
        def on_stop_motion(self, primitive, *, skill_name):
            return f"{skill_name}:{primitive.emergency}"

        def on_move_base(self, _primitive, *, _skill_name):
            return "move"

        def on_turn_base(self, _primitive, *, _skill_name):
            return "turn"

        def on_base_velocity_step(self, _primitive, *, _skill_name):
            return "velocity"

        def on_set_arm_pose(self, _primitive, *, _skill_name):
            return "pose"

        def on_move_arm_joints(self, _primitive, *, _skill_name):
            return "joints"

        def on_set_gripper(self, _primitive, *, _skill_name):
            return "gripper"

        def on_reset_posture(self, _primitive, *, _skill_name):
            return "reset"

        def on_perception(self, _primitive, *, _skill_name):
            return "perception"

    executor = ClassicSkillExecutor(FakeBackend())

    result = executor.execute(RobotSkillAction("stop_motion", {"emergency": True}))

    assert result == "stop_motion:True"
