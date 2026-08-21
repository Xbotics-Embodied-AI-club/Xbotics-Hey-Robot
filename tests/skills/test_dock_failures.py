from __future__ import annotations

from hey_robot.skills import SkillContext
from hey_robot.skills.builtins import dock


async def test_dock_skills_report_precondition_failures_without_robot_actions() -> None:
    context = SkillContext("run", "task", "robot")

    unavailable = await dock._call(context, "reset_posture", {})
    invalid_joints = await dock._move_joints(context, [1.0], 1.0)
    wrong_mode = await dock.pick_wand_from_dock(context, {"mode": "perception"})

    assert unavailable.failure_mode == "robot_client_unavailable"
    assert invalid_joints is not None
    assert invalid_joints.failure_mode == "ik_unreachable"
    assert wrong_mode.failure_mode == "mode_not_migrated"


class _StateRobot:
    async def execute(self, _robot_id, action, _arguments, *, run_id):
        del run_id
        if action == "sim_get_object_state":
            from hey_robot.robot_api import RobotActionResult

            return RobotActionResult(True, "state", data={"held_object": None})
        raise AssertionError(f"unexpected action {action}")


async def test_place_dock_stops_when_the_wand_is_not_held() -> None:
    result = await dock.place_wand_to_dock(
        SkillContext("run", "task", "robot", robot=_StateRobot()),  # type: ignore[arg-type]
        {},
    )

    assert result.failure_mode == "gripper_empty"


class _DockRobot:
    def __init__(self) -> None:
        self.held = False

    async def execute(self, _robot_id, action, arguments, *, run_id):
        del run_id
        from hey_robot.robot_api import RobotActionResult

        if action == "sim_locate_object":
            data = {
                "operation_success": True,
                "samples": [[0.1, 0.2, 0.3]],
                "grasp_axis": [0, 0, 1],
            }
        elif action == "arm_solve_position_ik":
            data = {"operation_success": True, "joint_positions": [0.0] * 5}
        elif action == "set_gripper":
            self.held = arguments["action"] == "close"
            data = {}
        elif action == "sim_get_object_state":
            data = {
                "held_object": "wand" if self.held else None,
                "welds": {"wand": self.held},
                "dock_target": [0.1, 0.2, 0.3],
            }
        else:
            data = {}
        return RobotActionResult(True, action, data=data)


async def test_dock_pick_and_place_complete_against_native_primitives() -> None:
    robot = _DockRobot()
    context = SkillContext("run", "task", "robot", robot=robot)  # type: ignore[arg-type]

    picked = await dock.pick_wand_from_dock(context, {})
    placed = await dock.place_wand_to_dock(context, {})

    assert picked.success
    assert picked.data["weld_active"]
    assert placed.success
    assert placed.data == {"released": True}
