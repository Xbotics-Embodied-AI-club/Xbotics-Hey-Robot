from __future__ import annotations

from hey_robot.config import RobotSpec
from hey_robot.robot_runtime.classic.primitives import SUPPORTED_CLASSIC_PRIMITIVES

SO101_TABLETOP_PRIMITIVES = (
    "arm_get_state",
    "arm_solve_position_ik",
    "move_arm_joints",
    "set_gripper",
    "sim_locate_object",
    "sim_get_object_state",
    "reset_posture",
    "set_arm_pose",
    "stop_motion",
    "inspect_scene",
)

SO101_MOBILE_PRIMITIVES = (
    "arm_get_state",
    "arm_solve_position_ik",
    "move_arm_joints",
    "set_gripper",
    "sim_locate_object",
    "sim_get_object_state",
    "reset_posture",
    "set_arm_pose",
    "stop_motion",
    "inspect_scene",
    "get_camera_geometry",
    "perceive_grasp_point",
)

XLEROBOT_SIM_PRIMITIVES = tuple(
    dict.fromkeys((*SUPPORTED_CLASSIC_PRIMITIVES, *SO101_MOBILE_PRIMITIVES))
)


def supported_driver_primitives(robot: RobotSpec) -> tuple[str, ...]:
    """Return canonical Skill primitive names supported by a deployment robot."""

    configured = robot.settings.get("supported_driver_primitives")
    if configured:
        return tuple(str(item) for item in configured)

    if robot.robot_family == "xlerobot" and robot.driver_kind == "mujoco":
        return XLEROBOT_SIM_PRIMITIVES

    if robot.robot_family == "xlerobot" and robot.driver_kind in {"mock", "native"}:
        return SUPPORTED_CLASSIC_PRIMITIVES

    if robot.robot_family == "so101" and robot.driver_kind == "mujoco":
        return SO101_TABLETOP_PRIMITIVES

    if robot.robot_family == "so101_mobile" and robot.driver_kind == "mujoco":
        return SO101_MOBILE_PRIMITIVES

    return ()
