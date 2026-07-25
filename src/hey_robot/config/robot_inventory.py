from __future__ import annotations

from hey_robot.config import RobotSpec
from hey_robot.robot_api.primitives import SUPPORTED_CLASSIC_PRIMITIVES

XLEROBOT_SIM_PRIMITIVES = tuple(
    dict.fromkeys(
        (
            *SUPPORTED_CLASSIC_PRIMITIVES,
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
    )
)


def supported_driver_primitives(robot: RobotSpec) -> tuple[str, ...]:
    """返回部署机器人支持的规范 Skill 原语名称。"""

    configured = robot.settings.get("supported_driver_primitives")
    if configured:
        return tuple(str(item) for item in configured)

    if robot.robot_family == "xlerobot" and robot.driver_kind == "mujoco":
        return XLEROBOT_SIM_PRIMITIVES

    if robot.robot_family == "xlerobot" and robot.driver_kind in {"mock", "native"}:
        return SUPPORTED_CLASSIC_PRIMITIVES

    if robot.robot_family == "robocasa" and robot.driver_kind == "grpc":
        return ("inspect_scene", "embodiment_native_action")

    return ()
