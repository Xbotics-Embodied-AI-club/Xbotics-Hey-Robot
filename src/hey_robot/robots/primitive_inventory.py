from __future__ import annotations

from hey_robot.config import RobotSpec
from hey_robot.robots.classic.primitives import SUPPORTED_CLASSIC_PRIMITIVES


def supported_driver_primitives(robot: RobotSpec) -> tuple[str, ...]:
    """Return canonical Skill primitive names supported by a deployment robot."""

    configured = robot.settings.get("supported_driver_primitives")
    if configured:
        return tuple(str(item) for item in configured)

    if robot.robot_family == "xlerobot" and robot.driver_kind in {
        "mock",
        "mujoco",
        "native",
    }:
        return SUPPORTED_CLASSIC_PRIMITIVES

    return ()
