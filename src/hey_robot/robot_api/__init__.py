"""Stable contracts shared by robot orchestration, skills, and backends."""

from hey_robot.robot_api.client import (
    RobotActionResult,
    RobotClient,
    RobotClientCapabilities,
)
from hey_robot.robot_api.driver import (
    BaseVelocityStreamDriver,
    PerceptionRequestDriver,
    RobotActionSpec,
    RobotCapabilities,
    RobotDriver,
    RobotDriverContext,
    RobotHealth,
)
from hey_robot.robot_api.embodiment import EmbodimentProfile
from hey_robot.robot_api.observation import DriverObservation, ObservationAsset

__all__ = [
    "BaseVelocityStreamDriver",
    "DriverObservation",
    "EmbodimentProfile",
    "ObservationAsset",
    "PerceptionRequestDriver",
    "RobotActionResult",
    "RobotActionSpec",
    "RobotCapabilities",
    "RobotClient",
    "RobotClientCapabilities",
    "RobotDriver",
    "RobotDriverContext",
    "RobotHealth",
]
