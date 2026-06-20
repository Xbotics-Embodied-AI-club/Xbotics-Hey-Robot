from __future__ import annotations

from hey_robot.config import DeploymentConfig
from hey_robot.robots.base import RobotDriver, RobotDriverContext
from hey_robot.robots.embodiments import get_embodiment_profile
from hey_robot.robots.lekiwi import LeKiwiDriver
from hey_robot.robots.mock import MockRobotDriver
from hey_robot.robots.simulation.xlerobot_sim_driver import XLeRobotSimDriver
from hey_robot.robots.so101 import SO101Driver
from hey_robot.robots.xlerobot import XLeRobotDriver
from hey_robot.skills.registry import registry_from_config


class RobotManager:
    def __init__(self, config: DeploymentConfig) -> None:
        self.config = config
        self.skill_catalog = registry_from_config(config).robot_skill_catalog()
        self._drivers: dict[str, RobotDriver] = {}
        self._build_drivers()

    def get(self, robot_id: str) -> RobotDriver | None:
        return self._drivers.get(robot_id)

    def require(self, robot_id: str) -> RobotDriver:
        driver = self.get(robot_id)
        if driver is None:
            raise KeyError(f"unknown robot: {robot_id}")
        return driver

    def all(self) -> list[RobotDriver]:
        return list(self._drivers.values())

    def _build_drivers(self) -> None:
        for robot_id, spec in self.config.robots.items():
            if not spec.enabled:
                continue
            context = RobotDriverContext(
                robot_id=robot_id,
                spec=spec,
                deployment_id=self.config.deployment.id,
                embodiment=get_embodiment_profile(spec),
                skill_catalog=self.skill_catalog,
            )
            if spec.driver_kind == "mock":
                self._drivers[robot_id] = MockRobotDriver(context)
                continue
            if spec.robot_family == "xlerobot" and spec.driver_kind == "mujoco":
                self._drivers[robot_id] = XLeRobotSimDriver(context)
                continue
            if spec.robot_family == "xlerobot" and spec.driver_kind == "native":
                self._drivers[robot_id] = XLeRobotDriver(context)
                continue
            if spec.robot_family == "so101":
                self._drivers[robot_id] = SO101Driver(context)
                continue
            if spec.robot_family == "lekiwi":
                self._drivers[robot_id] = LeKiwiDriver(context)
                continue
            raise ValueError(
                "unsupported robot driver combination: "
                f"family={spec.robot_family} environment={spec.robot_environment} driver={spec.driver_kind}"
            )
