from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any, cast

from hey_robot.config import DeploymentConfig
from hey_robot.robot_api import (
    RobotActionSpec,
    RobotDriver,
    RobotDriverContext,
)
from hey_robot.robot_runtime.embodiments import get_embodiment_profile

DriverFactory = Callable[[RobotDriverContext], RobotDriver]

_BUILTIN_DRIVERS = {
    ("*", "*", "mock"): "hey_robot.robot_backends.mock:MockRobotDriver",
    (
        "xlerobot",
        "sim",
        "mujoco",
    ): "hey_robot.robot_backends.simulation.xlerobot_sim_driver:XLeRobotSimDriver",
    (
        "xlerobot",
        "real",
        "native",
    ): "hey_robot.robot_backends.xlerobot.driver:XLeRobotDriver",
    (
        "robocasa",
        "remote",
        "grpc",
    ): "hey_robot.robot_backends.robocasa_remote.driver:create_driver",
}


class RobotManager:
    def __init__(
        self,
        config: DeploymentConfig,
        *,
        action_specs: tuple[RobotActionSpec, ...] = (),
    ) -> None:
        self.config = config
        self.action_specs = action_specs
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
            context = create_driver_context(
                robot_id,
                spec,
                self.config.deployment.id,
                action_specs=self.action_specs,
            )
            factory = _load_driver_factory(spec)
            self._drivers[robot_id] = factory(context)


def create_driver_context(
    robot_id: str,
    spec: Any,
    deployment_id: str,
    *,
    action_specs: tuple[RobotActionSpec, ...] = (),
) -> RobotDriverContext:
    """Translate deployment configuration into the backend-neutral driver contract."""
    return RobotDriverContext(
        robot_id=robot_id,
        deployment_id=deployment_id,
        robot_family=spec.robot_family,
        environment=spec.robot_environment,
        driver_kind=spec.driver_kind,
        settings=dict(spec.settings),
        embodiment=get_embodiment_profile(spec),
        action_specs=action_specs,
    )


def _load_driver_factory(spec: Any) -> DriverFactory:
    explicit = str(spec.settings.get("driver_factory") or "").strip()
    target = explicit or _BUILTIN_DRIVERS.get(
        (spec.robot_family, spec.robot_environment, spec.driver_kind)
    )
    if target is None and spec.driver_kind == "mock":
        target = _BUILTIN_DRIVERS[("*", "*", "mock")]
    if target is None:
        raise ValueError(
            "unsupported robot driver combination: "
            f"family={spec.robot_family} environment={spec.robot_environment} "
            f"driver={spec.driver_kind}"
        )
    module_name, separator, attribute = target.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"invalid robot driver factory: {target!r}")
    factory = getattr(import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"robot driver factory is not callable: {target!r}")
    return cast(DriverFactory, factory)
