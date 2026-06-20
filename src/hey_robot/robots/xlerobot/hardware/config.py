from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hey_robot.robots.components import (
    OpenCVCameraConfig,
    ServoBusBatteryConfig,
    ServoBusConfig,
)
from hey_robot.robots.lekiwi.config import LeKiwiBaseConfig, base_config_from_settings
from hey_robot.robots.so101.config import SO101ArmConfig, arm_config_from_settings


@dataclass(frozen=True)
class XLeRobotHardwareConfig:
    serial_bus: ServoBusConfig = field(default_factory=ServoBusConfig)
    base: LeKiwiBaseConfig = field(default_factory=LeKiwiBaseConfig)
    arm: SO101ArmConfig = field(default_factory=SO101ArmConfig)
    arms: dict[str, SO101ArmConfig] = field(default_factory=dict)
    default_arm: str = "arm"
    camera: OpenCVCameraConfig = field(default_factory=OpenCVCameraConfig)
    cameras: dict[str, OpenCVCameraConfig] = field(default_factory=dict)
    camera_owners: dict[str, str] = field(default_factory=dict)
    default_camera: str = "front"
    camera_owner: str = "robot_driver"
    battery: ServoBusBatteryConfig = field(default_factory=ServoBusBatteryConfig)
    startup_probe: bool = True


def hardware_config_from_settings(settings: dict[str, Any]) -> XLeRobotHardwareConfig:
    components = dict(settings.get("components", {}) or {})
    bus_settings = dict(settings.get("serial_bus", {}) or {})
    base_settings = dict(components.get("base", {}) or {})
    arm_settings = dict(components.get("arm", {}) or {})
    arms_settings = dict(components.get("arms", {}) or {})
    camera_settings = dict(components.get("camera", {}) or {})
    cameras_settings = dict(components.get("cameras", {}) or {})
    battery_settings = dict(components.get("battery", {}) or {})
    arms = _arm_configs(arms_settings, fallback=arm_settings)
    cameras, camera_owners = _camera_configs(cameras_settings, fallback=camera_settings)
    default_arm = str(components.get("default_arm", "") or next(iter(arms), "arm"))
    default_camera = str(
        components.get("default_camera", "") or next(iter(cameras), "front")
    )
    return XLeRobotHardwareConfig(
        serial_bus=ServoBusConfig(
            port=str(bus_settings.get("port", "COM5")),
            baudrate=int(bus_settings.get("baudrate", 1_000_000)),
        ),
        base=base_config_from_settings(base_settings),
        arm=arms.get(default_arm, arm_config_from_settings(arm_settings)),
        arms=arms,
        default_arm=default_arm,
        camera=cameras.get(default_camera, _camera_config(camera_settings)),
        cameras=cameras,
        camera_owners=camera_owners,
        default_camera=default_camera,
        camera_owner=camera_owners.get(
            default_camera, str(camera_settings.get("owner", "robot_driver"))
        ),
        battery=_battery_config(battery_settings),
        startup_probe=bool(settings.get("startup_probe", True)),
    )


def _camera_config(settings: dict[str, Any]) -> OpenCVCameraConfig:
    owner = str(settings.get("owner", "robot_driver"))
    driver_enabled = bool(settings.get("enabled", True)) and owner != "camera_service"
    return OpenCVCameraConfig(
        enabled=driver_enabled,
        device_id=int(settings.get("device_id", 0)),
        width=_optional_int(settings.get("width")),
        height=_optional_int(settings.get("height")),
        fps=_optional_int(settings.get("fps")),
        backend=str(settings.get("backend", "auto")),
    )


def _arm_configs(
    settings: dict[str, Any], *, fallback: dict[str, Any]
) -> dict[str, SO101ArmConfig]:
    configs: dict[str, SO101ArmConfig] = {}
    if settings:
        for name, arm_settings in settings.items():
            if isinstance(arm_settings, dict):
                configs[str(name)] = arm_config_from_settings(dict(arm_settings))
    if not configs:
        configs["arm"] = arm_config_from_settings(fallback)
    return configs


def _camera_configs(
    settings: dict[str, Any], *, fallback: dict[str, Any]
) -> tuple[dict[str, OpenCVCameraConfig], dict[str, str]]:
    configs: dict[str, OpenCVCameraConfig] = {}
    owners: dict[str, str] = {}
    if settings:
        for name, camera_settings in settings.items():
            if isinstance(camera_settings, dict):
                role = str(name)
                configs[role] = _camera_config(dict(camera_settings))
                owners[role] = str(camera_settings.get("owner", "robot_driver"))
    if not configs:
        role = "front"
        configs[role] = _camera_config(fallback)
        owners[role] = str(fallback.get("owner", "robot_driver"))
    return configs, owners


def _battery_config(settings: dict[str, Any]) -> ServoBusBatteryConfig:
    return ServoBusBatteryConfig(
        enabled=bool(settings.get("enabled", True)),
        servo_ids=[int(item) for item in list(settings.get("servo_ids", [1]) or [])],
        full_voltage=float(settings.get("full_voltage", 12.6)),
        low_voltage=float(settings.get("low_voltage", 10.5)),
        critical_voltage=float(settings.get("critical_voltage", 9.5)),
        min_voltage=float(settings.get("min_voltage", 9.0)),
    )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
