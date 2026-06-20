from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hey_robot.robots.components import (
    OpenCVCameraConfig,
    ServoBusBatteryConfig,
    ServoBusConfig,
)
from hey_robot.robots.so101.config import _battery_config, _camera_config


@dataclass(frozen=True)
class LeKiwiBaseConfig:
    type: str = "lekiwi_base"
    enabled: bool = True
    left_front_id: int = 7
    right_front_id: int = 9
    rear_id: int = 8
    wheel_radius_m: float = 0.08
    chassis_radius_m: float = 0.18
    max_linear_speed_mps: float = 0.5
    max_angular_speed_radps: float = 1.0
    default_wheel_speed: int = 3250


@dataclass(frozen=True)
class LeKiwiHardwareConfig:
    serial_bus: ServoBusConfig = field(default_factory=ServoBusConfig)
    base: LeKiwiBaseConfig = field(default_factory=LeKiwiBaseConfig)
    camera: OpenCVCameraConfig = field(
        default_factory=lambda: OpenCVCameraConfig(enabled=False)
    )
    battery: ServoBusBatteryConfig = field(default_factory=ServoBusBatteryConfig)
    startup_probe: bool = True


def hardware_config_from_settings(settings: dict[str, Any]) -> LeKiwiHardwareConfig:
    components = dict(settings.get("components", {}) or {})
    bus_settings = dict(settings.get("serial_bus", {}) or {})
    base_settings = dict(components.get("base", settings.get("base", {})) or {})
    camera_settings = dict(components.get("camera", settings.get("camera", {})) or {})
    battery_settings = dict(
        components.get("battery", settings.get("battery", {})) or {}
    )
    return LeKiwiHardwareConfig(
        serial_bus=ServoBusConfig(
            port=str(bus_settings.get("port", "COM5")),
            baudrate=int(bus_settings.get("baudrate", 1_000_000)),
        ),
        base=base_config_from_settings(base_settings),
        camera=_camera_config(camera_settings, default_enabled=False),
        battery=_battery_config(battery_settings),
        startup_probe=bool(settings.get("startup_probe", True)),
    )


def base_config_from_settings(settings: dict[str, Any]) -> LeKiwiBaseConfig:
    default = LeKiwiBaseConfig()
    return LeKiwiBaseConfig(
        type=str(settings.get("type", "lekiwi_base")),
        enabled=bool(settings.get("enabled", True)),
        left_front_id=int(settings.get("left_front_id", default.left_front_id)),
        right_front_id=int(settings.get("right_front_id", default.right_front_id)),
        rear_id=int(settings.get("rear_id", default.rear_id)),
        wheel_radius_m=float(settings.get("wheel_radius_m", default.wheel_radius_m)),
        chassis_radius_m=float(
            settings.get("chassis_radius_m", default.chassis_radius_m)
        ),
        max_linear_speed_mps=float(
            settings.get("max_linear_speed_mps", default.max_linear_speed_mps)
        ),
        max_angular_speed_radps=float(
            settings.get("max_angular_speed_radps", default.max_angular_speed_radps)
        ),
        default_wheel_speed=int(
            settings.get("default_wheel_speed", default.default_wheel_speed)
        ),
    )
