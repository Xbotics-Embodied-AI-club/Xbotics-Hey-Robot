from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hey_robot.robots.components import (
    OpenCVCameraConfig,
    ServoBusBatteryConfig,
    ServoBusConfig,
)


@dataclass(frozen=True)
class SO101ArmConfig:
    type: str = "so101_arm"
    enabled: bool = True
    joint_ids: dict[str, int] = field(
        default_factory=lambda: {
            "base": 1,
            "shoulder": 2,
            "elbow": 3,
            "wrist_flex": 4,
            "wrist_roll": 5,
            "gripper": 6,
        }
    )
    joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "base": (-180.0, 180.0),
            "shoulder": (0.0, 180.0),
            "elbow": (0.0, 180.0),
            "wrist_flex": (-90.0, 90.0),
            "wrist_roll": (-180.0, 180.0),
            "gripper": (0.0, 90.0),
        }
    )
    rest_position: dict[str, float] = field(
        default_factory=lambda: {
            # Low forward-facing center pose.
            "base": 0.0,
            "shoulder": 10.0,
            "elbow": 140.0,
            "wrist_flex": 25.0,
            "wrist_roll": -90.0,
            "gripper": 45.0,
        }
    )
    named_poses: dict[str, dict[str, float]] = field(default_factory=dict)
    default_speed: int = 1000
    default_acc: int = 50
    angle_offset: int = 2048
    angle_scale: float = 4096 / 360
    auto_home_on_startup: bool = False
    home_on_close: bool = False


@dataclass(frozen=True)
class SO101HardwareConfig:
    serial_bus: ServoBusConfig = field(default_factory=ServoBusConfig)
    arm: SO101ArmConfig = field(default_factory=SO101ArmConfig)
    camera: OpenCVCameraConfig = field(
        default_factory=lambda: OpenCVCameraConfig(enabled=False)
    )
    battery: ServoBusBatteryConfig = field(default_factory=ServoBusBatteryConfig)
    startup_probe: bool = True


def hardware_config_from_settings(settings: dict[str, Any]) -> SO101HardwareConfig:
    components = dict(settings.get("components", {}) or {})
    bus_settings = dict(settings.get("serial_bus", {}) or {})
    arm_settings = dict(components.get("arm", settings.get("arm", {})) or {})
    camera_settings = dict(components.get("camera", settings.get("camera", {})) or {})
    battery_settings = dict(
        components.get("battery", settings.get("battery", {})) or {}
    )
    return SO101HardwareConfig(
        serial_bus=ServoBusConfig(
            port=str(bus_settings.get("port", "COM5")),
            baudrate=int(bus_settings.get("baudrate", 1_000_000)),
        ),
        arm=arm_config_from_settings(arm_settings),
        camera=_camera_config(camera_settings, default_enabled=False),
        battery=_battery_config(battery_settings),
        startup_probe=bool(settings.get("startup_probe", True)),
    )


def arm_config_from_settings(settings: dict[str, Any]) -> SO101ArmConfig:
    default = SO101ArmConfig()
    return SO101ArmConfig(
        type=str(settings.get("type", "so101_arm")),
        enabled=bool(settings.get("enabled", True)),
        joint_ids=_joint_ids(settings),
        joint_limits=_joint_limits(settings),
        rest_position={
            str(k): float(v)
            for k, v in dict(settings.get("rest_position", {}) or {}).items()
        }
        or default.rest_position,
        named_poses={
            str(name): {
                str(joint): float(value) for joint, value in dict(pose or {}).items()
            }
            for name, pose in dict(settings.get("named_poses", {}) or {}).items()
            if isinstance(pose, dict)
        },
        default_speed=int(settings.get("default_speed", default.default_speed)),
        default_acc=int(settings.get("default_acc", default.default_acc)),
        auto_home_on_startup=bool(
            settings.get("auto_home_on_startup", default.auto_home_on_startup)
        ),
        home_on_close=bool(settings.get("home_on_close", default.home_on_close)),
    )


def _joint_ids(settings: dict[str, Any]) -> dict[str, int]:
    configured = settings.get("joint_ids")
    if isinstance(configured, dict):
        return {str(key): int(value) for key, value in configured.items()}
    defaults = SO101ArmConfig().joint_ids
    keys = {
        "base": "base_id",
        "shoulder": "shoulder_id",
        "elbow": "elbow_id",
        "wrist_flex": "wrist_flex_id",
        "wrist_roll": "wrist_roll_id",
        "gripper": "gripper_id",
    }
    return {
        joint: int(settings.get(config_key, defaults[joint]))
        for joint, config_key in keys.items()
    }


def _joint_limits(settings: dict[str, Any]) -> dict[str, tuple[float, float]]:
    configured = settings.get("joint_limits")
    if not isinstance(configured, dict):
        return SO101ArmConfig().joint_limits
    limits = {}
    for joint, value in configured.items():
        if isinstance(value, (list, tuple)) and len(value) == 2:
            limits[str(joint)] = (float(value[0]), float(value[1]))
    return limits or SO101ArmConfig().joint_limits


def _camera_config(
    settings: dict[str, Any], *, default_enabled: bool
) -> OpenCVCameraConfig:
    owner = str(settings.get("owner", "robot_driver"))
    driver_enabled = (
        bool(settings.get("enabled", default_enabled)) and owner != "camera_service"
    )
    return OpenCVCameraConfig(
        enabled=driver_enabled,
        device_id=int(settings.get("device_id", 0)),
        width=_optional_int(settings.get("width")),
        height=_optional_int(settings.get("height")),
        fps=_optional_int(settings.get("fps")),
        backend=str(settings.get("backend", "auto")),
    )


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
