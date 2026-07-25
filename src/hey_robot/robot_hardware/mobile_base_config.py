from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
