from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
            # 低位、朝前的中心姿态。
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
