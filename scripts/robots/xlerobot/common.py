"""XLeRobot 诊断脚本共享的工具函数。

包含：部署配置加载、OpenCV 摄像头参数覆盖、JSON/文本输出格式化等。
被 diagnose.py / check_arm.py / scan_servos.py / scan_cameras.py 复用。
本模块不直接运行，只供其他脚本 import。
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hey_robot.config import DeploymentConfig
from hey_robot.robots.components import OpenCVCameraConfig
from hey_robot.robots.xlerobot.hardware.config import (
    XLeRobotHardwareConfig,
    hardware_config_from_settings,
)


def load_hardware_config(
    config_path: str,
    robot_id: str,
    *,
    serial_port: str | None = None,
    camera_device: int | None = None,
    camera_backend: str | None = None,
) -> tuple[dict[str, Any], XLeRobotHardwareConfig]:
    deployment = DeploymentConfig.from_yaml(config_path)
    spec = deployment.robots[robot_id]
    settings = dict(spec.settings)
    if serial_port:
        settings["serial_bus"] = {
            **dict(settings.get("serial_bus", {}) or {}),
            "port": serial_port,
        }
    hardware = hardware_config_from_settings(settings)
    if camera_device is not None or camera_backend is not None:
        hardware = replace(
            hardware,
            camera=replace(
                hardware.camera,
                device_id=hardware.camera.device_id
                if camera_device is None
                else camera_device,
                backend=hardware.camera.backend
                if camera_backend is None
                else camera_backend,
            ),
        )
    return settings, hardware


def print_json_or_text(data: dict[str, Any], text: str, *, as_json: bool) -> None:
    if as_json:
        import json

        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(text)


def angle_from_position(position: int, *, offset: int, scale: float) -> float:
    return (position - offset) / scale


def camera_config_with_device(
    config: OpenCVCameraConfig, device_id: int
) -> OpenCVCameraConfig:
    return replace(config, device_id=device_id)
