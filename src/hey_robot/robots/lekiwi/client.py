from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from hey_robot.robots.components import OpenCVCamera, ServoBus, ServoBusBattery
from hey_robot.robots.lekiwi.base import LeKiwiBase
from hey_robot.robots.lekiwi.config import LeKiwiHardwareConfig
from hey_robot.robots.so101.client import _camera_probe, _service_probe


class LeKiwiClient:
    """Native client for a standalone LeKiwi mobile base."""

    def __init__(
        self,
        config: LeKiwiHardwareConfig,
        *,
        default_linear_speed: float = 0.2,
        default_angular_speed: float = 0.45,
        motion_time_scale: float = 2.0,
    ) -> None:
        self.config = config
        self.default_linear_speed = default_linear_speed
        self.default_angular_speed = default_angular_speed
        self.motion_time_scale = motion_time_scale
        self.bus = ServoBus(config.serial_bus.port, config.serial_bus.baudrate)
        self.base = LeKiwiBase(self.bus, config.base)
        self.camera = OpenCVCamera(config.camera)
        self.battery = ServoBusBattery(self.bus, config.battery)
        self._startup: dict[str, Any] = {}

    def connect(self) -> None:
        bus_ok = self.bus.connect()
        base = (
            self.base.initialize()
            if bus_ok
            else {"success": False, "message": "servo bus connection failed"}
        )
        camera = self.camera.open()
        self._startup = {
            "bus": {
                "ok": bus_ok,
                "port": self.config.serial_bus.port,
                "baudrate": self.config.serial_bus.baudrate,
                "message": "servo bus connected"
                if bus_ok
                else "servo bus connection failed",
            },
            "base": _service_probe(base),
            "camera": _camera_probe(camera),
            "battery": self.battery_status(),
        }

    def close(self) -> None:
        self.base.close()
        self.camera.close()
        self.bus.close()

    def diagnose(self, *, video_timeout_ms: int = 3000) -> dict[str, Any]:
        frame_id, image = self.capture_frame(timeout_ms=video_timeout_ms)
        camera_diag = self.camera.diagnostics()
        return {
            "bus": dict(self._startup.get("bus", {})),
            "base": _service_probe(self.base.status()),
            "camera": {
                "ok": image is not None
                or (camera_diag.get("success") and not self.config.camera.enabled),
                "frame_available": image is not None,
                "frame_id": frame_id,
                "image_shape": list(image.shape) if image is not None else None,
                "timeout_ms": video_timeout_ms,
                "issue": None
                if image is not None
                else camera_diag.get("error") or "no native camera frame available",
                "response": camera_diag,
            },
            "battery": self.battery_status(),
        }

    def set_velocity(self, vx: float, vz: float, *, vy: float = 0.0) -> dict[str, Any]:
        return self.base.set_velocity(vx, vy, vz)

    def stop(self, *, emergency: bool = False) -> dict[str, Any]:
        if emergency:
            return {
                **self.base.stop(),
                "emergency": True,
            }
        return self.base.stop()

    def unlock_chassis(self) -> dict[str, Any]:
        return {
            "success": True,
            "message": "native LeKiwi base does not use HomeBot emergency lock",
        }

    def move_forward_cm(
        self, distance_cm: float, *, speed: float | None = None
    ) -> dict[str, Any]:
        speed = abs(speed or self.default_linear_speed)
        direction = 1.0 if distance_cm >= 0 else -1.0
        duration = abs(distance_cm) / 100.0 / max(speed, 0.01) * self.motion_time_scale
        return self._pulse_velocity(vx=direction * speed, vz=0.0, duration=duration)

    def turn_deg(
        self, angle_deg: float, *, angular_speed: float | None = None
    ) -> dict[str, Any]:
        angular_speed = abs(angular_speed or self.default_angular_speed)
        direction = 1.0 if angle_deg >= 0 else -1.0
        duration = (
            abs(math.radians(angle_deg))
            / max(angular_speed, 0.01)
            * self.motion_time_scale
        )
        return self._pulse_velocity(
            vx=0.0, vz=direction * angular_speed, duration=duration
        )

    def strafe_left_cm(
        self, distance_cm: float, *, speed: float | None = None
    ) -> dict[str, Any]:
        speed = abs(speed or self.default_linear_speed)
        direction = 1.0 if distance_cm >= 0 else -1.0
        duration = abs(distance_cm) / 100.0 / max(speed, 0.01) * self.motion_time_scale
        return self._pulse_velocity(
            vx=0.0, vz=0.0, duration=duration, vy=direction * speed
        )

    def strafe_right_cm(
        self, distance_cm: float, *, speed: float | None = None
    ) -> dict[str, Any]:
        return self.strafe_left_cm(-abs(distance_cm), speed=speed)

    def battery_status(self) -> dict[str, Any]:
        return self.battery.read().to_dict()

    def capture_frame(
        self, *, timeout_ms: int = 2000
    ) -> tuple[int | None, np.ndarray | None]:
        return self.camera.capture_frame(timeout_ms=timeout_ms)

    def _pulse_velocity(
        self, *, vx: float, vz: float, duration: float, vy: float = 0.0
    ) -> dict[str, Any]:
        start = time.time()
        last_response: dict[str, Any] = {"success": True, "message": "not started"}
        while time.time() - start < duration:
            last_response = self.set_velocity(vx, vz, vy=vy)
            if not bool(last_response.get("success", False)):
                self.stop()
                return last_response
            time.sleep(0.2)
        stop_response = self.stop()
        return {
            "success": bool(stop_response.get("success", False)),
            "message": "motion completed"
            if stop_response.get("success", False)
            else "motion stop failed",
            "last_motion_response": last_response,
            "stop_response": stop_response,
        }
