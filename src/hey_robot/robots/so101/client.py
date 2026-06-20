from __future__ import annotations

from typing import Any

import numpy as np

from hey_robot.robots.components import OpenCVCamera, ServoBus, ServoBusBattery
from hey_robot.robots.so101.arm import SO101Arm
from hey_robot.robots.so101.config import SO101HardwareConfig


class SO101Client:
    """Native client for a standalone SO101 arm."""

    def __init__(self, config: SO101HardwareConfig) -> None:
        self.config = config
        self.bus = ServoBus(config.serial_bus.port, config.serial_bus.baudrate)
        self.arm = SO101Arm(self.bus, config.arm)
        self.camera = OpenCVCamera(config.camera)
        self.battery = ServoBusBattery(self.bus, config.battery)
        self._startup: dict[str, Any] = {}

    def connect(self) -> None:
        bus_ok = self.bus.connect()
        arm = (
            self.arm.initialize()
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
            "arm": _service_probe(arm, joint_count=_joint_count(arm)),
            "camera": _camera_probe(camera),
            "battery": self.battery_status(),
        }

    def close(self) -> None:
        self.arm.close()
        self.camera.close()
        self.bus.close()

    def diagnose(self, *, video_timeout_ms: int = 3000) -> dict[str, Any]:
        frame_id, image = self.capture_frame(timeout_ms=video_timeout_ms)
        arm = self.arm_status()
        camera_diag = self.camera.diagnostics()
        return {
            "bus": dict(self._startup.get("bus", {})),
            "arm": _service_probe(arm, joint_count=_joint_count(arm)),
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

    def stop(self, *, emergency: bool = False) -> dict[str, Any]:
        if emergency:
            return self.arm.emergency_stop()
        return {"success": True, "message": "SO101 has no mobile base to stop"}

    def arm_status(self) -> dict[str, Any]:
        return self.arm.status()

    def battery_status(self) -> dict[str, Any]:
        return self.battery.read().to_dict()

    def arm_home(self, *, speed: int = 1000) -> dict[str, Any]:
        return self.arm.home(speed=speed)

    def set_joint(
        self, name: str, angle: float, *, speed: int = 1000
    ) -> dict[str, Any]:
        return self.arm.set_joint(name, angle, speed=speed)

    def set_joint_delta(
        self, name: str, delta_angle: float, *, speed: int = 1000
    ) -> dict[str, Any]:
        return self.arm.set_joint_delta(name, delta_angle, speed=speed)

    def set_joints(
        self, joints: dict[str, float], *, speed: int = 1000
    ) -> dict[str, Any]:
        return self.arm.set_joints(joints, speed=speed)

    def set_joints_delta(
        self, joints: dict[str, float], *, speed: int = 1000
    ) -> dict[str, Any]:
        return self.arm.set_joints_delta(joints, speed=speed)

    def move_named_pose(self, pose_name: str, *, speed: int = 1000) -> dict[str, Any]:
        return self.arm.move_named_pose(pose_name, speed=speed)

    def open_gripper(self) -> dict[str, Any]:
        return self.arm.open_gripper()

    def close_gripper(self) -> dict[str, Any]:
        return self.arm.close_gripper()

    def set_gripper_opening_pct(
        self, pct: float, *, speed: int = 1000
    ) -> dict[str, Any]:
        return self.arm.set_gripper_opening_pct(pct, speed=speed)

    def capture_frame(
        self, *, timeout_ms: int = 2000
    ) -> tuple[int | None, np.ndarray | None]:
        return self.camera.capture_frame(timeout_ms=timeout_ms)


def _service_probe(
    response: dict[str, Any], *, joint_count: int | None = None
) -> dict[str, Any]:
    probe = {
        "ok": bool(response.get("success", False)),
        "issue": None
        if response.get("success", False)
        else response.get("message", "service failed"),
        "response": response,
    }
    if joint_count is not None:
        probe["joint_count"] = joint_count
    return probe


def _camera_probe(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(response.get("success", False)),
        "frame_available": False,
        "frame_id": None,
        "issue": None
        if response.get("success", False)
        else response.get("message", "camera failed"),
        "response": response,
    }


def _joint_count(response: dict[str, Any]) -> int:
    joint_states = response.get("joint_states")
    return len(joint_states) if isinstance(joint_states, dict) else 0
