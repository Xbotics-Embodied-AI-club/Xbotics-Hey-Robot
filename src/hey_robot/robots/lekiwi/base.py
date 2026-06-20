from __future__ import annotations

import math
import time
from typing import Any

from hey_robot.robots.components import ServoBus
from hey_robot.robots.lekiwi.config import LeKiwiBaseConfig


class LeKiwiBase:
    """LeKiwi three-wheel mobile base component."""

    def __init__(self, bus: ServoBus, config: LeKiwiBaseConfig) -> None:
        self.bus = bus
        self.config = config
        self.wheel_ids = [config.left_front_id, config.right_front_id, config.rear_id]
        self._initialized = False
        self._velocity = (0.0, 0.0, 0.0)
        self._last_velocity_command: dict = {}
        self._last_stop_command: dict = {}

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> dict:
        if not self.config.enabled:
            return {"success": True, "message": "base disabled", "enabled": False}
        if not self.bus.connected:
            return {"success": False, "message": "servo bus is not connected"}
        missing = [
            servo_id for servo_id in self.wheel_ids if not self.bus.ping(servo_id)
        ]
        if missing:
            return {
                "success": False,
                "message": f"missing base servos: {missing}",
                "missing_servos": missing,
            }
        self.bus.torque_disable()
        for servo_id in self.wheel_ids:
            if not self.bus.set_wheel_mode(servo_id):
                return {
                    "success": False,
                    "message": f"failed to set wheel mode for servo {servo_id}",
                }
        self.bus.torque_enable()
        self.stop()
        self._initialized = True
        return {
            "success": True,
            "message": "base initialized",
            "wheel_ids": self.wheel_ids,
        }

    def set_velocity(self, vx: float, vy: float, omega: float) -> dict:
        if not self.config.enabled:
            return {"success": True, "message": "base disabled"}
        if not self._initialized:
            return {"success": False, "message": "base is not initialized"}
        command: dict[str, Any] = {
            "kind": "set_velocity",
            "timestamp": time.time(),
            "requested": {"vx": float(vx), "vy": float(vy), "vz": float(omega)},
            "wheel_writes": [],
        }
        vx = _clamp(
            vx, -self.config.max_linear_speed_mps, self.config.max_linear_speed_mps
        )
        vy = _clamp(
            vy, -self.config.max_linear_speed_mps, self.config.max_linear_speed_mps
        )
        omega = _clamp(
            omega,
            -self.config.max_angular_speed_radps,
            self.config.max_angular_speed_radps,
        )
        command["clamped"] = {"vx": vx, "vy": vy, "vz": omega}
        wheel_speeds = self._inverse_kinematics(vx, vy, omega)
        for servo_id, wheel_speed in zip(self.wheel_ids, wheel_speeds, strict=True):
            servo_speed = self._wheel_speed_to_servo(wheel_speed)
            write_ok = self.bus.write_speed(servo_id, servo_speed)
            command["wheel_writes"].append(
                {
                    "servo_id": servo_id,
                    "wheel_speed": wheel_speed,
                    "servo_speed": servo_speed,
                    "success": write_ok,
                }
            )
            if not write_ok:
                command["success"] = False
                command["failed_servo_id"] = servo_id
                self._last_velocity_command = command
                stop_response = self.stop()
                command["stop_response"] = stop_response
                return {
                    "success": False,
                    "message": f"failed to write speed for wheel servo {servo_id}",
                    "control": command,
                }
        self._velocity = (vx, vy, omega)
        command["success"] = True
        self._last_velocity_command = command
        return {
            "success": True,
            "message": "velocity applied",
            "velocity": {"vx": vx, "vy": vy, "vz": omega},
            "control": command,
        }

    def stop(self) -> dict:
        previous_velocity = {
            "vx": self._velocity[0],
            "vy": self._velocity[1],
            "vz": self._velocity[2],
        }
        command: dict[str, Any] = {
            "kind": "stop",
            "timestamp": time.time(),
            "requested": {"vx": 0.0, "vy": 0.0, "vz": 0.0},
            "previous_velocity": previous_velocity,
            "wheel_writes": [],
        }
        self._velocity = (0.0, 0.0, 0.0)
        if not self.config.enabled:
            return {"success": True, "message": "base disabled"}
        ok = True
        for servo_id in self.wheel_ids:
            write_ok = self.bus.write_speed(servo_id, 0)
            command["wheel_writes"].append(
                {"servo_id": servo_id, "servo_speed": 0, "success": write_ok}
            )
            ok = ok and write_ok
        command["success"] = ok
        self._last_stop_command = command
        return {
            "success": ok,
            "message": "base stopped" if ok else "failed to stop base",
            "control": command,
        }

    def status(self) -> dict:
        return {
            "success": self._initialized or not self.config.enabled,
            "enabled": self.config.enabled,
            "initialized": self._initialized,
            "wheel_ids": self.wheel_ids,
            "velocity": {
                "vx": self._velocity[0],
                "vy": self._velocity[1],
                "vz": self._velocity[2],
            },
        }

    def control_diagnostics(self) -> dict:
        return {
            "last_velocity_command": dict(self._last_velocity_command),
            "last_stop_command": dict(self._last_stop_command),
            "current_velocity": {
                "vx": self._velocity[0],
                "vy": self._velocity[1],
                "vz": self._velocity[2],
            },
        }

    def close(self) -> None:
        if self._initialized:
            self.stop()
        self._initialized = False

    def _inverse_kinematics(self, vx: float, vy: float, omega: float) -> list[float]:
        radius = self.config.chassis_radius_m
        sqrt3_2 = math.sqrt(3) / 2
        return [
            -sqrt3_2 * vx - 0.5 * vy - radius * omega,
            sqrt3_2 * vx - 0.5 * vy - radius * omega,
            vy - radius * omega,
        ]

    def _wheel_speed_to_servo(self, wheel_speed: float) -> int:
        if self.config.max_linear_speed_mps <= 0:
            return 0
        ratio = wheel_speed / self.config.max_linear_speed_mps
        value = int(ratio * self.config.default_wheel_speed)
        return max(-3250, min(3250, value))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))
