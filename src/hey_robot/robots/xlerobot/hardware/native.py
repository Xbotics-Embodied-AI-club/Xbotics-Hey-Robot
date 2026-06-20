from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from hey_robot.robots.components import OpenCVCamera, ServoBus, ServoBusBattery
from hey_robot.robots.lekiwi.base import LeKiwiBase
from hey_robot.robots.so101.arm import SO101Arm
from hey_robot.robots.xlerobot.hardware.config import XLeRobotHardwareConfig


class NativeXLeRobotClient:
    """Native hardware client for XLeRobot as SO101 arm + LeKiwi base composition."""

    def __init__(
        self,
        config: XLeRobotHardwareConfig,
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
        self.arms = {
            name: SO101Arm(self.bus, arm_config)
            for name, arm_config in config.arms.items()
        }
        self.arm = self.arms[config.default_arm]
        self.cameras = {
            name: OpenCVCamera(camera_config)
            for name, camera_config in config.cameras.items()
        }
        self.camera = self.cameras[config.default_camera]
        self.battery = ServoBusBattery(self.bus, config.battery)
        self._startup: dict[str, Any] = {}
        self._last_motion_report: dict[str, Any] = {}

    def connect(self) -> None:
        bus_ok = self.bus.connect()
        base = (
            self.base.initialize()
            if bus_ok
            else {"success": False, "message": "servo bus connection failed"}
        )
        arm = self._arm_service_results("initialize", bus_ok=bus_ok)
        camera = self._camera_service_results("open")
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
            "arm": _multi_service_probe(arm),
            "camera": _multi_camera_probe(camera),
            "battery": self.battery_status(),
        }

    def close(self) -> None:
        self.base.close()
        for arm in self.arms.values():
            arm.close()
        for camera in self.cameras.values():
            camera.close()
        self.bus.close()

    def diagnose(self, *, video_timeout_ms: int = 3000) -> dict[str, Any]:
        frame_id, image = self.capture_frame(timeout_ms=video_timeout_ms)
        arm = self.arm_status()
        base = self.base.status()
        camera_diag = self.camera_status()
        return {
            "bus": dict(self._startup.get("bus", {})),
            "servo_bus": self._servo_bus_diagnostics(),
            "base": _service_diagnostic(self._startup.get("base"), base),
            "arm": _multi_service_diagnostic(self._startup.get("arm"), arm),
            "camera": _multi_camera_diagnostic(
                self._startup.get("camera"),
                camera_diag,
                default_camera=self.config.default_camera,
                frame_id=frame_id,
                image=image,
                timeout_ms=video_timeout_ms,
            ),
            "battery": self.battery_status(),
        }

    def set_velocity(self, vx: float, vz: float, *, vy: float = 0.0) -> dict[str, Any]:
        response = self.base.set_velocity(vx, vy, vz)
        self._remember_motion_report(
            {
                "kind": "set_velocity",
                "timestamp": time.time(),
                "requested": {"vx": float(vx), "vy": float(vy), "vz": float(vz)},
                "success": bool(response.get("success", False)),
                "message": response.get("message"),
                "control": response.get("control"),
            }
        )
        return response

    def stop(self, *, emergency: bool = False) -> dict[str, Any]:
        base = self.base.stop()
        if emergency:
            arm = self.arm.emergency_stop()
            response = {
                "success": bool(base.get("success")) and bool(arm.get("success")),
                "message": "emergency stop applied",
                "base": base,
                "arm": arm,
            }
            self._remember_motion_report(
                {
                    "kind": "emergency_stop",
                    "timestamp": time.time(),
                    "success": bool(response.get("success", False)),
                    "message": response.get("message"),
                    "base": base,
                    "arm": arm,
                    "control": base.get("control"),
                }
            )
            return response
        self._remember_motion_report(
            {
                "kind": "stop",
                "timestamp": time.time(),
                "success": bool(base.get("success", False)),
                "message": base.get("message"),
                "control": base.get("control"),
            }
        )
        return base

    def base_stop(self) -> dict[str, Any]:
        return self.base.stop()

    def base_emergency_stop(self) -> dict[str, Any]:
        return self.stop(emergency=True)

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

    def move_backward_cm(
        self, distance_cm: float, *, speed: float | None = None
    ) -> dict[str, Any]:
        return self.move_forward_cm(-abs(distance_cm), speed=speed)

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

    def turn_left_deg(
        self, angle_deg: float, *, angular_speed: float | None = None
    ) -> dict[str, Any]:
        return self.turn_deg(-abs(angle_deg), angular_speed=angular_speed)

    def turn_right_deg(
        self, angle_deg: float, *, angular_speed: float | None = None
    ) -> dict[str, Any]:
        return self.turn_deg(abs(angle_deg), angular_speed=angular_speed)

    def arm_status(self) -> dict[str, Any]:
        arms = {name: arm.status() for name, arm in self.arms.items()}
        default = arms.get(self.config.default_arm, {})
        return {
            **dict(default),
            "ok": all(bool(item.get("success", False)) for item in arms.values()),
            "default_arm": self.config.default_arm,
            "arms": arms,
        }

    def battery_status(self) -> dict[str, Any]:
        return self.battery.read().to_dict()

    def arm_home(
        self, *, speed: int = 1000, arm_name: str | None = None
    ) -> dict[str, Any]:
        arm = self._arm(arm_name)
        response = arm.home(speed=speed)
        return self._tag_arm_response(
            response, resolved_arm_name(arm_name, self.config.default_arm)
        )

    def set_joint(
        self, name: str, angle: float, *, speed: int = 1000, arm_name: str | None = None
    ) -> dict[str, Any]:
        response = self._arm(arm_name).set_joint(name, angle, speed=speed)
        return self._tag_arm_response(
            response, resolved_arm_name(arm_name, self.config.default_arm)
        )

    def set_joint_delta(
        self,
        name: str,
        delta_angle: float,
        *,
        speed: int = 1000,
        arm_name: str | None = None,
    ) -> dict[str, Any]:
        response = self._arm(arm_name).set_joint_delta(name, delta_angle, speed=speed)
        return self._tag_arm_response(
            response, resolved_arm_name(arm_name, self.config.default_arm)
        )

    def set_joints(
        self,
        joints: dict[str, float],
        *,
        speed: int = 1000,
        arm_name: str | None = None,
    ) -> dict[str, Any]:
        response = self._arm(arm_name).set_joints(joints, speed=speed)
        return self._tag_arm_response(
            response, resolved_arm_name(arm_name, self.config.default_arm)
        )

    def set_joints_delta(
        self,
        joints: dict[str, float],
        *,
        speed: int = 1000,
        arm_name: str | None = None,
    ) -> dict[str, Any]:
        response = self._arm(arm_name).set_joints_delta(joints, speed=speed)
        return self._tag_arm_response(
            response, resolved_arm_name(arm_name, self.config.default_arm)
        )

    def move_named_pose(
        self, pose_name: str, *, speed: int = 1000, arm_name: str | None = None
    ) -> dict[str, Any]:
        response = self._arm(arm_name).move_named_pose(pose_name, speed=speed)
        return self._tag_arm_response(
            response, resolved_arm_name(arm_name, self.config.default_arm)
        )

    def open_gripper(self, *, arm_name: str | None = None) -> dict[str, Any]:
        response = self._arm(arm_name).open_gripper()
        return self._tag_arm_response(
            response, resolved_arm_name(arm_name, self.config.default_arm)
        )

    def close_gripper(self, *, arm_name: str | None = None) -> dict[str, Any]:
        response = self._arm(arm_name).close_gripper()
        return self._tag_arm_response(
            response, resolved_arm_name(arm_name, self.config.default_arm)
        )

    def set_gripper_opening_pct(
        self, pct: float, *, speed: int = 1000, arm_name: str | None = None
    ) -> dict[str, Any]:
        response = self._arm(arm_name).set_gripper_opening_pct(pct, speed=speed)
        return self._tag_arm_response(
            response, resolved_arm_name(arm_name, self.config.default_arm)
        )

    def arm_stop(self, *, arm_name: str | None = None) -> dict[str, Any]:
        response = self._arm(arm_name).emergency_stop()
        return self._tag_arm_response(
            response, resolved_arm_name(arm_name, self.config.default_arm)
        )

    def capture_frame(
        self, *, timeout_ms: int = 2000
    ) -> tuple[int | None, np.ndarray | None]:
        return self.camera.capture_frame(timeout_ms=timeout_ms)

    def capture_frames(self, *, timeout_ms: int = 2000) -> dict[str, dict[str, Any]]:
        frames: dict[str, dict[str, Any]] = {}
        for name, camera in self.cameras.items():
            frame_id, image = camera.capture_frame(timeout_ms=timeout_ms)
            frames[name] = {"frame_id": frame_id, "image": image}
        return frames

    def camera_status(self) -> dict[str, Any]:
        cameras = {
            name: {
                **camera.diagnostics(),
                "owner": self.config.camera_owners.get(name, "robot_driver"),
            }
            for name, camera in self.cameras.items()
        }
        default = cameras.get(self.config.default_camera, {})
        ok = all(
            bool(item.get("success")) or item.get("owner") == "camera_service"
            for item in cameras.values()
        )
        issue = next(
            (
                item.get("error") or "camera failed"
                for item in cameras.values()
                if not (
                    bool(item.get("success")) or item.get("owner") == "camera_service"
                )
            ),
            None,
        )
        return {
            **dict(default),
            "ok": ok,
            "issue": issue,
            "default_camera": self.config.default_camera,
            "cameras": cameras,
        }

    def _servo_bus_diagnostics(self) -> dict[str, Any]:
        ids_by_role = {
            **{
                f"arm:{name}": [int(item) for item in arm.joint_ids.values()]
                for name, arm in self.config.arms.items()
            },
            "base": [int(item) for item in self.base.wheel_ids],
            "battery": [int(item) for item in self.config.battery.servo_ids],
        }
        configured_ids = sorted(
            {servo_id for ids in ids_by_role.values() for servo_id in ids}
        )
        items = []
        for servo_id in configured_ids:
            roles = sorted(role for role, ids in ids_by_role.items() if servo_id in ids)
            ping_ok = self.bus.ping(servo_id) if self.bus.connected else False
            state = (
                self.bus.read_state(servo_id)
                if self.bus.connected and ping_ok
                else None
            )
            items.append(
                {
                    "servo_id": servo_id,
                    "roles": roles,
                    "ping": ping_ok,
                    "position": state.position if state is not None else None,
                    "voltage": state.voltage if state is not None else None,
                    "temperature": state.temperature if state is not None else None,
                    "moving": state.moving if state is not None else None,
                }
            )
        missing: list[int] = [
            servo_id
            for servo_id, item in zip(configured_ids, items, strict=False)
            if not item["ping"]
        ]
        voltage_unavailable: list[int] = [
            servo_id
            for servo_id, item in zip(configured_ids, items, strict=False)
            if item["voltage"] is None
        ]
        return {
            "ok": self.bus.connected and not missing,
            "connected": self.bus.connected,
            "port": self.config.serial_bus.port,
            "baudrate": self.config.serial_bus.baudrate,
            "configured_ids": configured_ids,
            "ids_by_role": ids_by_role,
            "servos": items,
            "missing_or_unresponsive_ids": missing,
            "voltage_unavailable_ids": voltage_unavailable,
            "issue": _servo_bus_issue(self.bus.connected, missing),
        }

    def _pulse_velocity(
        self, *, vx: float, vz: float, duration: float, vy: float = 0.0
    ) -> dict[str, Any]:
        start = time.time()
        last_response: dict[str, Any] = {"success": True, "message": "not started"}
        iterations: list[dict[str, Any]] = []
        while time.time() - start < duration:
            last_response = self.set_velocity(vx, vz, vy=vy)
            iterations.append(
                {
                    "index": len(iterations) + 1,
                    "elapsed_sec": round(time.time() - start, 3),
                    "success": bool(last_response.get("success", False)),
                    "message": last_response.get("message"),
                    "control": last_response.get("control"),
                }
            )
            if not bool(last_response.get("success", False)):
                self.stop()
                response = {
                    **last_response,
                    "motion_trace": {
                        "kind": "pulse_velocity",
                        "requested": {"vx": vx, "vy": vy, "vz": vz},
                        "duration_sec": duration,
                        "iterations": iterations,
                        "stop_reason": "write_failed",
                    },
                }
                self._remember_motion_report(response["motion_trace"])
                return response
            time.sleep(0.2)
        stop_response = self.stop()
        response = {
            "success": bool(stop_response.get("success", False)),
            "message": "motion completed"
            if stop_response.get("success", False)
            else "motion stop failed",
            "last_motion_response": last_response,
            "stop_response": stop_response,
            "motion_trace": {
                "kind": "pulse_velocity",
                "requested": {"vx": vx, "vy": vy, "vz": vz},
                "duration_sec": duration,
                "iterations": iterations,
                "stop_reason": "completed"
                if stop_response.get("success", False)
                else "stop_failed",
            },
        }
        self._remember_motion_report(response["motion_trace"])
        return response

    def base_control_diagnostics(self) -> dict[str, Any]:
        return {
            "last_motion_report": dict(self._last_motion_report),
            "base": self.base.control_diagnostics(),
        }

    def _remember_motion_report(self, report: dict[str, Any]) -> None:
        self._last_motion_report = dict(report)

    def _arm(self, arm_name: str | None) -> SO101Arm:
        return self.arms[resolved_arm_name(arm_name, self.config.default_arm)]

    def _arm_service_results(
        self, method_name: str, *, bus_ok: bool
    ) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for name, arm in self.arms.items():
            if bus_ok:
                results[name] = getattr(arm, method_name)()
            else:
                results[name] = {
                    "success": False,
                    "message": "servo bus connection failed",
                }
        return results

    def _camera_service_results(self, method_name: str) -> dict[str, dict[str, Any]]:
        return {
            name: getattr(camera, method_name)()
            for name, camera in self.cameras.items()
        }

    @staticmethod
    def _tag_arm_response(response: dict[str, Any], arm_name: str) -> dict[str, Any]:
        return {**dict(response), "arm_name": arm_name}


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


def _multi_service_probe(responses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    items = {
        name: _service_probe(response, joint_count=_joint_count(response))
        for name, response in responses.items()
    }
    default_name = next(iter(items), "arm")
    default = items.get(default_name, {})
    return {
        **dict(default),
        "ok": all(bool(item.get("ok")) for item in items.values()),
        "default_arm": default_name,
        "arms": items,
        "joint_count": sum(
            int(item.get("joint_count", 0) or 0) for item in items.values()
        ),
    }


def _service_diagnostic(
    startup_probe: dict[str, Any] | None,
    status_response: dict[str, Any],
    *,
    joint_count: int | None = None,
) -> dict[str, Any]:
    startup = startup_probe if isinstance(startup_probe, dict) else None
    status = _service_probe(status_response, joint_count=joint_count)
    ok = (
        bool(startup.get("ok")) and bool(status.get("ok"))
        if startup is not None
        else bool(status.get("ok"))
    )
    issue = None if ok else _first_issue(startup, status)
    diagnostic = {
        "ok": ok,
        "issue": issue,
        "startup_response": startup.get("response") if startup is not None else None,
        "status_response": status_response,
        "startup": startup,
        "status": status,
        "response": status_response,
    }
    if joint_count is not None:
        diagnostic["joint_count"] = joint_count
    return diagnostic


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


def _multi_camera_probe(responses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    items = {name: _camera_probe(response) for name, response in responses.items()}
    default_name = next(iter(items), "front")
    default = items.get(default_name, {})
    return {
        **dict(default),
        "ok": all(bool(item.get("ok")) for item in items.values()),
        "default_camera": default_name,
        "cameras": items,
    }


def _multi_service_diagnostic(
    startup_probe: dict[str, Any] | None,
    status_response: dict[str, Any],
) -> dict[str, Any]:
    startup_arms = dict((startup_probe or {}).get("arms", {}) or {})
    status_arms = dict(status_response.get("arms", {}) or {})
    diagnostics = {
        name: _service_diagnostic(
            startup_arms.get(name), response, joint_count=_joint_count(response)
        )
        for name, response in status_arms.items()
    }
    default_name = str(
        status_response.get("default_arm") or next(iter(diagnostics), "arm")
    )
    default = diagnostics.get(default_name, {})
    return {
        **dict(default),
        "ok": all(bool(item.get("ok")) for item in diagnostics.values()),
        "default_arm": default_name,
        "arms": diagnostics,
        "joint_count": sum(
            int(item.get("joint_count", 0) or 0) for item in diagnostics.values()
        ),
    }


def _multi_camera_diagnostic(
    startup_probe: dict[str, Any] | None,
    status_response: dict[str, Any],
    *,
    default_camera: str,
    frame_id: int | None,
    image: np.ndarray | None,
    timeout_ms: int,
) -> dict[str, Any]:
    startup_cameras = dict((startup_probe or {}).get("cameras", {}) or {})
    status_cameras = dict(status_response.get("cameras", {}) or {})
    diagnostics: dict[str, Any] = {}
    for name, response in status_cameras.items():
        owner = str(response.get("owner", "robot_driver"))
        frame_available = (
            bool(response.get("opened"))
            if owner != "camera_service"
            else bool(response.get("success"))
        )
        diagnostics[name] = {
            "ok": bool(response.get("success")) or owner == "camera_service",
            "frame_available": frame_available,
            "frame_id": frame_id if name == default_camera else None,
            "image_shape": list(image.shape)
            if image is not None and name == default_camera
            else None,
            "timeout_ms": timeout_ms,
            "issue": None
            if (bool(response.get("success")) or owner == "camera_service")
            else response.get("error") or "no native camera frame available",
            "owner": owner,
            "response": response,
            "startup": startup_cameras.get(name),
        }
    default = diagnostics.get(default_camera, {})
    return {
        **dict(default),
        "ok": all(bool(item.get("ok")) for item in diagnostics.values()),
        "default_camera": default_camera,
        "cameras": diagnostics,
    }


def resolved_arm_name(arm_name: str | None, default_arm: str) -> str:
    target = str(arm_name or "").strip()
    return target or default_arm


def _joint_count(response: dict[str, Any]) -> int:
    joint_states = response.get("joint_states")
    return len(joint_states) if isinstance(joint_states, dict) else 0


def _first_issue(*items: dict[str, Any] | None) -> str:
    for item in items:
        if not isinstance(item, dict):
            continue
        issue = item.get("issue")
        if issue:
            return str(issue)
        response = item.get("response")
        if isinstance(response, dict) and response.get("message"):
            return str(response["message"])
    return "service failed"


def _servo_bus_issue(connected: bool, missing: list[int]) -> str | None:
    if not connected:
        return "servo bus is not connected"
    if missing:
        return f"unresponsive servo ids: {missing}"
    return None
