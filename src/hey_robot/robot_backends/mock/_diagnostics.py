from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hey_robot.protocol import RobotSkillResult
from hey_robot.robot_api import RobotDriverContext


class _DiagnosticsMixin:
    # ── attributes from _MockRobotDriverBase.__init__ ──────────────────────
    _hardware_summary: dict[str, Any]
    startup_diagnostics: dict[str, Any]
    last_camera: dict[str, Any]
    base_control: dict[str, Any]
    base_pose: dict[str, float]
    base_velocity: dict[str, float]
    object_held: str | None
    last_skill_result: RobotSkillResult | None
    settings: dict[str, Any]
    arm_joints: dict[str, float]
    frame_id: int
    battery_percentage: float
    gripper_opening_pct: float
    context: RobotDriverContext
    last_battery: dict[str, Any]
    last_arm_status: dict[str, Any]

    if TYPE_CHECKING:

        def _scene_summary(self) -> dict[str, Any]: ...

        def _world_snapshot(self) -> dict[str, Any]: ...

        def readiness(self, *, attempt_index: int | None = None) -> dict[str, Any]: ...

    def _metrics(self) -> dict[str, Any]:
        self.last_battery = self._battery_status()
        self.last_arm_status = self._arm_status()
        return {
            "driver": "mock",
            "body": "xlerobot",
            "runtime": "mock_xlerobot",
            "hardware": self._hardware_summary,
            "startup_diagnostics": self.startup_diagnostics,
            "camera": self.last_camera,
            "arm_status": self.last_arm_status,
            "battery": self.last_battery,
            "base_control": dict(self.base_control),
            "base_pose": dict(self.base_pose),
            "base_velocity": dict(self.base_velocity),
            "object_held": self.object_held,
            "scene": self._scene_summary(),
            "world": self._world_snapshot(),
            "last_skill_result": self.last_skill_result.to_dict()
            if self.last_skill_result
            else None,
            "readiness": self.readiness(),
            **dict(self.settings.get("safety", {}) or {}),
        }

    def _diagnostics(self) -> dict[str, Any]:
        return {
            "bus": {
                "ok": True,
                "port": "MOCK",
                "baudrate": self._hardware_summary["baudrate"],
                "message": "mock bus",
            },
            "servo_bus": {
                "ok": True,
                "configured_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                "missing_or_unresponsive_ids": [],
                "voltage_unavailable_ids": [],
                "servos": [
                    {
                        "servo_id": servo_id,
                        "roles": ["mock"],
                        "ping": True,
                        "voltage": self._battery_voltage(),
                    }
                    for servo_id in [1, 2, 3, 4, 5, 6, 7, 8, 9]
                ],
            },
            "base": {
                "ok": bool(self.settings.get("base_available", True)),
                "response": {
                    "success": bool(self.settings.get("base_available", True)),
                    "message": "mock base ready"
                    if bool(self.settings.get("base_available", True))
                    else "mock base unavailable",
                },
            },
            "arm": {
                "ok": bool(self.settings.get("arm_available", True)),
                "joint_count": len(self.arm_joints),
                "lift_height": 0.0,
                "response": {
                    "success": bool(self.settings.get("arm_available", True)),
                    "message": "mock arm ready"
                    if bool(self.settings.get("arm_available", True))
                    else "mock arm unavailable",
                },
                "status_response": self._arm_status(),
            },
            "camera": {
                "ok": bool(self.settings.get("camera_available", True)),
                "frame_available": bool(self.settings.get("camera_available", True)),
                "frame_id": self.frame_id,
                "jpeg_bytes": 0,
                "timeout_ms": self._hardware_summary["video_timeout_ms"],
            },
            "battery": self._battery_status(),
            "safety": {
                "emergency_stop": bool(self.settings.get("emergency_stop", False))
            },
        }

    def _arm_status(self) -> dict[str, Any]:
        ok = bool(self.settings.get("arm_available", True))
        return {
            "success": ok,
            "enabled": ok,
            "initialized": ok,
            "message": "mock arm ready" if ok else "mock arm unavailable",
            "joint_states": dict(self.arm_joints),
            "joint_count": len(self.arm_joints),
            "lift_height": 0.0,
            "gripper_opening_pct": self.gripper_opening_pct,
        }

    def _battery_status(self) -> dict[str, Any]:
        status_override = self.settings.get("battery_status")
        if status_override is None:
            if self.battery_percentage <= 5.0:
                status = "critical"
            elif self.battery_percentage <= 20.0:
                status = "low"
            else:
                status = "normal"
        else:
            status = str(status_override).lower()
        percentage_by_status = {
            "normal": 85.0,
            "low": 18.0,
            "critical": 4.0,
            "unknown": None,
        }
        voltage_by_status = {
            "normal": 12.0,
            "low": 10.7,
            "critical": 9.6,
            "unknown": None,
        }
        percentage = (
            self.settings.get(
                "battery_percentage", percentage_by_status.get(status, 85.0)
            )
            if status_override is not None
            else self.battery_percentage
        )
        voltage = self.settings.get(
            "battery_voltage", voltage_by_status.get(status, 12.0)
        )
        return {
            "ok": status not in {"critical", "unknown"},
            "status": status,
            "voltage": None if voltage is None else float(voltage),
            "percentage": None if percentage is None else float(percentage),
            "servo_id": 7,
        }

    def _battery_voltage(self) -> float:
        return float(self._battery_status().get("voltage") or 0.0)

    def _resource_available(self, resource: str, *, attempt_index: int) -> bool:
        base_value = bool(self.settings.get(f"{resource}_available", True))
        if not base_value:
            return False
        return not self._has_active_readiness_fault(
            resource=resource, attempt_index=attempt_index
        )

    def _has_active_readiness_fault(
        self, *, resource: str | None = None, attempt_index: int
    ) -> bool:
        faults = self.settings.get("readiness_faults") or []
        if not isinstance(faults, list):
            return False
        for item in faults:
            if not isinstance(item, dict):
                continue
            fault_resource = str(item.get("resource") or "").strip().lower()
            if resource is not None and fault_resource != resource:
                continue
            until_attempt = int(item.get("until_attempt", 0) or 0)
            if until_attempt > 0 and attempt_index <= until_attempt:
                return True
        return False

    def _readiness_resources(self) -> tuple[str, ...]:
        if self.context.embodiment and self.context.embodiment.readiness_resources:
            return self.context.embodiment.readiness_resources
        return ("base", "arm", "gripper", "camera")

    def _pose_or_none(self, pose_name: str) -> dict[str, float] | None:
        if self.context.embodiment:
            pose = self.context.embodiment.named_pose(pose_name)
            if pose is not None:
                return pose
        return None

    def _named_pose(self, pose_name: str) -> dict[str, float]:
        pose = self._pose_or_none(pose_name)
        if pose is not None:
            return pose
        raise KeyError(f"unknown embodiment pose: {pose_name}")

    @staticmethod
    def _diagnostics_ready(diagnostics: dict[str, Any]) -> bool:
        return all(
            bool((diagnostics.get(service) or {}).get("ok"))
            for service in ("base", "arm", "camera")
        )

    @staticmethod
    def _diagnostic_failure_summary(diagnostics: dict[str, Any]) -> str:
        failed = []
        for service in ("base", "arm", "camera"):
            item = diagnostics.get(service) or {}
            if not bool(item.get("ok")):
                failed.append(f"{service}: {item.get('issue') or 'not ready'}")
        return "; ".join(failed) if failed else "unknown mock diagnostic failure"
