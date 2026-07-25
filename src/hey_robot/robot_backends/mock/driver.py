from __future__ import annotations

import asyncio
import math
import time
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from hey_robot.protocol import (
    Envelope,
    RobotAction,
    RobotSkillAction,
    RobotSkillResult,
    RobotStatus,
)
from hey_robot.robot_api import (
    DriverObservation,
    ObservationAsset,
    RobotCapabilities,
    RobotDriverContext,
    RobotHealth,
)
from hey_robot.robot_api.primitives import SUPPORTED_CLASSIC_PRIMITIVES
from hey_robot.robot_runtime.skill_gate import SkillAdmissionGate


class _MockRobotDriverBase:
    if TYPE_CHECKING:

        def before_perception_request(
            self, skill_name: str, arguments: dict[str, Any] | None = None
        ) -> None: ...

        def readiness(self, *, attempt_index: int | None = None) -> dict[str, Any]: ...

        def _world_from_settings(self) -> dict[str, Any]: ...

        def _world_snapshot(self) -> dict[str, Any]: ...

        def _scene_summary(self) -> dict[str, Any]: ...

        def _object(self, name: str) -> dict[str, Any] | None: ...

        def _object_visible(self, name: str) -> bool: ...

        def _nearest_graspable_object(self) -> str | None: ...

        def _release_held_object(self, location: str) -> None: ...

        def _record_world_event(self, kind: str, **payload: Any) -> None: ...

        def _scripted_failure(
            self, skill: RobotSkillAction
        ) -> RobotSkillResult | None: ...

        def _drain_battery_for_skill(self, skill: RobotSkillAction) -> None: ...

        def _front_view(self) -> np.ndarray: ...

        def _left_wrist_view(self) -> np.ndarray: ...

        def _right_wrist_view(self) -> np.ndarray: ...

        def _default_camera(self) -> str: ...

        def _build_camera_status_map(
            self,
            *,
            frame_id: int | None,
            camera_ok: bool,
            image_shape: list[int] | None,
            drop_reason: str | None,
        ) -> dict[str, Any]: ...

        def _camera_names(self) -> tuple[str, ...]: ...

        def _camera_available_for_observe(self, observe_count: int) -> bool: ...

        def _camera_drop_reason(self, observe_count: int) -> str | None: ...

        def _metrics(self) -> dict[str, Any]: ...

        def _diagnostics(self) -> dict[str, Any]: ...

        def _arm_status(self) -> dict[str, Any]: ...

        def _battery_status(self) -> dict[str, Any]: ...

        def _has_active_readiness_fault(
            self, *, resource: str | None = None, attempt_index: int
        ) -> bool: ...

        def _pose_or_none(self, pose_name: str) -> dict[str, float] | None: ...

        def _named_pose(self, pose_name: str) -> dict[str, float]: ...

        @staticmethod
        def _diagnostics_ready(diagnostics: dict[str, Any]) -> bool: ...

        @staticmethod
        def _diagnostic_failure_summary(diagnostics: dict[str, Any]) -> str: ...

    _JOINT_LIMITS: ClassVar[dict[str, tuple[float, float]]] = {
        "shoulder_pan": (-180.0, 180.0),
        "shoulder_lift": (-90.0, 120.0),
        "elbow_flex": (-120.0, 120.0),
        "wrist_flex": (-120.0, 120.0),
        "wrist_roll": (-180.0, 180.0),
        "gripper": (0.0, 100.0),
    }
    _DEFAULT_WORLD: ClassVar[dict[str, Any]] = {
        "robot_near": "front_workspace",
        "locations": {
            "front_workspace": {"x_cm": 80.0, "y_cm": 0.0, "label": "front workspace"},
            "table": {"x_cm": 100.0, "y_cm": 20.0, "label": "table"},
            "bin": {"x_cm": 120.0, "y_cm": -30.0, "label": "bin"},
            "shelf": {"x_cm": 160.0, "y_cm": 45.0, "label": "shelf"},
        },
        "objects": {
            "mock_object": {
                "label": "mock object",
                "location": "front_workspace",
                "visible": True,
                "graspable": True,
                "color": [210, 80, 70],
            },
            "cup": {
                "label": "cup",
                "location": "table",
                "visible": True,
                "graspable": True,
                "color": [50, 120, 210],
            },
            "block": {
                "label": "block",
                "location": "shelf",
                "visible": True,
                "graspable": True,
                "color": [230, 175, 45],
            },
        },
    }
    _DEFAULT_CAMERA_NAMES: ClassVar[tuple[str, ...]] = (
        "front",
        "left_wrist",
        "right_wrist",
    )

    def __init__(self, context: RobotDriverContext) -> None:
        self.context = context
        self.robot_id = context.robot_id
        self.settings = dict(context.settings or {})
        self.contracts = SkillAdmissionGate(context.action_specs)
        self.frame_id = 0
        self.observe_count = 0
        self.action_attempts = 0
        self.state = "created"
        self.last_error: str | None = None
        self.last_skill_result: RobotSkillResult | None = None
        self.base_pose = {"x_cm": 0.0, "y_cm": 0.0, "yaw_deg": 0.0}
        self.base_velocity = {"vx": 0.0, "vy": 0.0, "vz": 0.0}
        self.arm_joints = dict(self._named_pose("home"))
        self.gripper_opening_pct = 80.0
        self.object_held: str | None = None
        self.world = self._world_from_settings()
        self.world_events: list[dict[str, Any]] = []
        self.skill_counts: dict[str, int] = {}
        self.scan_counts: dict[str, int] = {}
        self.battery_percentage = float(self.settings.get("battery_percentage", 85.0))
        self.last_camera: dict[str, Any] = {
            "ok": bool(self.settings.get("camera_available", True)),
            "frame_available": bool(self.settings.get("camera_available", True)),
            "frame_id": None,
            "image_shape": None,
        }
        self.last_cameras_status: dict[str, Any] = self._build_camera_status_map(
            frame_id=None,
            camera_ok=bool(self.settings.get("camera_available", True)),
            image_shape=None,
            drop_reason=None,
        )
        self.last_battery = self._battery_status()
        self.last_arm_status = self._arm_status()
        self.startup_diagnostics: dict[str, Any] = {}
        self.base_control: dict[str, Any] = {
            "last_motion_report": None,
            "last_stop_command": None,
            "emergency_stop_active": False,
        }
        self._hardware_summary = {
            "serial_port": self.settings.get("serial_port", "MOCK"),
            "baudrate": int(self.settings.get("baudrate", 1000000)),
            "camera_device_id": self.settings.get("camera_device_id", "mock_front"),
            "video_timeout_ms": int(self.settings.get("video_timeout_ms", 500)),
            "base_type": "mock_lekiwi_base",
            "base_wheel_ids": [7, 8, 9],
            "arm_type": "mock_so101_arm",
            "arm_joint_ids": {
                "shoulder_pan": 1,
                "shoulder_lift": 2,
                "elbow_flex": 3,
                "wrist_flex": 4,
                "wrist_roll": 5,
                "gripper": 6,
            },
            "battery_servo_ids": [7, 8, 9],
        }

    async def start(self) -> None:
        self.startup_diagnostics = self._diagnostics()
        self.last_battery = self._battery_status()
        self.last_arm_status = self._arm_status()
        self.state = (
            "idle" if self._diagnostics_ready(self.startup_diagnostics) else "degraded"
        )
        self.last_error = (
            None
            if self.state == "idle"
            else self._diagnostic_failure_summary(self.startup_diagnostics)
        )

    async def capabilities(self) -> RobotCapabilities:
        return RobotCapabilities(
            robot_id=self.robot_id,
            driver_type="mock",
            action_dimensions=None,
            control_hz=float(self.settings.get("control_hz", 2.0)),
            cameras=list(self._camera_names()),
            observation_modalities=["image", "arm_state", "status"],
            supports_reset=True,
            supports_interrupt=True,
            metadata={
                "body": "xlerobot",
                "robot_family": self.context.robot_family,
                "environment": self.context.environment,
                "driver_kind": self.context.driver_kind,
                "embodiment_profile": (
                    self.context.embodiment.name if self.context.embodiment else None
                ),
                "control": "skill_action",
                "runtime": "mock_xlerobot",
                "supported_skills": list(SUPPORTED_CLASSIC_PRIMITIVES),
                "safety": dict(self.settings.get("safety", {}) or {}),
            },
        )

    async def health(self) -> RobotHealth:
        return RobotHealth(
            robot_id=self.robot_id,
            online=self.state != "closed",
            state=self.state,
            frame_id=self.frame_id,
            error=self.last_error,
            metrics=self._metrics(),
        )

    async def observe(self) -> DriverObservation:
        self.observe_count += 1
        self.frame_id += 1
        camera_ok = self._camera_available_for_observe(self.observe_count)
        self.last_cameras_status = self._build_camera_status_map(
            frame_id=self.frame_id,
            camera_ok=camera_ok,
            image_shape=[160, 240, 3] if camera_ok else None,
            drop_reason=self._camera_drop_reason(self.observe_count),
        )
        self.last_camera = {
            **dict(self.last_cameras_status.get(self._default_camera(), {})),
            "default_camera": self._default_camera(),
        }
        self.last_arm_status = self._arm_status()
        self.last_battery = self._battery_status()
        assets: list[ObservationAsset] = []
        if camera_ok:
            frames = {
                "front": self._front_view(),
                "left_wrist": self._left_wrist_view(),
                "right_wrist": self._right_wrist_view(),
            }
            assets.extend(
                ObservationAsset(
                    kind="image",
                    role="camera",
                    name=name,
                    data=frames.get(name, self._front_view()),
                    metadata={
                        "driver": "mock",
                        "body": "xlerobot",
                        "camera_role": name,
                    },
                )
                for name in self._camera_names()
            )
        return DriverObservation(
            envelope=self._envelope(),
            frame_id=self.frame_id,
            assets=assets,
            proprioception=self._proprioception(),
            metadata={
                "driver": "mock",
                "body": "xlerobot",
                "robot_family": self.context.robot_family,
                "environment": self.context.environment,
                "embodiment_profile": (
                    self.context.embodiment.name if self.context.embodiment else None
                ),
                "state": self.state,
                "camera": self.last_camera,
                "cameras": self.last_cameras_status,
                "arm_status": self.last_arm_status,
                "battery": self.last_battery,
                "base_pose": dict(self.base_pose),
                "base_velocity": dict(self.base_velocity),
                "object_held": self.object_held,
                "scene": self._scene_summary(),
                "world": self._world_snapshot(),
                "startup_diagnostics": self.startup_diagnostics,
                "last_skill_result": self.last_skill_result.to_dict()
                if self.last_skill_result
                else None,
                "readiness": self.readiness(),
            },
        )

    async def status(self) -> RobotStatus:
        return RobotStatus(
            envelope=self._envelope(),
            frame_id=self.frame_id,
            state=self.state,  # type: ignore[arg-type]
            success=None,
            error=self.last_error,
            metrics=self._metrics(),
        )

    async def apply_action(self, action: RobotAction) -> RobotStatus:
        await self._action_latency()
        try:
            skill = RobotSkillAction.from_robot_action(action)
        except ValueError as exc:
            result = RobotSkillResult(
                False,
                str(exc),
                {"failure_mode": "invalid_action", "values": list(action.values)},
            )
            self.last_skill_result = result
            self.state = "failed"
            self.last_error = result.message
            return self._status_for_action(action, success=False)

        attempt_index = self.action_attempts + 1
        transient_fault_active = self._has_active_readiness_fault(
            attempt_index=attempt_index
        )
        _, decision = self.contracts.validate_action(
            skill,
            robot_type="xlerobot",
            status=await self.status(),
            readiness=self.readiness(attempt_index=attempt_index),
        )
        self.action_attempts = attempt_index
        if not decision.allowed:
            result = RobotSkillResult(
                False,
                decision.reason,
                {
                    "skill": skill.to_dict(),
                    "failure_mode": decision.failure_mode,
                    "contract_decision": decision.metadata,
                },
            )
        elif skill.name in set(self.settings.get("fail_skills", []) or []):
            self.skill_counts[skill.name] = self.skill_counts.get(skill.name, 0) + 1
            result = RobotSkillResult(
                False,
                f"mock injected failure for {skill.name}",
                {"skill": skill.to_dict(), "failure_mode": "injected_failure"},
            )
        else:
            self.skill_counts[skill.name] = self.skill_counts.get(skill.name, 0) + 1
            scripted = self._scripted_failure(skill)
            result = scripted if scripted is not None else self._execute_skill(skill)

        self.last_skill_result = result
        failure_mode = str(result.data.get("failure_mode") or "").strip().lower()
        if result.success:
            self.state = "skill_completed"
            self.last_error = None
            self._drain_battery_for_skill(skill)
        elif failure_mode == "readiness_failed" and transient_fault_active:
            self.state = "idle"
            self.last_error = result.message or "resource not ready"
        else:
            self.state = "failed"
            self.last_error = result.message or "skill failed"
        return self._status_for_action(action, success=result.success)

    async def reset(self) -> RobotStatus:
        self.frame_id = 0
        self.observe_count = 0
        self.action_attempts = 0
        self.base_pose = {"x_cm": 0.0, "y_cm": 0.0, "yaw_deg": 0.0}
        self.base_velocity = {"vx": 0.0, "vy": 0.0, "vz": 0.0}
        self.arm_joints = dict(self._named_pose("home"))
        self.gripper_opening_pct = 80.0
        self.object_held = None
        self.world = self._world_from_settings()
        self.world_events = []
        self.skill_counts = {}
        self.scan_counts = {}
        self.battery_percentage = float(self.settings.get("battery_percentage", 85.0))
        self.last_skill_result = RobotSkillResult(
            True, "mock reset", {"skill": "reset"}
        )
        self.last_cameras_status = self._build_camera_status_map(
            frame_id=None,
            camera_ok=bool(self.settings.get("camera_available", True)),
            image_shape=None,
            drop_reason=None,
        )
        self.last_camera = {
            **dict(self.last_cameras_status.get(self._default_camera(), {})),
            "default_camera": self._default_camera(),
        }
        self.last_error = None
        self.state = "idle"
        self.base_control = {
            "last_motion_report": None,
            "last_stop_command": None,
            "emergency_stop_active": False,
        }
        return await self.status()

    async def close(self) -> None:
        self.state = "closed"

    def _execute_skill(self, skill: RobotSkillAction) -> RobotSkillResult:
        name = skill.name
        args = dict(skill.arguments)
        if name == "stop_motion":
            self.base_velocity = {"vx": 0.0, "vy": 0.0, "vz": 0.0}
            self.base_control["last_stop_command"] = {
                "success": True,
                "emergency": bool(args.get("emergency", False)),
                "frame_id": self.frame_id,
            }
            if bool(args.get("emergency", False)):
                self.state = "emergency"
                self.base_control["emergency_stop_active"] = True
                self.base_control["last_motion_report"] = {
                    "kind": "emergency_stop",
                    "success": True,
                    "frame_id": self.frame_id,
                }
                return self._ok(skill, "emergency stop active")
            self.base_control["emergency_stop_active"] = False
            self.base_control["last_motion_report"] = {
                "kind": "stop_motion",
                "success": True,
                "frame_id": self.frame_id,
            }
            return self._ok(skill, "base stopped")
        if name == "move_base":
            distance = float(args["distance_cm"])
            direction = str(args.get("direction", "forward")).lower()
            if direction == "backward":
                return self._move(skill, -abs(distance), 0.0)
            if direction == "left":
                return self._move(skill, 0.0, abs(distance))
            if direction == "right":
                return self._move(skill, 0.0, -abs(distance))
            return self._move(skill, abs(distance), 0.0)
        if name == "turn_base":
            angle = float(args["angle_deg"])
            if str(args.get("direction", "left")).lower() == "right":
                angle = -abs(angle)
            self.base_pose["yaw_deg"] = self._wrap_yaw(
                self.base_pose["yaw_deg"] + angle
            )
            return self._ok(skill, f"base turned {args.get('direction', 'left')}")
        if name == "base_velocity_step":
            duration_sec = max(0.001, float(args.get("duration_ms", 250)) / 1000.0)
            vx = float(args.get("vx", 0.0))
            vy = float(args.get("vy", 0.0))
            wz = float(args.get("wz", 0.0))
            self.base_velocity = {"vx": vx, "vy": vy, "vz": wz}
            if abs(vx) > 0:
                self._move(skill, vx * duration_sec * 100.0, 0.0)
            if abs(wz) > 0:
                self.base_pose["yaw_deg"] = self._wrap_yaw(
                    self.base_pose["yaw_deg"]
                    + wz * duration_sec * 180.0 / 3.141592653589793
                )
            self.base_control["last_motion_report"] = {
                "kind": "base_velocity_step",
                "success": True,
                "frame_id": self.frame_id,
                "command": {"vx": vx, "vy": vy, "wz": wz},
                "duration_ms": int(args.get("duration_ms", 250)),
            }
            return self._ok(skill, "base velocity step completed")
        if name == "move_arm_joints":
            return self._set_joints(
                skill,
                dict(args["joints"]),
                absolute=str(args.get("mode", "absolute")) != "delta",
            )
        if name == "set_arm_pose":
            pose_name = str(args["pose_name"])
            pose = self._pose_or_none(pose_name)
            if pose is None:
                return self._fail(
                    skill, f"unknown named pose: {pose_name}", "unknown_pose"
                )
            self.arm_joints.update(pose)
            self.gripper_opening_pct = self.arm_joints["gripper"]
            return self._ok(skill, f"arm moved to {pose_name}")
        if name == "set_gripper":
            action = str(args.get("action", "")).lower()
            pct = (
                100.0
                if action == "open"
                else 0.0
                if action == "close"
                else self._clamp(float(args["opening_pct"]), 0.0, 100.0)
            )
            self.gripper_opening_pct = pct
            self.arm_joints["gripper"] = pct
            if pct > 60.0:
                self._release_held_object("front_workspace")
                self.object_held = None
            elif pct <= 5.0:
                target = str(
                    args.get("object")
                    or args.get("target")
                    or self._nearest_graspable_object()
                    or ""
                )
                obj = self._object(target) if target else None
                if obj is not None and self._object_visible(target):
                    self.object_held = target
                    obj["held"] = True
                    obj["visible"] = False
                    obj["location"] = "gripper"
                    self._record_world_event("grasped", object=target)
            return self._ok(skill, "gripper opening set")
        if name == "reset_posture":
            self.base_velocity = {"vx": 0.0, "vy": 0.0, "vz": 0.0}
            self.arm_joints = dict(self._named_pose("home"))
            self.gripper_opening_pct = self.arm_joints["gripper"]
            self.base_control["last_stop_command"] = {
                "success": True,
                "emergency": False,
                "frame_id": self.frame_id,
            }
            return self._ok(skill, "robot reset posture")
        if name in {
            "inspect_scene",
            "look_around",
            "detect_marker",
        }:
            return self._ok(
                skill, "mock perception available", {"scene": self._scene_summary()}
            )
        return self._fail(
            skill, f"unsupported mock xlerobot skill: {name}", "unknown_skill"
        )

    def _move(
        self, skill: RobotSkillAction, forward_cm: float, left_cm: float
    ) -> RobotSkillResult:
        yaw = math.radians(self.base_pose["yaw_deg"])
        dx = forward_cm * math.cos(yaw) - left_cm * math.sin(yaw)
        dy = forward_cm * math.sin(yaw) + left_cm * math.cos(yaw)
        self.base_pose["x_cm"] += dx
        self.base_pose["y_cm"] += dy
        self.base_control["last_motion_report"] = {
            "kind": skill.name,
            "success": True,
            "forward_cm": forward_cm,
            "left_cm": left_cm,
            "base_pose": dict(self.base_pose),
            "frame_id": self.frame_id,
        }
        self._record_world_event("base_moved", base_pose=dict(self.base_pose))
        return self._ok(skill, "base moved", {"base_pose": dict(self.base_pose)})

    def _set_joints(
        self, skill: RobotSkillAction, values: dict[str, Any], *, absolute: bool
    ) -> RobotSkillResult:
        next_values = dict(self.arm_joints)
        for joint, value in values.items():
            name = str(joint)
            next_values[name] = (
                float(value) if absolute else next_values.get(name, 0.0) + float(value)
            )
            if name not in self._JOINT_LIMITS:
                return self._fail(skill, f"unknown joint: {name}", "invalid_joint")
            low, high = self._JOINT_LIMITS[name]
            if next_values[name] < low or next_values[name] > high:
                return self._fail(
                    skill, f"joint {name} outside limit [{low}, {high}]", "joint_limit"
                )
        self.arm_joints.update(next_values)
        self.gripper_opening_pct = self.arm_joints["gripper"]
        return self._ok(skill, "joints set", {"joint_states": dict(self.arm_joints)})

    def _proprioception(self) -> list[float]:
        return [
            self.base_pose["x_cm"],
            self.base_pose["y_cm"],
            self.base_pose["yaw_deg"],
            self.base_velocity["vx"],
            self.base_velocity["vy"],
            self.base_velocity["vz"],
            *(self.arm_joints[joint] for joint in self._JOINT_LIMITS),
        ]

    def _status_for_action(self, action: RobotAction, *, success: bool) -> RobotStatus:
        return RobotStatus(
            envelope=self._envelope(trace_id=action.envelope.trace_id),
            frame_id=self.frame_id,
            state=self.state,  # type: ignore[arg-type]
            skill_id=action.skill_id,
            success=success,
            error=None if success else self.last_error,
            metrics=self._metrics(),
        )

    def _ok(
        self, skill: RobotSkillAction, message: str, data: dict[str, Any] | None = None
    ) -> RobotSkillResult:
        return RobotSkillResult(
            True, message, {"skill": skill.to_dict(), **(data or {})}
        )

    def _fail(
        self, skill: RobotSkillAction, message: str, failure_mode: str
    ) -> RobotSkillResult:
        return RobotSkillResult(
            False, message, {"skill": skill.to_dict(), "failure_mode": failure_mode}
        )

    async def _action_latency(self) -> None:
        latency_ms = int(self.settings.get("action_latency_ms", 0))
        if latency_ms > 0:
            await asyncio.sleep(latency_ms / 1000.0)

    def _envelope(self, *, trace_id: str | None = None) -> Envelope:
        return Envelope(
            trace_id=trace_id
            or f"mock_xlerobot_{self.robot_id}_{int(time.time() * 1000)}",
            robot_id=self.robot_id,
            deployment_id=self.context.deployment_id,
        )

    @staticmethod
    def _wrap_yaw(value: float) -> float:
        return ((value + 180.0) % 360.0) - 180.0

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))
