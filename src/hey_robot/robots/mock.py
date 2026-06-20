from __future__ import annotations

import asyncio
import math
import time
from typing import Any, ClassVar

import numpy as np

from hey_robot.perception import DriverObservation, ObservationAsset
from hey_robot.protocol import Envelope, RobotAction, RobotStatus
from hey_robot.robots.base import RobotCapabilities, RobotDriverContext, RobotHealth
from hey_robot.robots.classic.primitives import SUPPORTED_CLASSIC_PRIMITIVES
from hey_robot.skills import RobotSkillAction, RobotSkillResult
from hey_robot.skills.contracts import SkillContractRuntime


class MockRobotDriver:
    """Deterministic xlerobot simulator for system validation without hardware."""

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
        self.settings = dict(context.spec.settings or {})
        self.contracts = SkillContractRuntime(context.skill_catalog)
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
                "robot_family": self.context.spec.robot_family,
                "environment": self.context.spec.robot_environment,
                "driver_kind": self.context.spec.driver_kind,
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
                "robot_family": self.context.spec.robot_family,
                "environment": self.context.spec.robot_environment,
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
            state=self.state,
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

    def before_perception_request(
        self, skill_name: str, arguments: dict[str, Any] | None = None
    ) -> None:
        if skill_name not in {
            "inspect_scene",
            "look_around",
            "detect_marker",
            "human_follow",
        }:
            return
        args = dict(arguments or {})
        target = _target_from_task(str(args.get("question") or args.get("task") or ""))
        if not target:
            return
        self.scan_counts[target] = self.scan_counts.get(target, 0) + 1
        self._reveal_after_scan(target)
        self._record_world_event(
            "perception_scan",
            object=target,
            scans=self.scan_counts[target],
        )

    def readiness(self, *, attempt_index: int | None = None) -> dict[str, Any]:
        resolved_attempt = (
            self.action_attempts if attempt_index is None else attempt_index
        )
        readiness: dict[str, Any] = {
            "robot": self.state != "closed",
            "battery": self._battery_status(),
            "emergency_stop": bool(self.settings.get("emergency_stop", False))
            or bool((self.settings.get("safety") or {}).get("estop", False)),
        }
        for resource in self._readiness_resources():
            readiness[resource] = {
                "ok": self._resource_available(resource, attempt_index=resolved_attempt)
            }
        for camera_name, status in self.last_cameras_status.items():
            readiness.setdefault(
                f"{camera_name}_camera",
                {"ok": bool(status.get("ok")), "owner": "mock"},
            )
        return readiness

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
            if str(args.get("direction", "forward")).lower() == "backward":
                distance = -abs(distance)
            return self._move(skill, distance, 0.0)
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
            "human_follow",
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

    def _set_joint(
        self, skill: RobotSkillAction, joint: str, angle: float
    ) -> RobotSkillResult:
        if joint not in self._JOINT_LIMITS:
            return self._fail(skill, f"unknown joint: {joint}", "invalid_joint")
        low, high = self._JOINT_LIMITS[joint]
        if angle < low or angle > high:
            return self._fail(
                skill, f"joint {joint} outside limit [{low}, {high}]", "joint_limit"
            )
        self.arm_joints[joint] = angle
        if joint == "gripper":
            self.gripper_opening_pct = angle
        return self._ok(skill, "joint set", {"joint": joint, "angle": angle})

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

    def _mock_vla_grasp(self, skill: RobotSkillAction, target: str) -> RobotSkillResult:
        obj = self._object(target)
        if obj is None:
            return self._fail(skill, f"object not found: {target}", "target_not_found")
        if not self._object_visible(target):
            return self._fail(
                skill, f"object not visible: {target}", "target_not_found"
            )
        if not bool(obj.get("graspable", True)):
            return self._fail(
                skill, f"object is not graspable: {target}", "not_graspable"
            )
        self.arm_joints.update(self._named_pose("pregrasp"))
        self.gripper_opening_pct = 0.0
        self.arm_joints["gripper"] = 0.0
        self.object_held = target
        obj["held"] = True
        obj["visible"] = False
        obj["location"] = "gripper"
        self._record_world_event("picked", object=target)
        return self._ok(
            skill,
            "object picked",
            {"object_held": self.object_held, "world": self._world_snapshot()},
        )

    def _mock_vla_place(
        self, skill: RobotSkillAction, location: str
    ) -> RobotSkillResult:
        target = self.object_held
        if not target:
            return self._fail(skill, "no held object to place", "no_held_object")
        obj = self._object(target)
        if obj is None:
            self.object_held = None
            return self._fail(
                skill, f"held object missing from world: {target}", "target_not_found"
            )
        obj["held"] = False
        obj["visible"] = location != "bin"
        obj["location"] = location
        self.object_held = None
        self.gripper_opening_pct = 80.0
        self.arm_joints["gripper"] = 80.0
        self._record_world_event("placed", object=target, location=location)
        return self._ok(
            skill,
            f"object placed in {location}",
            {"object": target, "location": location, "world": self._world_snapshot()},
        )

    def _world_from_settings(self) -> dict[str, Any]:
        source = self.settings.get("world") or self.settings.get("mock_world") or {}
        world = {
            "robot_near": self._DEFAULT_WORLD["robot_near"],
            "locations": {
                name: dict(value)
                for name, value in dict(self._DEFAULT_WORLD["locations"]).items()
            },
            "objects": {
                name: dict(value)
                for name, value in dict(self._DEFAULT_WORLD["objects"]).items()
            },
        }
        if isinstance(source, dict):
            if isinstance(source.get("locations"), dict):
                for name, value in source["locations"].items():
                    world["locations"][str(name)] = dict(value or {})
            if isinstance(source.get("objects"), dict):
                for name, value in source["objects"].items():
                    base = dict(world["objects"].get(str(name), {}))
                    base.update(dict(value or {}))
                    world["objects"][str(name)] = base
            if source.get("robot_near"):
                world["robot_near"] = str(source["robot_near"])
        for name, obj in world["objects"].items():
            obj.setdefault("label", name)
            obj.setdefault("location", "front_workspace")
            obj.setdefault("visible", True)
            obj.setdefault("graspable", True)
            obj.setdefault("held", False)
            obj.setdefault("color", [210, 80, 70])
        return world

    def _world_snapshot(self) -> dict[str, Any]:
        objects = {
            name: dict(value) for name, value in self.world.get("objects", {}).items()
        }
        visible = [name for name in objects if self._object_visible(name)]
        return {
            "robot_near": self.world.get("robot_near"),
            "held_object": self.object_held,
            "visible_objects": visible,
            "objects": objects,
            "locations": {
                name: dict(value)
                for name, value in self.world.get("locations", {}).items()
            },
            "events": list(self.world_events[-20:]),
            "scan_counts": dict(self.scan_counts),
        }

    def _scene_summary(self) -> dict[str, Any]:
        visible = [
            name for name in self.world.get("objects", {}) if self._object_visible(name)
        ]
        held = self.object_held or "nothing"
        visible_text = ", ".join(visible) or "none"
        return {
            "summary": f"front view near {self.world.get('robot_near')}; visible: {visible_text}; holding: {held}",
            "visible_objects": visible,
            "held_object": self.object_held,
            "robot_near": self.world.get("robot_near"),
            "object_locations": {
                name: self._object_location(name)
                for name in self.world.get("objects", {})
            },
            "last_event": self.world_events[-1] if self.world_events else None,
        }

    def _object(self, name: str) -> dict[str, Any] | None:
        obj = self.world.get("objects", {}).get(name)
        return obj if isinstance(obj, dict) else None

    def _object_location(self, name: str) -> str | None:
        obj = self._object(name)
        return (
            str(obj.get("location"))
            if obj is not None and obj.get("location") is not None
            else None
        )

    def _object_visible(self, name: str) -> bool:
        obj = self._object(name)
        if obj is None:
            return False
        if bool(obj.get("held")):
            return False
        if not bool(obj.get("visible", True)):
            return False
        false_negative_scans = self.settings.get("perception_false_negative_scans", {})
        if isinstance(false_negative_scans, dict):
            required = int(false_negative_scans.get(name, 0) or 0)
            if required > 0 and self.scan_counts.get(name, 0) <= required:
                return False
        return True

    def _nearest_graspable_object(self) -> str | None:
        robot_near = self.world.get("robot_near")
        for name, obj in self.world.get("objects", {}).items():
            if (
                bool(obj.get("graspable", True))
                and self._object_visible(str(name))
                and obj.get("location") == robot_near
            ):
                return str(name)
        for name, obj in self.world.get("objects", {}).items():
            if bool(obj.get("graspable", True)) and self._object_visible(str(name)):
                return str(name)
        return None

    def _release_held_object(self, location: str) -> None:
        if not self.object_held:
            return
        obj = self._object(self.object_held)
        if obj is not None:
            obj["held"] = False
            obj["visible"] = True
            obj["location"] = location
            self._record_world_event(
                "released", object=self.object_held, location=location
            )

    def _reveal_after_scan(self, target: str) -> None:
        obj = self._object(target)
        if obj is None:
            return
        reveal_after = self.settings.get("visible_after_scans", {})
        if not isinstance(reveal_after, dict):
            return
        required = int(reveal_after.get(target, 0) or 0)
        if (
            required > 0
            and self.scan_counts.get(target, 0) >= required
            and not bool(obj.get("visible", True))
        ):
            obj["visible"] = True
            self._record_world_event(
                "revealed", object=target, scans=self.scan_counts.get(target, 0)
            )

    def _record_world_event(self, kind: str, **payload: Any) -> None:
        self.world_events.append(
            {"kind": kind, "frame_id": self.frame_id, "time": time.time(), **payload}
        )

    def _scripted_failure(self, skill: RobotSkillAction) -> RobotSkillResult | None:
        scripts = (
            self.settings.get("scripted_failures")
            or self.settings.get("failure_script")
            or []
        )
        if not isinstance(scripts, list):
            return None
        attempt = self.skill_counts.get(skill.name, 0)
        for item in scripts:
            if not isinstance(item, dict):
                continue
            if str(item.get("skill") or item.get("name") or "") != skill.name:
                continue
            expected_attempt = item.get("attempt")
            if expected_attempt is not None and int(expected_attempt) != attempt:
                continue
            remaining_key = "_remaining"
            if "times" in item:
                item[remaining_key] = int(
                    item.get(remaining_key) or item.get("times") or 0
                )
                if int(item[remaining_key]) <= 0:
                    continue
                item[remaining_key] = int(item[remaining_key]) - 1
            message = str(
                item.get("message") or f"mock scripted failure for {skill.name}"
            )
            failure_mode = str(item.get("failure_mode") or "scripted_failure")
            return RobotSkillResult(
                False,
                message,
                {
                    "skill": skill.to_dict(),
                    "failure_mode": failure_mode,
                    "script": dict(item),
                    "attempt": attempt,
                },
            )
        return None

    def _drain_battery_for_skill(self, skill: RobotSkillAction) -> None:
        if self.settings.get("battery_status") is not None:
            return
        drain = 0.05
        if skill.name.startswith("base_"):
            drain = 0.25
        elif skill.name in {"set_arm_pose", "move_arm_joints", "set_gripper"}:
            drain = 0.18
        self.battery_percentage = self._clamp(
            self.battery_percentage - drain, 0.0, 100.0
        )

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

    def _camera_available_for_observe(self, observe_count: int) -> bool:
        if not bool(self.settings.get("camera_available", True)):
            return False
        warmup_missing = int(self.settings.get("camera_warmup_missing_frames", 0) or 0)
        if observe_count <= warmup_missing:
            return False
        drop_every = int(self.settings.get("camera_drop_every_n_observe", 0) or 0)
        return not (drop_every > 0 and observe_count % drop_every == 0)

    def _camera_drop_reason(self, observe_count: int) -> str | None:
        if not bool(self.settings.get("camera_available", True)):
            return "camera_unavailable"
        warmup_missing = int(self.settings.get("camera_warmup_missing_frames", 0) or 0)
        if observe_count <= warmup_missing:
            return "camera_warmup"
        drop_every = int(self.settings.get("camera_drop_every_n_observe", 0) or 0)
        if drop_every > 0 and observe_count % drop_every == 0:
            return "intermittent_drop"
        return None

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

    def _front_view(self) -> np.ndarray:
        image = np.full((160, 240, 3), 235, dtype=np.uint8)
        image[95:135, 25:215] = np.array([170, 190, 205], dtype=np.uint8)
        slots = {
            "front_workspace": (120, 78),
            "table": (62, 58),
            "bin": (178, 104),
            "shelf": (178, 50),
        }
        for name, obj in self.world.get("objects", {}).items():
            if not self._object_visible(str(name)):
                continue
            x, y = slots.get(str(obj.get("location") or "front_workspace"), (120, 78))
            jitter = (sum(ord(ch) for ch in str(name)) % 13) - 6
            x = int(self._clamp(x + jitter, 16, 224))
            object_color = np.array(obj.get("color") or [210, 80, 70], dtype=np.uint8)
            image[y - 10 : y + 10, x - 10 : x + 10] = object_color
            image[y - 14 : y - 10, x - 8 : x + 8] = np.array(
                [35, 35, 35], dtype=np.uint8
            )
        obstacle_x = int(118 + 30 * math.sin(self.frame_id / 3.0))
        image[86:120, obstacle_x : obstacle_x + 14] = np.array(
            [60, 65, 70], dtype=np.uint8
        )
        arm_x = int(self._clamp(120 + self.arm_joints["shoulder_pan"] / 3.0, 20, 220))
        arm_y = int(self._clamp(82 - self.arm_joints["shoulder_lift"] / 2.0, 20, 130))
        image[arm_y : arm_y + 8, 110:arm_x] = np.array([40, 150, 95], dtype=np.uint8)
        grip = int(self.gripper_opening_pct / 100.0 * 20)
        image[arm_y - 8 : arm_y + 16, arm_x : arm_x + 4] = np.array(
            [35, 95, 80], dtype=np.uint8
        )
        image[arm_y - grip // 2 : arm_y - grip // 2 + 4, arm_x - 8 : arm_x + 10] = (
            np.array([35, 95, 80], dtype=np.uint8)
        )
        image[arm_y + grip // 2 : arm_y + grip // 2 + 4, arm_x - 8 : arm_x + 10] = (
            np.array([35, 95, 80], dtype=np.uint8)
        )
        bar = self.frame_id % image.shape[1]
        image[:4, :bar] = np.array([30, 130, 220], dtype=np.uint8)
        battery = self._battery_status()["status"]
        battery_color = {
            "normal": [45, 170, 80],
            "low": [230, 165, 35],
            "critical": [220, 55, 55],
        }.get(str(battery), [120, 120, 120])
        image[8:18, 204:232] = np.array(battery_color, dtype=np.uint8)
        return image

    def _left_wrist_view(self) -> np.ndarray:
        image = np.full((160, 240, 3), 225, dtype=np.uint8)
        image[28:132, 42:198] = np.array([235, 239, 242], dtype=np.uint8)
        image[54:118, 64:176] = np.array([190, 214, 228], dtype=np.uint8)
        image[70:108, 88:152] = np.array([40, 150, 95], dtype=np.uint8)
        image[78:100, 150:192] = np.array([35, 95, 80], dtype=np.uint8)
        return image

    def _right_wrist_view(self) -> np.ndarray:
        image = np.full((160, 240, 3), 228, dtype=np.uint8)
        image[24:128, 34:190] = np.array([238, 240, 236], dtype=np.uint8)
        image[58:122, 58:170] = np.array([210, 196, 182], dtype=np.uint8)
        image[72:108, 84:146] = np.array([35, 95, 80], dtype=np.uint8)
        image[68:112, 144:188] = np.array([230, 175, 45], dtype=np.uint8)
        return image

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
            state=self.state,
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

    @staticmethod
    def _wrap_yaw(value: float) -> float:
        return ((value + 180.0) % 360.0) - 180.0

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _default_camera(self) -> str:
        if self.context.embodiment and self.context.embodiment.default_camera:
            return self.context.embodiment.default_camera
        return "front"

    def _build_camera_status_map(
        self,
        *,
        frame_id: int | None,
        camera_ok: bool,
        image_shape: list[int] | None,
        drop_reason: str | None,
    ) -> dict[str, Any]:
        return {
            name: {
                "ok": camera_ok,
                "frame_available": camera_ok,
                "frame_id": frame_id,
                "image_shape": image_shape,
                "owner": "mock",
                "drop_reason": drop_reason,
            }
            for name in self._camera_names()
        }

    def _camera_names(self) -> tuple[str, ...]:
        if self.context.embodiment is not None:
            raw = self.context.embodiment.camera_layout.get("cameras")
            if isinstance(raw, (list, tuple)):
                names = tuple(str(item) for item in raw if str(item).strip())
                if names:
                    return names
        return self._DEFAULT_CAMERA_NAMES

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


_TASK_OBJECT_ALIASES: dict[str, tuple[str, ...]] = {
    "marker": ("marker", "pen", "马克笔", "记号笔"),
    "cup": ("cup", "mug", "杯", "杯子", "水杯"),
    "block": ("block", "cube", "积木", "方块"),
    "apple": ("apple", "苹果"),
    "bottle": ("bottle", "瓶子", "水瓶"),
    "mock_object": ("mock_object", "object", "物体", "目标物"),
}

_PLACE_ACTION_MARKERS: tuple[str, ...] = (
    "place",
    "put",
    "drop",
    "throw",
    "放",
    "放到",
    "放进",
    "放入",
)

_TASK_LOCATION_ALIASES: dict[str, tuple[str, ...]] = {
    "bin": ("bin", "trash", "垃圾桶", "回收箱", "收纳箱"),
    "table": ("table", "desk", "桌", "桌子"),
    "shelf": ("shelf", "架子", "货架"),
    "front_workspace": ("front_workspace", "front workspace", "工作区", "前方工作区"),
}


def _target_from_task(task: str) -> str | None:
    text = str(task).lower()
    for candidate, aliases in _TASK_OBJECT_ALIASES.items():
        if any(alias in text for alias in aliases):
            return candidate
    return None


def _looks_like_place_task(task: str) -> bool:
    text = str(task).lower()
    return any(marker in text for marker in _PLACE_ACTION_MARKERS)


def _place_location_from_task(task: str) -> str | None:
    text = str(task).lower()
    for location, aliases in _TASK_LOCATION_ALIASES.items():
        if any(alias in text for alias in aliases):
            return location
    return None
