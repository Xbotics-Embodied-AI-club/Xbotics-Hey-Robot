from __future__ import annotations

import asyncio
import time
from typing import Any

from hey_robot.perception import DriverObservation, ObservationAsset
from hey_robot.protocol import Envelope, RobotAction, RobotStatus
from hey_robot.robots.base import RobotCapabilities, RobotDriverContext, RobotHealth
from hey_robot.robots.so101.client import SO101Client
from hey_robot.robots.so101.config import hardware_config_from_settings
from hey_robot.skills import RobotSkillAction, RobotSkillResult


class SO101Driver:
    """Standalone SO101 arm driver for Hey Robot."""

    def __init__(self, context: RobotDriverContext) -> None:
        self.context = context
        self.robot_id = context.robot_id
        self.hardware_config = hardware_config_from_settings(context.spec.settings)
        self.client = SO101Client(self.hardware_config)
        self.state = "created"
        self.frame_id = 0
        self.last_error: str | None = None
        self.last_skill_result: RobotSkillResult | None = None
        self.last_arm_status: dict[str, Any] = {}
        self.last_battery: dict[str, Any] = {
            "status": "unknown",
            "voltage": None,
            "percentage": None,
        }
        self.startup_diagnostics: dict[str, Any] = {}
        self._default_camera = (
            context.embodiment.default_camera
            if context.embodiment and context.embodiment.default_camera
            else "front"
        )

    async def start(self) -> None:
        await asyncio.to_thread(self.client.connect)
        self.startup_diagnostics = await asyncio.to_thread(
            self.client.diagnose,
            video_timeout_ms=int(
                self.context.spec.settings.get("startup_video_timeout_ms", 1000)
            ),
        )
        self.state = (
            "idle"
            if bool((self.startup_diagnostics.get("arm") or {}).get("ok"))
            else "degraded"
        )
        self.last_error = (
            None
            if self.state == "idle"
            else _failure_summary(self.startup_diagnostics, ("arm",))
        )

    async def capabilities(self) -> RobotCapabilities:
        return RobotCapabilities(
            robot_id=self.robot_id,
            driver_type="so101",
            cameras=[self._default_camera]
            if self.hardware_config.camera.enabled
            else [],
            observation_modalities=["image", "arm_state", "status"],
            supports_reset=True,
            supports_interrupt=True,
            metadata={
                "body": "so101",
                "robot_family": self.context.spec.robot_family,
                "environment": self.context.spec.robot_environment,
                "driver_kind": self.context.spec.driver_kind,
                "embodiment_profile": (
                    self.context.embodiment.name if self.context.embodiment else None
                ),
                "control": "skill_action",
                "runtime": "native_hardware",
            },
        )

    async def health(self) -> RobotHealth:
        return RobotHealth(
            robot_id=self.robot_id,
            online=self.state != "closed",
            state=self.state,
            frame_id=self.frame_id,
            error=self.last_error,
            metrics={
                "startup_diagnostics": self.startup_diagnostics,
                "battery": self.last_battery,
                "readiness": self.readiness(),
            },
        )

    async def observe(self) -> DriverObservation:
        frame_id, image = await asyncio.to_thread(
            self.client.capture_frame,
            timeout_ms=int(self.context.spec.settings.get("video_timeout_ms", 500)),
        )
        self.frame_id = int(frame_id) if frame_id is not None else self.frame_id + 1
        self.last_arm_status = await asyncio.to_thread(self.client.arm_status)
        self.last_battery = await asyncio.to_thread(self.client.battery_status)
        assets = []
        if image is not None:
            assets.append(
                ObservationAsset(
                    kind="image",
                    role="camera",
                    name=self._default_camera,
                    data=image,
                    metadata={"driver": "so101"},
                )
            )
        return DriverObservation(
            envelope=self._envelope(),
            frame_id=self.frame_id,
            assets=assets,
            metadata={
                "driver": "so101",
                "body": "so101",
                "robot_family": self.context.spec.robot_family,
                "environment": self.context.spec.robot_environment,
                "embodiment_profile": (
                    self.context.embodiment.name if self.context.embodiment else None
                ),
                "state": self.state,
                "arm_status": self.last_arm_status,
                "battery": self.last_battery,
                "startup_diagnostics": self.startup_diagnostics,
            },
        )

    async def status(self) -> RobotStatus:
        return RobotStatus(
            envelope=self._envelope(),
            frame_id=self.frame_id,
            state=self.state,
            success=None,
            error=self.last_error,
            metrics={
                "driver": "so101",
                "startup_diagnostics": self.startup_diagnostics,
                "arm_status": self.last_arm_status,
                "battery": self.last_battery,
                "last_skill_result": self.last_skill_result.to_dict()
                if self.last_skill_result
                else None,
                "readiness": self.readiness(),
            },
        )

    async def apply_action(self, action: RobotAction) -> RobotStatus:
        skill = RobotSkillAction.from_robot_action(action)
        result = await asyncio.to_thread(self._execute_skill, skill)
        self.last_skill_result = result
        self.state = "skill_completed" if result.success else "failed"
        self.last_error = None if result.success else result.message
        status = await self.status()
        return RobotStatus(
            envelope=status.envelope,
            frame_id=status.frame_id,
            state=status.state,
            task=status.task,
            skill_id=action.skill_id,
            success=result.success,
            error=None if result.success else self.last_error,
            metrics=status.metrics,
        )

    async def reset(self) -> RobotStatus:
        self.last_skill_result = await asyncio.to_thread(
            self._execute_skill, RobotSkillAction("base_stop")
        )
        self.state = "idle" if self.last_skill_result.success else "failed"
        return await self.status()

    async def close(self) -> None:
        await asyncio.to_thread(self.client.close)
        self.state = "closed"

    def readiness(self) -> dict[str, Any]:
        diagnostics = self.startup_diagnostics or {}
        diagnostic_arm = (
            diagnostics.get("arm") if isinstance(diagnostics.get("arm"), dict) else {}
        )
        diagnostic_camera = (
            diagnostics.get("camera")
            if isinstance(diagnostics.get("camera"), dict)
            else {}
        )
        arm = self.last_arm_status or diagnostic_arm
        readiness: dict[str, Any] = {
            "robot": self.state != "closed",
            "battery": self.last_battery or {"status": "unknown"},
            "emergency_stop": False,
        }
        for resource in self._readiness_resources():
            if resource == "arm":
                readiness["arm"] = {"ok": _arm_ready(arm or {})}
            elif resource == "gripper":
                readiness["gripper"] = {"ok": _arm_ready(arm or {})}
            elif resource == "camera":
                readiness["camera"] = {
                    "ok": _camera_ready(
                        diagnostic_camera or {}, self.hardware_config.camera.enabled
                    )
                }
        return readiness

    def _readiness_resources(self) -> tuple[str, ...]:
        if self.context.embodiment and self.context.embodiment.readiness_resources:
            return self.context.embodiment.readiness_resources
        return ("arm", "gripper", "camera")

    def _execute_skill(self, skill: RobotSkillAction) -> RobotSkillResult:
        try:
            if skill.name == "base_stop" and bool(
                skill.arguments.get("emergency", False)
            ):
                response = self.client.stop(emergency=True)
            elif skill.name == "base_stop":
                response = self.client.stop()
            elif skill.name == "arm_home":
                response = self.client.arm_home()
            elif skill.name == "arm_set_pose":
                response = self.client.move_named_pose(
                    str(skill.arguments["pose_name"])
                )
            elif skill.name == "arm_set_joints":
                joints = skill.arguments.get("joints")
                if not isinstance(joints, dict):
                    raise ValueError("arm_set_joints requires arguments.joints")
                values = {str(key): float(value) for key, value in joints.items()}
                response = (
                    self.client.set_joints_delta(values)
                    if str(skill.arguments.get("mode", "absolute")).lower() == "delta"
                    else self.client.set_joints(values)
                )
            elif skill.name == "gripper_set":
                action = str(skill.arguments.get("action", "")).lower()
                if action == "open":
                    response = self.client.open_gripper()
                elif action == "close":
                    response = self.client.close_gripper()
                else:
                    response = self.client.set_gripper_opening_pct(
                        float(skill.arguments["opening_pct"])
                    )
            elif skill.name == "arm_stop":
                response = self.client.stop(emergency=True)
            elif skill.name in {
                "camera_capture",
                "camera_inspect",
                "camera_look_around",
            }:
                response = {
                    "success": False,
                    "message": f"{skill.name} must be handled by RobotRuntime PerceptionService",
                }
            else:
                response = {
                    "success": False,
                    "message": f"SO101 does not support skill: {skill.name}",
                }
        except Exception as exc:
            response = {"success": False, "message": f"{type(exc).__name__}: {exc}"}
        return RobotSkillResult.from_response(response)

    def _envelope(self) -> Envelope:
        return Envelope(
            robot_id=self.robot_id,
            deployment_id=self.context.deployment_id,
            trace_id=f"so101_{self.robot_id}_{int(time.time() * 1000)}",
        )


def _failure_summary(diagnostics: dict[str, Any], services: tuple[str, ...]) -> str:
    failed = []
    for service in services:
        item = diagnostics.get(service) or {}
        if not bool(item.get("ok")):
            failed.append(f"{service}: {item.get('issue') or 'not ready'}")
    return "; ".join(failed) if failed else "unknown hardware diagnostic failure"


def _arm_ready(status: dict[str, Any]) -> bool:
    if "success" in status:
        return bool(status.get("success"))
    return bool(status.get("ok", False))


def _camera_ready(status: dict[str, Any], enabled: bool) -> bool:
    if not enabled:
        return False
    if "frame_available" in status:
        return bool(status.get("frame_available"))
    return bool(status.get("ok", enabled))
