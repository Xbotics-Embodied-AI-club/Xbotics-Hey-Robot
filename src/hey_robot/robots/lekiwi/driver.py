from __future__ import annotations

import asyncio
import time
from typing import Any

from hey_robot.perception import DriverObservation, ObservationAsset
from hey_robot.protocol import Envelope, RobotAction, RobotStatus
from hey_robot.robots.base import RobotCapabilities, RobotDriverContext, RobotHealth
from hey_robot.robots.lekiwi.client import LeKiwiClient
from hey_robot.robots.lekiwi.config import hardware_config_from_settings
from hey_robot.robots.so101.driver import _camera_ready, _failure_summary
from hey_robot.skills import RobotSkillAction, RobotSkillResult


class LeKiwiDriver:
    """Standalone LeKiwi mobile-base driver for Hey Robot."""

    def __init__(self, context: RobotDriverContext) -> None:
        self.context = context
        self.robot_id = context.robot_id
        settings = context.spec.settings
        self.hardware_config = hardware_config_from_settings(settings)
        self.client = LeKiwiClient(
            self.hardware_config,
            default_linear_speed=float(settings.get("default_linear_speed", 0.2)),
            default_angular_speed=float(settings.get("default_angular_speed", 0.45)),
            motion_time_scale=float(settings.get("motion_time_scale", 2.0)),
        )
        self.state = "created"
        self.frame_id = 0
        self.last_error: str | None = None
        self.last_skill_result: RobotSkillResult | None = None
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
            if bool((self.startup_diagnostics.get("base") or {}).get("ok"))
            else "degraded"
        )
        self.last_error = (
            None
            if self.state == "idle"
            else _failure_summary(self.startup_diagnostics, ("base",))
        )

    async def capabilities(self) -> RobotCapabilities:
        return RobotCapabilities(
            robot_id=self.robot_id,
            driver_type="lekiwi",
            cameras=[self._default_camera]
            if self.hardware_config.camera.enabled
            else [],
            observation_modalities=["image", "base_state", "status"],
            supports_reset=True,
            supports_interrupt=True,
            metadata={
                "body": "lekiwi",
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
        self.last_battery = await asyncio.to_thread(self.client.battery_status)
        assets = []
        if image is not None:
            assets.append(
                ObservationAsset(
                    kind="image",
                    role="camera",
                    name=self._default_camera,
                    data=image,
                    metadata={"driver": "lekiwi"},
                )
            )
        return DriverObservation(
            envelope=self._envelope(),
            frame_id=self.frame_id,
            assets=assets,
            metadata={
                "driver": "lekiwi",
                "body": "lekiwi",
                "robot_family": self.context.spec.robot_family,
                "environment": self.context.spec.robot_environment,
                "embodiment_profile": (
                    self.context.embodiment.name if self.context.embodiment else None
                ),
                "state": self.state,
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
                "driver": "lekiwi",
                "startup_diagnostics": self.startup_diagnostics,
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
        diagnostic_base = (
            diagnostics.get("base") if isinstance(diagnostics.get("base"), dict) else {}
        )
        diagnostic_camera = (
            diagnostics.get("camera")
            if isinstance(diagnostics.get("camera"), dict)
            else {}
        )
        readiness: dict[str, Any] = {
            "robot": self.state != "closed",
            "battery": self.last_battery or {"status": "unknown"},
            "emergency_stop": False,
        }
        for resource in self._readiness_resources():
            if resource == "base":
                readiness["base"] = {
                    "ok": bool(
                        (diagnostic_base or {}).get("ok", self.state != "closed")
                    )
                }
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
        return ("base", "camera")

    def _execute_skill(self, skill: RobotSkillAction) -> RobotSkillResult:
        try:
            if skill.name == "base_stop" and bool(
                skill.arguments.get("emergency", False)
            ):
                response = self.client.stop(emergency=True)
            elif skill.name == "base_stop":
                response = self.client.stop()
            elif skill.name == "base_move":
                distance = float(
                    skill.arguments.get("distance_cm", skill.arguments.get("cm", 0.0))
                )
                if (
                    str(skill.arguments.get("direction", "forward")).lower()
                    == "backward"
                ):
                    distance = -abs(distance)
                response = self.client.move_forward_cm(distance)
            elif skill.name == "base_strafe":
                distance = float(
                    skill.arguments.get("distance_cm", skill.arguments.get("cm", 0.0))
                )
                if str(skill.arguments.get("direction", "left")).lower() == "right":
                    distance = -abs(distance)
                response = self.client.strafe_left_cm(distance)
            elif skill.name == "base_turn":
                angle = float(
                    skill.arguments.get("angle_deg", skill.arguments.get("deg", 0.0))
                )
                if str(skill.arguments.get("direction", "left")).lower() == "left":
                    angle = -abs(angle)
                response = self.client.turn_deg(angle)
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
                    "message": f"LeKiwi does not support skill: {skill.name}",
                }
        except Exception as exc:
            response = {"success": False, "message": f"{type(exc).__name__}: {exc}"}
        return RobotSkillResult.from_response(response)

    def _envelope(self) -> Envelope:
        return Envelope(
            robot_id=self.robot_id,
            deployment_id=self.context.deployment_id,
            trace_id=f"lekiwi_{self.robot_id}_{int(time.time() * 1000)}",
        )
