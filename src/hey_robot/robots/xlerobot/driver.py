from __future__ import annotations

import asyncio
import time
from typing import Any, Protocol

from hey_robot.logging import HeyRobotLogger
from hey_robot.perception import DriverObservation, ObservationAsset
from hey_robot.protocol import Envelope, RobotAction, RobotStatus
from hey_robot.robots.base import RobotCapabilities, RobotDriverContext, RobotHealth
from hey_robot.robots.xlerobot.client import XLeRobotClient
from hey_robot.robots.xlerobot.executor import XLeRobotSkillExecutor
from hey_robot.robots.xlerobot.hardware.config import hardware_config_from_settings
from hey_robot.skills import RobotSkillAction, RobotSkillResult
from hey_robot.skills.contracts import SkillContractRuntime

logger = HeyRobotLogger(name="xlerobot")


class XLeRobotClientProtocol(Protocol):
    def connect(self) -> None: ...
    def diagnose(self, *, video_timeout_ms: int) -> dict[str, Any]: ...
    def base_control_diagnostics(self) -> dict[str, Any]: ...
    def capture_frame(self, *, timeout_ms: int = 2000) -> Any: ...
    def capture_frames(
        self, *, timeout_ms: int = 2000
    ) -> dict[str, dict[str, Any]]: ...
    def arm_status(self) -> dict[str, Any]: ...
    def camera_status(self) -> dict[str, Any]: ...
    def battery_status(self) -> dict[str, Any]: ...
    def close(self) -> None: ...


class XLeRobotDriver:
    """XLeRobot embodiment driver for the hey-robot runtime."""

    def __init__(self, context: RobotDriverContext) -> None:
        self.context = context
        self.robot_id = context.robot_id
        settings = context.spec.settings
        self.hardware_config = hardware_config_from_settings(settings)
        self.client: XLeRobotClientProtocol = XLeRobotClient(
            self.hardware_config,
            default_linear_speed=float(settings.get("default_linear_speed", 0.2)),
            default_angular_speed=float(settings.get("default_angular_speed", 0.45)),
            motion_time_scale=float(settings.get("motion_time_scale", 2.0)),
        )
        self.executor = XLeRobotSkillExecutor(self.client)
        self.contracts = SkillContractRuntime(context.skill_catalog)
        self.state = "created"
        self.frame_id = 0
        self.last_error: str | None = None
        self.last_skill_result: RobotSkillResult | None = None
        self.last_arm_status: dict[str, Any] = {}
        self.last_arms_status: dict[str, Any] = {}
        self.last_camera: dict[str, Any] = {
            "frame_available": False,
            "frame_id": None,
            "image_shape": None,
        }
        self.last_cameras_status: dict[str, Any] = {}
        self.last_battery: dict[str, Any] = {
            "status": "unknown",
            "voltage": None,
            "percentage": None,
        }
        self.startup_diagnostics: dict[str, Any] = {}
        self._hardware_summary = {
            "serial_port": self.hardware_config.serial_bus.port,
            "baudrate": self.hardware_config.serial_bus.baudrate,
            "default_arm": self.hardware_config.default_arm,
            "default_camera": self.hardware_config.default_camera,
            "camera_device_id": self.hardware_config.camera.device_id,
            "camera_device_ids": {
                name: camera.device_id
                for name, camera in self.hardware_config.cameras.items()
            },
            "video_timeout_ms": int(settings.get("video_timeout_ms", 500)),
            "base_type": self.hardware_config.base.type,
            "base_wheel_ids": [
                self.hardware_config.base.left_front_id,
                self.hardware_config.base.right_front_id,
                self.hardware_config.base.rear_id,
            ],
            "arm_type": self.hardware_config.arm.type,
            "arm_joint_ids": dict(self.hardware_config.arm.joint_ids),
            "arm_joint_ids_by_name": {
                name: dict(arm.joint_ids)
                for name, arm in self.hardware_config.arms.items()
            },
            "battery_servo_ids": list(self.hardware_config.battery.servo_ids),
        }
        self._last_camera_log_at = 0.0
        self._last_camera_available: bool | None = None
        self._last_arm_log_at = 0.0
        self._last_arm_ok: bool | None = None
        self._last_battery_log_at = 0.0
        self._last_battery_status: str | None = None
        self._telemetry_interval_sec = 1.0 / max(
            float(settings.get("telemetry_hz", 2.0)), 0.1
        )
        self._last_telemetry_at = 0.0
        self._default_camera = (
            context.embodiment.default_camera
            if context.embodiment and context.embodiment.default_camera
            else "front"
        )

    async def start(self) -> None:
        logger.info(
            f"{self.robot_id} 正在打开原生硬件 serial={self._hardware_summary['serial_port']} "
            f"baudrate={self._hardware_summary['baudrate']} camera={self._hardware_summary['camera_device_id']}"
        )
        await asyncio.to_thread(self.client.connect)
        self.startup_diagnostics = await asyncio.to_thread(
            self.client.diagnose,
            video_timeout_ms=int(
                self.context.spec.settings.get("startup_video_timeout_ms", 3000)
            ),
        )
        self._seed_runtime_state_from_diagnostics(self.startup_diagnostics)
        self._log_startup_diagnostics(self.startup_diagnostics)
        if self._diagnostics_ready(self.startup_diagnostics):
            self.state = "idle"
            self.last_error = None
        else:
            self.state = "degraded"
            self.last_error = self._diagnostic_failure_summary(self.startup_diagnostics)
            logger.warning(
                f"{self.robot_id} 硬件异常: {self.last_error}; 执行真实 action 前请检查串口、舵机电源、相机设备"
            )

    async def capabilities(self) -> RobotCapabilities:
        return RobotCapabilities(
            robot_id=self.robot_id,
            driver_type="xlerobot",
            action_dimensions=None,
            control_hz=float(self.context.spec.settings.get("control_hz", 2.0)),
            cameras=list(self.hardware_config.cameras.keys()),
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
                "runtime": "native_hardware",
                "supported_skills": list(self.executor.supported_skills),
                "default_arm": self.hardware_config.default_arm,
                "default_camera": self.hardware_config.default_camera,
                "arms": list(self.hardware_config.arms.keys()),
                "cameras": list(self.hardware_config.cameras.keys()),
                "safety": dict(self.context.spec.settings.get("safety", {}) or {}),
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
                "driver": "xlerobot",
                "hardware": self._hardware_summary,
                "startup_diagnostics": self.startup_diagnostics,
                "last_skill_result": self.last_skill_result.to_dict()
                if self.last_skill_result
                else None,
                "base_control": self.client.base_control_diagnostics(),
                "battery": self.last_battery,
                "readiness": self.readiness(),
                **dict(self.context.spec.settings.get("safety", {}) or {}),
            },
        )

    async def observe(self) -> DriverObservation:
        timeout_ms = int(self.context.spec.settings.get("video_timeout_ms", 500))
        frames = await asyncio.to_thread(
            self.client.capture_frames, timeout_ms=timeout_ms
        )
        default_frame = dict(frames.get(self.hardware_config.default_camera, {}) or {})
        frame_id = default_frame.get("frame_id")
        image = default_frame.get("image")
        self.frame_id = int(frame_id) if frame_id is not None else self.frame_id + 1
        camera_status = await asyncio.to_thread(self.client.camera_status)
        now = time.monotonic()
        if now - self._last_telemetry_at >= self._telemetry_interval_sec:
            arm_status, battery = await asyncio.to_thread(self._read_telemetry)
            self.last_arm_status = arm_status
            self.last_arms_status = dict(arm_status.get("arms", {}) or {})
            self.last_battery = battery
            self._last_telemetry_at = now
        arm_status = self.last_arm_status
        battery = self.last_battery
        camera_diagnostics = dict(camera_status.get("cameras", {}) or {})
        camera_frames = {
            name: {
                "frame_available": item.get("image") is not None,
                "frame_id": item.get("frame_id"),
                "image_shape": (
                    list(item["image"].shape) if item.get("image") is not None else None
                ),
            }
            for name, item in frames.items()
        }
        camera_names = sorted({*camera_diagnostics.keys(), *camera_frames.keys()})
        self.last_cameras_status = {
            name: {
                **dict(camera_diagnostics.get(name, {}) or {}),
                **dict(camera_frames.get(name, {}) or {}),
            }
            for name in camera_names
        }
        default_camera_status = dict(
            self.last_cameras_status.get(self.hardware_config.default_camera, {})
        )
        self.last_camera = {
            **default_camera_status,
            "frame_available": image is not None,
            "frame_id": self.frame_id,
            "image_shape": list(image.shape) if image is not None else None,
            "default_camera": self.hardware_config.default_camera,
            "cameras": self.last_cameras_status,
        }
        self._log_runtime_diagnostics()
        assets: list[ObservationAsset] = []
        for name, item in frames.items():
            if item.get("image") is not None:
                assets.append(
                    ObservationAsset(
                        kind="image",
                        role="camera",
                        name=name,
                        data=item["image"],
                        metadata={"driver": "xlerobot", "camera_role": name},
                    )
                )
        return DriverObservation(
            envelope=self._envelope(),
            frame_id=self.frame_id,
            assets=assets,
            metadata={
                "driver": "xlerobot",
                "body": "xlerobot",
                "robot_family": self.context.spec.robot_family,
                "environment": self.context.spec.robot_environment,
                "embodiment_profile": (
                    self.context.embodiment.name if self.context.embodiment else None
                ),
                "state": self.state,
                "camera": self.last_camera,
                "cameras": self.last_cameras_status,
                "arm_status": arm_status,
                "arms": self.last_arms_status,
                "battery": battery,
                "startup_diagnostics": self.startup_diagnostics,
                "last_skill_result": self.last_skill_result.to_dict()
                if self.last_skill_result
                else None,
                "readiness": self.readiness(),
            },
        )

    def _read_telemetry(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.client.arm_status(), self.client.battery_status()

    async def status(self) -> RobotStatus:
        return RobotStatus(
            envelope=self._envelope(),
            frame_id=self.frame_id,
            state=self.state,
            success=None,
            error=self.last_error,
            metrics={
                "driver": "xlerobot",
                "hardware": self._hardware_summary,
                "startup_diagnostics": self.startup_diagnostics,
                "camera": self.last_camera,
                "cameras": self.last_cameras_status,
                "arm_status": self.last_arm_status,
                "arms": self.last_arms_status,
                "battery": self.last_battery,
                "last_skill_result": self.last_skill_result.to_dict()
                if self.last_skill_result
                else None,
                "base_control": self.client.base_control_diagnostics(),
                "readiness": self.readiness(),
            },
        )

    async def apply_action(self, action: RobotAction) -> RobotStatus:
        skill = RobotSkillAction.from_robot_action(action)
        _, decision = self.contracts.validate_action(
            skill,
            robot_type="xlerobot",
            status=await self.status(),
            readiness=self.readiness(),
        )
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
        else:
            try:
                result = await asyncio.to_thread(self.executor.execute, skill)
            except Exception as exc:
                result = RobotSkillResult(
                    False, f"{type(exc).__name__}: {exc}", {"skill": skill.to_dict()}
                )
        self.last_skill_result = result
        if result.success:
            self.state = "skill_completed"
            self.last_error = None
        else:
            self.state = "failed"
            self.last_error = result.message or "skill failed"
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

    def readiness(self) -> dict[str, Any]:
        diagnostics = self.startup_diagnostics or {}
        diagnostic_base = (
            diagnostics.get("base") if isinstance(diagnostics.get("base"), dict) else {}
        )
        diagnostic_arm = (
            diagnostics.get("arm") if isinstance(diagnostics.get("arm"), dict) else {}
        )
        diagnostic_camera = (
            diagnostics.get("camera")
            if isinstance(diagnostics.get("camera"), dict)
            else {}
        )
        diagnostic_battery = (
            diagnostics.get("battery")
            if isinstance(diagnostics.get("battery"), dict)
            else {}
        )
        arm = self.last_arm_status or diagnostic_arm
        camera = self.last_camera or diagnostic_camera
        battery = self.last_battery or diagnostic_battery
        robot_online = self.state != "closed"
        readiness: dict[str, Any] = {
            "robot": robot_online,
            "battery": battery or {"status": "unknown"},
            "emergency_stop": bool(
                (diagnostics.get("safety") or {}).get("emergency_stop", False)
            )
            if isinstance(diagnostics.get("safety"), dict)
            else False,
        }
        for resource in self._readiness_resources():
            if resource == "base":
                readiness["base"] = {
                    "ok": bool((diagnostic_base or {}).get("ok", robot_online))
                }
            elif resource == "arm":
                readiness["arm"] = {"ok": self._arm_ready(arm or {})}
            elif resource == "gripper":
                readiness["gripper"] = {"ok": self._arm_ready(arm or {})}
            elif resource == "camera":
                readiness["camera"] = {
                    "ok": self._camera_ready(camera or {}),
                    "owner": self.hardware_config.camera_owner,
                }
        for arm_name, status in self.last_arms_status.items():
            readiness.setdefault(f"{arm_name}_arm", {"ok": self._arm_ready(status)})
            readiness.setdefault(f"{arm_name}_gripper", {"ok": self._arm_ready(status)})
        for camera_name, status in self.last_cameras_status.items():
            readiness.setdefault(
                f"{camera_name}_camera",
                {
                    "ok": self._camera_ready(status),
                    "owner": self.hardware_config.camera_owners.get(
                        camera_name, "robot_driver"
                    ),
                },
            )
        return readiness

    def _readiness_resources(self) -> tuple[str, ...]:
        if self.context.embodiment and self.context.embodiment.readiness_resources:
            return self.context.embodiment.readiness_resources
        return ("base", "arm", "gripper", "camera")

    def _seed_runtime_state_from_diagnostics(self, diagnostics: dict[str, Any]) -> None:
        arm = diagnostics.get("arm")
        if isinstance(arm, dict) and arm:
            self.last_arm_status = dict(
                arm.get("status_response") or arm.get("response") or arm
            )
            self.last_arms_status = dict(arm.get("arms", {}) or {})
        camera = diagnostics.get("camera")
        if isinstance(camera, dict) and camera:
            self.last_camera = dict(camera)
            self.last_cameras_status = dict(camera.get("cameras", {}) or {})
        battery = diagnostics.get("battery")
        if isinstance(battery, dict) and battery:
            self.last_battery = dict(battery)

    @staticmethod
    def _arm_ready(status: dict[str, Any]) -> bool:
        if "success" in status:
            return bool(status.get("success"))
        return bool(status.get("ok", False))

    @staticmethod
    def _camera_ready(status: dict[str, Any]) -> bool:
        if status.get("owner") == "camera_service":
            return bool(status.get("ok", False))
        if "frame_available" in status:
            return bool(status.get("frame_available"))
        return bool(status.get("ok", False))

    async def reset(self) -> RobotStatus:
        result = await asyncio.to_thread(
            self.executor.execute, RobotSkillAction("stop_motion")
        )
        self.last_skill_result = result
        self.last_error = None if result.success else result.message
        self.state = "idle" if result.success else "failed"
        return await self.status()

    async def close(self) -> None:
        self.executor.close()
        await asyncio.to_thread(self.client.close)
        self.state = "closed"

    async def stream_camera_frames(
        self, *, timeout_ms: int = 100
    ) -> dict[str, dict[str, Any]]:
        return await asyncio.to_thread(
            self.client.capture_frames, timeout_ms=timeout_ms
        )

    async def apply_stream_velocity(
        self, *, vx: float, vy: float, wz: float, watchdog_ms: int
    ) -> RobotSkillResult:
        return await asyncio.to_thread(
            self.executor.execute,
            RobotSkillAction(
                "base_velocity_step",
                {
                    "vx": vx,
                    "vy": vy,
                    "wz": wz,
                    "duration_ms": watchdog_ms,
                },
            ),
        )

    async def stop_base_stream(self) -> dict[str, Any]:
        return await asyncio.to_thread(self.executor.stop_base_only)

    def _envelope(self) -> Envelope:
        return Envelope(
            robot_id=self.robot_id,
            deployment_id=self.context.deployment_id,
            trace_id=f"xlerobot_{self.robot_id}_{int(time.time() * 1000)}",
        )

    def _log_startup_diagnostics(self, diagnostics: dict[str, Any]) -> None:
        bus = diagnostics.get("bus", {})
        servo_bus = diagnostics.get("servo_bus", {})
        logger.info(
            f"{self.robot_id} 舵机总线诊断: connected={bool(bus.get('ok'))} "
            f"port={bus.get('port')} baudrate={bus.get('baudrate')} message={bus.get('message')!r}"
        )
        if servo_bus:
            missing = servo_bus.get("missing_or_unresponsive_ids") or []
            voltage_missing = servo_bus.get("voltage_unavailable_ids") or []
            level = logger.warning if missing else logger.info
            level(
                f"{self.robot_id} 舵机ID扫描: configured={servo_bus.get('configured_ids')} "
                f"missing_or_unresponsive={missing} voltage_unavailable={voltage_missing} "
                f"issue={servo_bus.get('issue')!r}"
            )
            for item in servo_bus.get("servos") or []:
                level = logger.warning if not item.get("ping") else logger.info
                level(
                    f"{self.robot_id} 舵机 {item.get('servo_id')} "
                    f"roles={item.get('roles')} ping={item.get('ping')} "
                    f"position={item.get('position')} voltage={item.get('voltage')} "
                    f"temperature={item.get('temperature')}"
                )

        base = diagnostics.get("base", {})
        base_response = base.get("startup_response") or base.get("response") or {}
        base_status = base.get("status_response") or {}
        self._log_probe(
            "base",
            bool(base.get("ok")),
            f"{self.robot_id} 底盘诊断 ok={bool(base.get('ok'))} "
            f"type={self._hardware_summary['base_type']} wheels={self._hardware_summary['base_wheel_ids']} "
            f"owner={base_response.get('current_owner')} "
            f"priority={base_response.get('current_priority')} "
            f"startup_message={base_response.get('message')!r} "
            f"status_initialized={base_status.get('initialized')} "
            f"status_message={base_status.get('message')!r} issue={base.get('issue')!r}",
        )

        arm = diagnostics.get("arm", {})
        arm_response = arm.get("startup_response") or arm.get("response") or {}
        arm_status = arm.get("status_response") or {}
        self._log_probe(
            "arm",
            bool(arm.get("ok")),
            f"{self.robot_id} 机械臂诊断 ok={bool(arm.get('ok'))} "
            f"joint_ids={self._hardware_summary['arm_joint_ids']} "
            f"joint_count={arm.get('joint_count', 0)} lift_height={arm.get('lift_height')} "
            f"owner={arm_response.get('current_owner')} "
            f"priority={arm_response.get('current_priority')} "
            f"startup_message={arm_response.get('message')!r} "
            f"status_initialized={arm_status.get('initialized')} "
            f"status_message={arm_status.get('message')!r} issue={arm.get('issue')!r}",
        )
        for arm_name, arm_item in dict(arm.get("arms", {}) or {}).items():
            self._log_probe(
                f"arm:{arm_name}",
                bool(arm_item.get("ok")),
                f"{self.robot_id} 机械臂诊断 arm={arm_name} ok={bool(arm_item.get('ok'))} "
                f"joint_count={arm_item.get('joint_count', 0)} issue={arm_item.get('issue')!r}",
            )

        camera = diagnostics.get("camera", {})
        self._log_probe(
            "camera",
            bool(camera.get("ok")),
            f"{self.robot_id} 相机诊断 ok={bool(camera.get('ok'))} "
            f"device={self._hardware_summary['camera_device_id']} "
            f"owner={camera.get('owner')} "
            f"frame={camera.get('frame_id')} jpeg_bytes={camera.get('jpeg_bytes', 0)} "
            f"timeout_ms={camera.get('timeout_ms')} issue={camera.get('issue')!r}",
        )
        for camera_name, camera_item in dict(camera.get("cameras", {}) or {}).items():
            self._log_probe(
                f"camera:{camera_name}",
                bool(camera_item.get("ok")),
                f"{self.robot_id} 相机诊断 camera={camera_name} "
                f"ok={bool(camera_item.get('ok'))} owner={camera_item.get('owner')} "
                f"issue={camera_item.get('issue')!r}",
            )
        battery = diagnostics.get("battery", {})
        self.last_battery = battery or self.last_battery
        self._log_probe(
            "battery",
            bool(battery.get("ok", False)),
            f"{self.robot_id} 电池诊断 ok={bool(battery.get('ok', False))} "
            f"status={battery.get('status')} voltage={battery.get('voltage')} "
            f"percentage={battery.get('percentage')} servo={battery.get('servo_id')} issue={battery.get('issue')!r}",
        )
        if not self._diagnostics_ready(diagnostics):
            logger.warning(
                "XLeRobot 启动检查未通过: 请检查 serial_bus.port、舵机电源、底盘轮子ID、机械臂关节ID、"
                "相机 device_id 和 opencv-python 安装"
            )

    def _log_runtime_diagnostics(self) -> None:
        now = time.time()
        camera_available = bool(self.last_camera.get("frame_available"))
        camera_changed = (
            self._last_camera_available is None
            or camera_available != self._last_camera_available
        )
        if camera_changed or now - self._last_camera_log_at >= 30.0:
            self._log_runtime(
                camera_available,
                f"{self.robot_id} 相机运行时 available={camera_available} "
                f"frame={self.last_camera.get('frame_id')} shape={self.last_camera.get('image_shape')} "
                f"device={self._hardware_summary['camera_device_id']}",
            )
            self._last_camera_available = camera_available
            self._last_camera_log_at = now

        arm_ok = bool(self.last_arm_status.get("success", False))
        arm_changed = self._last_arm_ok is None or arm_ok != self._last_arm_ok
        if arm_changed or now - self._last_arm_log_at >= 30.0:
            joints = self.last_arm_status.get("joint_states") or {}
            self._log_runtime(
                arm_ok,
                f"{self.robot_id} 机械臂运行时 ok={arm_ok} "
                f"joints={len(joints)} lift_height={self.last_arm_status.get('lift_height')} "
                f"message={self.last_arm_status.get('message')!r}",
            )
            self._last_arm_ok = arm_ok
            self._last_arm_log_at = now
            for arm_name, status in self.last_arms_status.items():
                named_arm_ok = bool(status.get("success", False))
                self._log_runtime(
                    named_arm_ok,
                    f"{self.robot_id} 机械臂运行时 arm={arm_name} ok={named_arm_ok} "
                    f"message={status.get('message')!r}",
                )

        battery_status = str(self.last_battery.get("status") or "unknown")
        battery_changed = (
            self._last_battery_status is None
            or battery_status != self._last_battery_status
        )
        if battery_changed or now - self._last_battery_log_at >= 30.0:
            ok = bool(self.last_battery.get("ok", False)) and battery_status not in {
                "critical",
                "unknown",
            }
            self._log_runtime(
                ok,
                f"{self.robot_id} 电池运行时 status={battery_status} "
                f"voltage={self.last_battery.get('voltage')} percentage={self.last_battery.get('percentage')} "
                f"servo={self.last_battery.get('servo_id')} issue={self.last_battery.get('issue')!r}",
            )
            self._last_battery_status = battery_status
            self._last_battery_log_at = now

    @staticmethod
    def _log_probe(_service: str, ok: bool, message: str) -> None:
        if ok:
            logger.info(message)
        else:
            logger.warning(message)

    @staticmethod
    def _log_runtime(ok: bool, message: str) -> None:
        if ok:
            logger.info(message)
        else:
            logger.warning(message)

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
        return "; ".join(failed) if failed else "unknown hardware diagnostic failure"
