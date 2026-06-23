from __future__ import annotations

import asyncio
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import grpc
from google.protobuf.struct_pb2 import Struct

from hey_robot.capability.contract.v1 import capability_pb2, capability_pb2_grpc
from hey_robot.config import CapabilityServiceSpec, DeploymentConfig
from hey_robot.logging import HeyRobotLogger

logger = HeyRobotLogger(name="capability.vla")

DEFAULT_ARM_CALIBRATION_DIR = "~/.cache/hey_robot/calibrations/robots/so_follower/"


@dataclass
class VLAServiceState:
    service_id: str
    spec: CapabilityServiceSpec
    busy: bool = False
    current_skill_id: str | None = None
    last_error: str | None = None
    last_result: dict[str, Any] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


class LeRobotVLAExecutor:
    """Runs a LeRobot single-arm VLA as a capability service."""

    def __init__(self, service_id: str, spec: CapabilityServiceSpec) -> None:
        self.service_id = service_id
        self.spec = spec
        self._active_policy_client: Any | None = None

    def health(self) -> dict[str, Any]:
        missing = self._missing_config(self._base_config({}))
        return {
            "name": self.service_id,
            "online": True,
            "loaded": not missing,
            "robot_id": self.spec.robot_id,
            "error": f"missing VLA configuration: {', '.join(missing)}"
            if missing
            else None,
            "metrics": {
                "type": self.spec.type,
                "policy_type": self.spec.settings.get("policy_type"),
                "model_path": self.spec.settings.get("model_path")
                or self.spec.settings.get("policy_name"),
                "runtime": self.spec.settings.get("runtime", "lerobot_single_arm"),
            },
        }

    def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        config = self._base_config(payload)
        missing = self._missing_config(config)
        if missing:
            return {
                "success": False,
                "status": "failed",
                "failure_mode": "invalid_configuration",
                "summary": f"missing VLA configuration: {', '.join(missing)}",
                "metrics": {"vla": self._public_config(config)},
            }

        started_at = time.time()
        timeout_sec = float(config["timeout_sec"])
        timeout_fired = threading.Event()
        try:
            (
                robot_client_cls,
                robot_client_config_cls,
                so_follower_config_cls,
                camera_config_cls,
            ) = self._lerobot_classes()
            robot_config = self._build_robot_config(
                so_follower_config_cls, camera_config_cls, config
            )
            runtime_config = robot_client_config_cls(
                robot=robot_config,
                task=str(config["task"]),
                server_address=str(config["server_address"]),
                policy_type=str(config["policy_type"]),
                pretrained_name_or_path=str(config["model_path"]),
                policy_device=str(config["policy_device"]),
                actions_per_chunk=int(config["actions_per_chunk"]),
                chunk_size_threshold=float(config.get("chunk_size_threshold", 0.5)),
                fps=int(config["fps"]),
            )

            policy_client = robot_client_cls(runtime_config)
            self._active_policy_client = policy_client
            if not policy_client.start():
                return {
                    "success": False,
                    "status": "failed",
                    "failure_mode": "policy_server_unavailable",
                    "summary": "failed to connect to LeRobot VLA policy server",
                    "metrics": {"vla": self._public_config(config)},
                }

            timer = threading.Timer(
                timeout_sec, self._stop_due_to_timeout, args=(timeout_fired,)
            )
            timer.start()
            receiver = threading.Thread(
                target=policy_client.receive_actions, daemon=True
            )
            receiver.start()
            try:
                policy_client.control_loop(task=str(config["task"]))
            except Exception as exc:
                if not timeout_fired.is_set():
                    return {
                        "success": False,
                        "status": "failed",
                        "failure_mode": "execution_failed",
                        "summary": f"VLA control loop failed: {type(exc).__name__}: {exc}",
                        "error": str(exc),
                        "metrics": {"vla": self._public_config(config)},
                    }
            finally:
                timer.cancel()
                self.cancel()

            return {
                "success": True,
                "status": "completed",
                "summary": "Arm manipulation done",
                "metrics": {
                    "duration_sec": round(time.time() - started_at, 3),
                    "timed_out": timeout_fired.is_set(),
                    "vla": self._public_config(config),
                },
            }
        except ImportError as exc:
            return {
                "success": False,
                "status": "failed",
                "failure_mode": "missing_dependency",
                "summary": f"LeRobot VLA dependencies are unavailable: {exc}",
                "error": str(exc),
                "metrics": {"vla": self._public_config(config)},
            }
        except Exception as exc:
            return {
                "success": False,
                "status": "failed",
                "failure_mode": "execution_failed",
                "summary": f"{self.spec.settings.get('tool_name', 'vla_manipulation')} failed: {type(exc).__name__}: {exc}",
                "error": str(exc),
                "metrics": {"vla": self._public_config(config)},
            }
        finally:
            self._active_policy_client = None

    def cancel(self) -> None:
        client = self._active_policy_client
        if client is not None:
            stop = getattr(client, "stop", None)
            if callable(stop):
                with suppress(Exception):
                    stop()

    def _stop_due_to_timeout(self, timeout_fired: threading.Event) -> None:
        timeout_fired.set()
        self.cancel()

    def _base_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        arguments = dict(payload.get("arguments", {}) or {})
        execution_time = (
            self.spec.settings.get("execution_time")
            or self.spec.settings.get("execution_time_sec")
            or self.spec.timeout_sec
        )
        config = {
            "server_address": self.spec.settings.get("server_address"),
            "model_path": self.spec.settings.get("policy_name")
            or self.spec.settings.get("model_path"),
            "policy_type": self.spec.settings.get("policy_type"),
            "arm_port": self.spec.settings.get("arm_port"),
            "camera_config": dict(self.spec.settings.get("camera_config", {}) or {}),
            "camera_source": self.spec.settings.get("camera_source", "opencv"),
            "task": self.spec.settings.get("task_prompt")
            or self.spec.settings.get("task"),
            "policy_device": self.spec.settings.get("policy_device", "cuda"),
            "fps": int(self.spec.settings.get("fps", 30)),
            "actions_per_chunk": int(self.spec.settings.get("actions_per_chunk", 50)),
            "timeout_sec": float(execution_time),
            "calibration_dir": self.spec.settings.get(
                "calibration_dir", DEFAULT_ARM_CALIBRATION_DIR
            ),
            "robot_id": self.spec.settings.get("vla_robot_id", "robot_arm"),
            "chunk_size_threshold": float(
                self.spec.settings.get("chunk_size_threshold", 0.5)
            ),
            "load_on_startup": bool(self.spec.settings.get("load_on_startup", False)),
            "tool_name": self.spec.settings.get("tool_name", "vla_manipulation"),
            "tool_description": self.spec.settings.get("tool_description", ""),
            "arm_side": self.spec.settings.get("arm_side"),
        }
        config.update(
            {key: value for key, value in arguments.items() if value is not None}
        )
        if payload.get("timeout_sec") is not None:
            config["timeout_sec"] = float(payload["timeout_sec"])
        if arguments.get("execution_time") is not None:
            config["timeout_sec"] = float(arguments["execution_time"])
        if not config.get("task"):
            config["task"] = (
                arguments.get("task_prompt")
                or payload.get("objective")
                or arguments.get("objective")
            )
        if not config.get("arm_side"):
            config["arm_side"] = _infer_arm_side(config.get("arm_port"))
        return config

    @staticmethod
    def _missing_config(config: dict[str, Any]) -> list[str]:
        missing = [
            key
            for key in (
                "server_address",
                "model_path",
                "policy_type",
                "arm_port",
                "task",
            )
            if not config.get(key)
        ]
        if (
            not isinstance(config.get("camera_config"), dict)
            or not config["camera_config"]
        ):
            missing.append("camera_config")
        return missing

    def _build_robot_config(
        self, so_follower_config: Any, camera_config_cls: Any, config: dict[str, Any]
    ) -> Any:
        cameras = {}
        for name, settings in dict(config["camera_config"]).items():
            cameras[str(name)] = camera_config_cls(
                index_or_path=settings.get(
                    "index_or_path", settings.get("device_id", 0)
                ),
                width=settings.get("width", 640),
                height=settings.get("height", 480),
                fps=settings.get("fps", int(config["fps"])),
            )
        robot_config = so_follower_config(port=str(config["arm_port"]), cameras=cameras)
        robot_config.type = "so101_follower"
        robot_config.id = str(config["robot_id"])
        if config.get("calibration_dir"):
            robot_config.calibration_dir = Path(
                str(config["calibration_dir"])
            ).expanduser()
        return robot_config

    @staticmethod
    def _lerobot_classes() -> tuple[Any, Any, Any, Any]:
        from lerobot.async_inference.configs import RobotClientConfig
        from lerobot.async_inference.robot_client import RobotClient
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig

        return RobotClient, RobotClientConfig, SOFollowerConfig, OpenCVCameraConfig

    @staticmethod
    def _public_config(config: dict[str, Any]) -> dict[str, Any]:
        return {
            "server_address": config.get("server_address"),
            "model_path": config.get("model_path"),
            "policy_type": config.get("policy_type"),
            "arm_port": config.get("arm_port"),
            "camera_source": config.get("camera_source"),
            "camera_names": sorted(dict(config.get("camera_config") or {}).keys()),
            "policy_device": config.get("policy_device"),
            "fps": config.get("fps"),
            "actions_per_chunk": config.get("actions_per_chunk"),
            "timeout_sec": config.get("timeout_sec"),
            "arm_side": config.get("arm_side"),
            "runtime": "lerobot_single_arm",
        }


def _infer_arm_side(arm_port: Any) -> str | None:
    lowered = str(arm_port or "").lower()
    if "right" in lowered:
        return "right"
    if "left" in lowered:
        return "left"
    return None


class VLACapabilityServicer(capability_pb2_grpc.CapabilityServiceServicer):
    def __init__(self, state: VLAServiceState, executor: LeRobotVLAExecutor) -> None:
        self.state = state
        self.executor = executor

    async def GetHealth(self, request, context):  # type: ignore[override]
        del request, context
        payload = self.executor.health()
        metrics = {
            **dict(payload.get("metrics", {}) or {}),
            **dict(self.state.metrics),
            "last_result": self.state.last_result,
        }
        return capability_pb2.GetHealthResponse(
            service_id=self.state.service_id,
            name=str(payload.get("name") or self.state.service_id),
            robot_id=str(payload.get("robot_id") or self.state.spec.robot_id),
            online=bool(payload.get("online", True)),
            loaded=bool(payload.get("loaded", True)),
            busy=bool(self.state.busy),
            current_skill_id=self.state.current_skill_id or "",
            error_message=str(self.state.last_error or payload.get("error") or ""),
            metrics=_dict_to_struct(metrics),
            version="grpc-v1",
        )

    async def ExecuteCapability(self, request, context):  # type: ignore[override]
        del context
        if self.state.busy:
            return capability_pb2.ExecuteCapabilityResponse(
                success=False,
                status="failed",
                summary=f"capability {self.state.service_id} is busy",
                failure_mode="capability_busy",
                error_code="CAPABILITY_BUSY",
                error_message=f"capability {self.state.service_id} is busy",
            )
        self.state.busy = True
        self.state.current_skill_id = request.skill_id or None
        self.state.last_error = None
        payload = {
            "service_id": request.service_id,
            "trace_id": request.trace_id,
            "episode_id": request.episode_id,
            "skill_id": request.skill_id,
            "skill_name": request.skill_name,
            "robot_id": request.robot_id,
            "objective": request.objective,
            "arguments": _struct_to_dict(request.arguments),
            "timeout_sec": request.timeout_sec,
            "metadata": _struct_to_dict(request.metadata),
        }
        try:
            result = await asyncio.to_thread(self.executor.execute, payload)
        finally:
            self.state.busy = False
            self.state.current_skill_id = None
        self.state.last_result = result
        self.state.last_error = (
            (str(result.get("error") or "") or None)
            if not result.get("success")
            else None
        )
        return capability_pb2.ExecuteCapabilityResponse(
            success=bool(result.get("success", False)),
            status=str(
                result.get("status")
                or ("completed" if result.get("success") else "failed")
            ),
            summary=str(result.get("summary") or ""),
            failure_mode=str(result.get("failure_mode") or ""),
            error_code=str(result.get("error_code") or ""),
            error_message=str(result.get("error") or ""),
            metrics=_dict_to_struct(dict(result.get("metrics", {}) or {})),
        )

    async def CancelCapability(self, request, context):  # type: ignore[override]
        del request, context
        self.executor.cancel()
        return capability_pb2.CancelCapabilityResponse(
            accepted=True, summary="cancel requested"
        )


class VLACapabilityService:
    def __init__(
        self,
        config: DeploymentConfig,
        *,
        service_id: str,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        self.config = config
        self.service_id = service_id
        self.spec = config.capability_services[service_id]
        self.host = host or str(self.spec.settings.get("host", "127.0.0.1"))
        self.port = port or int(self.spec.settings.get("port", 9090))
        self.state = VLAServiceState(service_id, self.spec)
        self.executor = LeRobotVLAExecutor(service_id, self.spec)
        self._server: grpc.aio.Server | None = None

    async def start(self) -> None:
        self._server = grpc.aio.server()
        capability_pb2_grpc.add_CapabilityServiceServicer_to_server(
            VLACapabilityServicer(self.state, self.executor),
            self._server,
        )
        bind_target = f"{self.host}:{self.port}"
        self._server.add_insecure_port(bind_target)
        logger.info(
            f"VLA capability [{self.service_id}] listening on grpc://{bind_target}"
        )
        await self._server.start()
        await self._server.wait_for_termination()

    async def stop(self) -> None:
        self.executor.cancel()
        if self._server is not None:
            await self._server.stop(grace=0.5)


def _dict_to_struct(value: dict[str, Any]) -> Struct:
    message = Struct()
    message.update(value)
    return message


def _struct_to_dict(value: Struct) -> dict[str, Any]:
    return dict(value) if value is not None else {}
