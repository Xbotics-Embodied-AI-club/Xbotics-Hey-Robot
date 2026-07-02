"""LeRobot VLA executor — runs a single-arm manipulation policy.

Current form: bundles inference + control loop + hardware access.
Target form (VLA Step 2): stateless inference only, control loop moves to Skill OS.
"""

from __future__ import annotations

import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from hey_robot.config import ModelServiceSpec

DEFAULT_ARM_CALIBRATION_DIR = "~/.cache/hey_robot/calibrations/robots/so_follower/"


class LeRobotVLAExecutor:
    """Runs a LeRobot single-arm VLA as a model service."""

    def __init__(self, service_id: str, spec: ModelServiceSpec) -> None:
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


class LeRobotVLAPolicyExecutor:
    """Stateless VLA inference executor — one image in, one action out.

    Supports mock mode (returns fake joint actions) for testing the full
    flow without a real LeRobot policy server.
    """

    def __init__(self, service_id: str, spec: ModelServiceSpec) -> None:
        self.service_id = service_id
        self.spec = spec

    def health(self) -> dict[str, Any]:
        mock_mode = self._mock_mode()
        loaded = mock_mode or bool(
            self.spec.settings.get("server_address")
            and self.spec.settings.get("model_path")
        )
        return {
            "name": self.service_id,
            "online": True,
            "loaded": loaded,
            "robot_id": self.spec.robot_id,
            "error": None
            if loaded
            else "VLA policy server or model_path not configured",
            "metrics": {
                "type": self.spec.type,
                "mock_mode": mock_mode,
                "policy_type": self.spec.settings.get("policy_type"),
                "runtime": "lerobot_policy_inference",
            },
        }

    def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._mock_mode():
            return self._mock_inference(payload)
        return self._real_inference(payload)

    def cancel(self) -> None:
        pass

    # -- mock mode ----------------------------------------------------------

    def _mock_mode(self) -> bool:
        settings = self.spec.settings
        if "mock_mode" in settings:
            return bool(settings.get("mock_mode"))
        if not settings.get("server_address"):
            return True
        return bool(not settings.get("model_path"))

    @staticmethod
    def _mock_inference(payload: dict[str, Any]) -> dict[str, Any]:
        """Return fake VLA inference: a reach-then-grasp action sequence.

        The control loop in Skill OS drives multi-step execution.
        Each single call returns one set of joint targets.
        """
        arguments = dict(payload.get("arguments", {}) or {})
        task = str(
            arguments.get("task_prompt") or payload.get("objective") or "manipulate"
        ).lower()
        step = int(
            arguments.get("vla_step")
            or payload.get("metadata", {}).get("vla_step", 0)
            or 0
        )

        # Simulated 3-phase action: reach → grasp → lift
        if "pick" in task or "grasp" in task or "拿" in task or "抓" in task:
            if step == 0:
                # Reach toward object
                return _vla_result(
                    joint_angles={
                        "shoulder_lift": 0.3,
                        "shoulder_pan": 0.15,
                        "elbow_flex": 0.4,
                        "wrist_flex": 0.2,
                        "wrist_roll": 0.0,
                    },
                    gripper_action=1.0,  # open
                    task_done=False,
                )
            if step == 1:
                # Grasp
                return _vla_result(
                    joint_angles={
                        "shoulder_lift": 0.35,
                        "shoulder_pan": 0.15,
                        "elbow_flex": 0.45,
                        "wrist_flex": 0.2,
                        "wrist_roll": 0.0,
                    },
                    gripper_action=0.2,  # close
                    task_done=False,
                )
            # Lift
            return _vla_result(
                joint_angles={
                    "shoulder_lift": 0.1,
                    "shoulder_pan": 0.0,
                    "elbow_flex": 0.8,
                    "wrist_flex": 0.0,
                    "wrist_roll": 0.0,
                },
                gripper_action=0.2,
                task_done=True,
            )
        if "place" in task or "put" in task or "放" in task:
            if step == 0:
                return _vla_result(
                    joint_angles={
                        "shoulder_lift": 0.1,
                        "shoulder_pan": 0.0,
                        "elbow_flex": 0.8,
                        "wrist_flex": 0.0,
                        "wrist_roll": 0.0,
                    },
                    gripper_action=0.2,
                    task_done=False,
                )
            return _vla_result(
                joint_angles={
                    "shoulder_lift": 0.3,
                    "shoulder_pan": 0.15,
                    "elbow_flex": 0.4,
                    "wrist_flex": 0.2,
                    "wrist_roll": 0.0,
                },
                gripper_action=1.0,
                task_done=True,
            )
        return _vla_result(
            joint_angles={
                "shoulder_lift": 0.2,
                "shoulder_pan": 0.0,
                "elbow_flex": 0.5,
                "wrist_flex": 0.1,
                "wrist_roll": 0.0,
            },
            gripper_action=0.5,
            task_done=True,
        )

    def _real_inference(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run a single VLA inference step via LeRobot policy server.

        This is a single frame → action call, NOT a control loop.
        The control loop lives in Skill OS.
        """
        try:
            (
                robot_client_cls,
                robot_client_config_cls,
                so_follower_config_cls,
                camera_config_cls,
            ) = self._lerobot_classes()
        except ImportError as exc:
            return {
                "success": False,
                "status": "failed",
                "failure_mode": "missing_dependency",
                "summary": f"LeRobot VLA dependencies unavailable: {exc}",
                "error": str(exc),
            }

        try:
            config = self._base_config(payload)
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
            if not policy_client.start():
                return {
                    "success": False,
                    "status": "failed",
                    "failure_mode": "policy_server_unavailable",
                    "summary": "failed to connect to LeRobot VLA policy server",
                }
            # Single inference — not control_loop
            action = policy_client.get_action(str(config["task"]))
            return {
                "success": True,
                "status": "completed",
                "summary": "VLA inference completed",
                "metrics": {
                    "vla": {
                        "joint_angles": getattr(action, "joint_angles", None),
                        "gripper_action": getattr(action, "gripper_action", None),
                        "task_done": getattr(action, "task_done", False),
                    }
                },
            }
        except Exception as exc:
            return {
                "success": False,
                "status": "failed",
                "failure_mode": "execution_failed",
                "summary": f"VLA inference failed: {type(exc).__name__}: {exc}",
                "error": str(exc),
            }

    def _base_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        arguments = dict(payload.get("arguments", {}) or {})
        return {
            "server_address": self.spec.settings.get("server_address", ""),
            "model_path": self.spec.settings.get("model_path")
            or self.spec.settings.get("policy_name", ""),
            "policy_type": self.spec.settings.get("policy_type", "act"),
            "arm_port": self.spec.settings.get("arm_port", ""),
            "camera_config": dict(self.spec.settings.get("camera_config", {}) or {}),
            "task": arguments.get("task_prompt")
            or self.spec.settings.get("task_prompt")
            or payload.get("objective", "manipulate"),
            "policy_device": self.spec.settings.get("policy_device", "cuda"),
            "fps": int(self.spec.settings.get("fps", 30)),
            "actions_per_chunk": int(self.spec.settings.get("actions_per_chunk", 50)),
            "timeout_sec": float(payload.get("timeout_sec", 30.0)),
        }

    @staticmethod
    def _lerobot_classes() -> tuple[Any, Any, Any, Any]:
        from lerobot.async_inference.configs import RobotClientConfig
        from lerobot.async_inference.robot_client import RobotClient
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig

        return RobotClient, RobotClientConfig, SOFollowerConfig, OpenCVCameraConfig

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
        robot_config.id = str(self.spec.settings.get("vla_robot_id", "robot_arm"))
        return robot_config


def _vla_result(
    joint_angles: dict[str, float],
    gripper_action: float,
    task_done: bool,
) -> dict[str, Any]:
    return {
        "success": True,
        "status": "completed",
        "summary": "VLA inference completed",
        "metrics": {
            "vla": {
                "joint_angles": joint_angles,
                "gripper_action": gripper_action,
                "task_done": task_done,
                "mode": "mock",
            }
        },
    }
