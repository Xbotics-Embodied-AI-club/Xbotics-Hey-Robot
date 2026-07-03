"""LeRobot VLA executor — runs a single-arm manipulation policy.

Current form: bundles inference + control loop + hardware access.
Target form (VLA Step 2): stateless inference only, control loop moves to Skill OS.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import request as urllib_request
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image
from PIL.Image import Resampling

from hey_robot.config import ModelServiceSpec
from hey_robot.foundation.clients.models import PolicyStepResult

DEFAULT_ARM_CALIBRATION_DIR = "~/.cache/hey_robot/calibrations/robots/so_follower/"

# ── ACT model utilities ────────────────────────────────────────────────────

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Fallback action normalization stats for XLeRobot single-arm joint space.
# Joint order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
_ACT_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
_DEFAULT_ACTION_MEAN = (0.2808, -51.1163, 45.7853, 77.6119, 1.6111, 8.2912)
_DEFAULT_ACTION_STD = (9.3055, 53.7685, 52.8102, 9.1411, 5.3389, 10.5236)

_RAD_TO_DEG = 180.0 / 3.141592653589793
_DEG_TO_RAD = 3.141592653589793 / 180.0

_CAMERA_KEY_MAP = {
    "front": "observation.images.front",
    "handeye": "observation.images.handeye",
    "right_wrist": "observation.images.handeye",
    "left_wrist": "observation.images.handeye",
}


@dataclass
class _NormalizationStats:
    mean: torch.Tensor
    std: torch.Tensor

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std.clamp(min=1e-8)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


def _extract_input_stats(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, _NormalizationStats]:
    """Extract per-feature input normalization stats from a LeRobot policy checkpoint."""
    stats: dict[str, _NormalizationStats] = {}
    for key, tensor in state_dict.items():
        if not key.startswith("normalize_inputs.buffer_"):
            continue
        remainder = key[len("normalize_inputs.buffer_") :]
        parts = remainder.rsplit(".", 1)
        if len(parts) != 2:
            continue
        feature, stat_type = parts
        feature = feature.replace("_", ".")
        if feature not in stats:
            stats[feature] = _NormalizationStats(
                mean=torch.zeros_like(tensor), std=torch.ones_like(tensor)
            )
        if stat_type == "mean":
            stats[feature].mean = tensor.clone()
        elif stat_type == "std":
            stats[feature].std = tensor.clone()
    return stats


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

            elapsed = time.time() - started_at
            return {
                "success": True,
                "status": "completed",
                "summary": "Arm manipulation done",
                "metrics": {
                    "duration_sec": round(elapsed, 3),
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
                "summary": f"{self.spec.settings.get('skill_name', 'manipulate')} failed: {type(exc).__name__}: {exc}",
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
            "skill_name": self.spec.settings.get("skill_name", "manipulate"),
            "skill_description": self.spec.settings.get("skill_description", ""),
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

    Supports:
      - mock mode (returns fake joint actions) for testing
      - direct ACT model loading/inference (no external HTTP server needed)
      - legacy HTTP endpoint fallback (action_chunk_endpoint)
    """

    def __init__(self, service_id: str, spec: ModelServiceSpec) -> None:
        self.service_id = service_id
        self.spec = spec
        self._policy: Any = None
        self._policy_stats: dict[str, _NormalizationStats] = {}
        self._state_stat: _NormalizationStats | None = None
        self._action_stat: _NormalizationStats | None = None
        self._imagenet_mean: torch.Tensor | None = None
        self._imagenet_std: torch.Tensor | None = None

    def health(self) -> dict[str, Any]:
        mock_mode = self._mock_mode()
        backend_mode = self._backend_mode()
        has_model = bool(self.spec.settings.get("model_path"))
        has_endpoint = bool(self.spec.settings.get("action_chunk_endpoint"))
        loaded = mock_mode or has_model or has_endpoint
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
                "backend_mode": backend_mode,
                "hardware_ownership": "none"
                if backend_mode == "action_chunk_policy"
                else "legacy_lerobot_client",
            },
        }

    def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._mock_mode():
            return self._mock_inference(payload)
        if self._backend_mode() != "action_chunk_policy":
            return {
                "success": False,
                "status": "failed",
                "failure_mode": "unsupported_vla_backend_mode",
                "summary": (
                    "LeRobot RobotClient is a legacy hardware-owning control loop; "
                    "use backend_mode=action_chunk_policy for Foundation inference"
                ),
                "metrics": {
                    "vla": {
                        "backend_mode": self._backend_mode(),
                        "hardware_ownership": "legacy_lerobot_client",
                    }
                },
            }
        return self._real_inference(payload)

    def cancel(self) -> None:
        pass

    # -- mock mode ----------------------------------------------------------

    def _mock_mode(self) -> bool:
        settings = self.spec.settings
        if "mock_mode" in settings:
            return bool(settings.get("mock_mode"))
        return not (settings.get("model_path") or settings.get("action_chunk_endpoint"))

    def _backend_mode(self) -> str:
        settings = self.spec.settings
        value = str(
            settings.get("backend_mode")
            or settings.get("mode")
            or settings.get("backend")
            or "action_chunk_policy"
        )
        if value in {"lerobot", "lerobot_single_arm"}:
            return "lerobot_control_loop"
        return value

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
        arguments = dict(payload.get("arguments", {}) or {})
        observation = arguments.get("observation") or payload.get("observation")
        if not isinstance(observation, dict) or not observation.get("images"):
            return {
                "success": False,
                "status": "failed",
                "failure_mode": "observation_unavailable",
                "summary": "VLA action_chunk_policy requires observation.images",
            }

        model_path = self.spec.settings.get("model_path")
        if model_path:
            return self._direct_act_inference(
                str(model_path),
                observation=observation,
            )

        endpoint = self.spec.settings.get("action_chunk_endpoint")
        if endpoint:
            return self._call_action_chunk_endpoint(
                str(endpoint),
                payload=payload,
                arguments=arguments,
                observation=observation,
            )

        return {
            "success": False,
            "status": "failed",
            "failure_mode": "action_chunk_policy_client_unavailable",
            "summary": "no VLA model_path or action_chunk_endpoint configured",
            "metrics": {
                "vla": {
                    "backend_mode": "action_chunk_policy",
                    "frame_id": observation.get("frame_id"),
                    "hardware_ownership": "none",
                }
            },
        }

    def _call_action_chunk_endpoint(
        self,
        endpoint: str,
        *,
        payload: dict[str, Any],
        arguments: dict[str, Any],
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        task = (
            arguments.get("task_prompt")
            or arguments.get("objective")
            or payload.get("objective")
            or "manipulate"
        )
        request_payload = {
            "policy_session_id": arguments.get("policy_session_id")
            or payload.get("skill_id"),
            "skill_name": arguments.get("skill_name") or payload.get("skill_name"),
            "atomic_command": str(task),
            "observation": observation,
            "proprioception": observation.get("proprioception") or {},
            "frame_id": observation.get("frame_id"),
            "metadata": {
                "service_id": self.service_id,
                "robot_id": payload.get("robot_id") or self.spec.robot_id,
                "embodiment": self.spec.settings.get("embodiment", "xlerobot"),
            },
        }
        parsed_endpoint = urlparse(endpoint)
        if parsed_endpoint.scheme not in {"http", "https"}:
            return {
                "success": False,
                "status": "failed",
                "failure_mode": "action_chunk_policy_invalid_endpoint",
                "summary": "VLA action chunk endpoint must use http or https",
            }
        body = json.dumps(request_payload).encode("utf-8")
        timeout_sec = float(
            arguments.get("timeout_sec")
            or payload.get("timeout_sec")
            or self.spec.timeout_sec
            or 30.0
        )
        http_request = urllib_request.Request(  # noqa: S310 - scheme is validated above.
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(  # noqa: S310 - scheme is validated above.
                http_request, timeout=timeout_sec
            ) as response:
                raw = response.read().decode("utf-8")
        except Exception as exc:
            return {
                "success": False,
                "status": "failed",
                "failure_mode": "action_chunk_policy_unavailable",
                "summary": f"VLA action chunk endpoint failed: {type(exc).__name__}: {exc}",
                "error": str(exc),
                "metrics": {
                    "vla": {
                        "backend_mode": "action_chunk_policy",
                        "frame_id": observation.get("frame_id"),
                        "hardware_ownership": "none",
                    }
                },
            }
        try:
            decoded = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            return {
                "success": False,
                "status": "failed",
                "failure_mode": "action_chunk_policy_invalid_response",
                "summary": "VLA action chunk endpoint returned invalid JSON",
                "error": str(exc),
            }
        return _action_chunk_policy_result(decoded, observation=observation)

    # -- direct ACT model inference -------------------------------------------

    def _load_act_policy(self, model_path: str) -> None:
        """Lazy-load the ACT policy model and normalization stats."""
        if self._policy is not None:
            return

        from lerobot.policies.factory import get_policy_class
        from safetensors.torch import load_file as load_safetensors

        device = str(self.spec.settings.get("policy_device") or "cuda")
        model_dir = Path(model_path)

        state_dict = load_safetensors(str(model_dir / "model.safetensors"))
        config = json.loads((model_dir / "config.json").read_text())

        stats = _extract_input_stats(state_dict)
        self._state_stat = stats.get("observation.state")
        self._action_stat = _NormalizationStats(
            mean=torch.tensor(_DEFAULT_ACTION_MEAN, device=device),
            std=torch.tensor(_DEFAULT_ACTION_STD, device=device),
        )
        for s in (self._state_stat, self._action_stat):
            if s is not None:
                s.mean = s.mean.to(device)
                s.std = s.std.to(device)

        self._imagenet_mean = torch.tensor(_IMAGENET_MEAN, device=device).view(3, 1, 1)
        self._imagenet_std = torch.tensor(_IMAGENET_STD, device=device).view(3, 1, 1)

        policy_class = get_policy_class(config["type"])
        self._policy = policy_class.from_pretrained(str(model_dir))
        self._policy.to(device)
        self._policy.eval()

    def _direct_act_inference(
        self,
        model_path: str,
        *,
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            self._load_act_policy(model_path)
        except ImportError as exc:
            return {
                "success": False,
                "status": "failed",
                "failure_mode": "missing_dependency",
                "summary": f"ACT model dependencies unavailable: {exc}",
                "error": str(exc),
            }
        except Exception as exc:
            return {
                "success": False,
                "status": "failed",
                "failure_mode": "model_load_failed",
                "summary": f"Failed to load ACT model from {model_path}: {exc}",
                "error": str(exc),
            }

        try:
            obs_tensors = self._preprocess_act_observation(observation)
        except Exception as exc:
            return {
                "success": False,
                "status": "failed",
                "failure_mode": "preprocessing_failed",
                "summary": f"ACT preprocessing failed: {exc}",
                "error": str(exc),
            }

        try:
            raw = self._run_act_inference(obs_tensors)
        except Exception as exc:
            return {
                "success": False,
                "status": "failed",
                "failure_mode": "inference_failed",
                "summary": f"ACT inference failed: {type(exc).__name__}: {exc}",
                "error": str(exc),
                "metrics": {
                    "vla": {
                        "backend_mode": "action_chunk_policy",
                        "frame_id": observation.get("frame_id"),
                        "hardware_ownership": "none",
                    }
                },
            }
        actions_out = raw["actions"]
        return {
            "success": True,
            "status": "completed",
            "summary": "ACT inference completed",
            "metrics": {
                "policy_result": {
                    "kind": "action_chunk",
                    "actions": actions_out,
                    "action_space": "xlerobot_single_arm_joint",
                    "embodiment": "xlerobot",
                    "horizon": raw["horizon"],
                    "done": False,
                    "confidence": 0.8,
                },
                "action_chunk": {
                    "kind": "action_chunk",
                    "actions": actions_out,
                    "action_space": "xlerobot_single_arm_joint",
                    "embodiment": "xlerobot",
                    "horizon": raw["horizon"],
                    "done": False,
                    "confidence": 0.8,
                },
                "vla": {
                    "joint_angles": actions_out[0]["joints"] if actions_out else {},
                    "gripper_action": actions_out[0]["gripper"] if actions_out else 0.0,
                    "task_done": False,
                    "backend_mode": "action_chunk_policy",
                    "frame_id": observation.get("frame_id"),
                    "hardware_ownership": "none",
                },
            },
        }

    def _preprocess_act_observation(
        self, observation: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        device = str(self.spec.settings.get("policy_device") or "cuda")
        result: dict[str, torch.Tensor] = {}

        images = observation.get("images")
        if isinstance(images, list):
            for img_entry in images:
                camera = str(img_entry.get("camera", ""))
                data = img_entry.get("data", "")
                target_key = _CAMERA_KEY_MAP.get(camera)
                if not target_key or not data:
                    continue
                try:
                    raw = base64.b64decode(data)
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    img = img.resize((640, 480), Resampling.BILINEAR)
                    arr = np.array(img, dtype=np.float32) / 255.0
                    tensor = torch.from_numpy(arr).permute(2, 0, 1).to(device)
                    result[target_key] = tensor
                except Exception as exc:
                    logger = __import__("logging").getLogger(__name__)
                    logger.debug(
                        "Failed to decode ACT image from camera %s: %s", camera, exc
                    )
                    continue

        state = observation.get("proprioception")
        if isinstance(state, list) and len(state) >= 6:
            state_deg = [
                float(v) * _RAD_TO_DEG
                for v in np.asarray(state[:6], dtype=np.float32).flat
            ]
            result["observation.state"] = torch.tensor(
                state_deg, dtype=torch.float32, device=device
            )
        elif isinstance(state, (int, float)):
            result["observation.state"] = torch.tensor(
                [float(state)], dtype=torch.float32, device=device
            )

        return result

    def _run_act_inference(self, obs: dict[str, torch.Tensor]) -> dict[str, Any]:
        assert self._action_stat is not None, "ACT model not loaded"
        assert self._imagenet_mean is not None, "ACT model not loaded"
        assert self._imagenet_std is not None, "ACT model not loaded"
        batch: dict[str, torch.Tensor] = {}
        for key, value in obs.items():
            if "state" in key and self._state_stat is not None:
                value = self._state_stat.normalize(value)
            elif "image" in key:
                value = (value - self._imagenet_mean) / self._imagenet_std
            batch[key] = value.unsqueeze(0)

        with torch.no_grad():
            action_chunk = self._policy.predict_action_chunk(batch)
            if action_chunk.ndim == 3:
                action_chunk = action_chunk.squeeze(0)

        action_chunk = self._action_stat.unnormalize(action_chunk)
        actions_raw = action_chunk.cpu().tolist()

        actions_out: list[dict[str, Any]] = []
        for action_vec in actions_raw:
            joints: dict[str, float] = {}
            for i, name in enumerate(_ACT_JOINT_NAMES):
                if i < len(action_vec):
                    joints[name] = round(float(action_vec[i]) * _DEG_TO_RAD, 6)
            gripper_rad = joints.pop("gripper", 0.0)
            gripper_pct = max(0.0, min(1.0, (gripper_rad + 0.1) / 1.3))
            actions_out.append({"joints": joints, "gripper": gripper_pct})

        return {"actions": actions_out, "horizon": len(actions_out)}

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
    action = {
        "joints": joint_angles,
        "gripper": gripper_action,
        "done": task_done,
    }
    policy_result = PolicyStepResult(
        kind="action_chunk",
        action_space="xlerobot_single_arm_joint",
        embodiment="xlerobot",
        horizon=1,
        dt=1.0 / 30.0,
        actions=[action],
        done=task_done,
        confidence=0.5,
        raw={"mode": "mock"},
    ).to_metrics()
    return {
        "success": True,
        "status": "completed",
        "summary": "VLA inference completed",
        "metrics": {
            "policy_result": policy_result,
            "action_chunk": policy_result,
            "vla": {
                "joint_angles": joint_angles,
                "gripper_action": gripper_action,
                "task_done": task_done,
                "mode": "mock",
                "backend_mode": "action_chunk_policy",
            },
        },
    }


def _action_chunk_policy_result(
    payload: dict[str, Any], *, observation: dict[str, Any]
) -> dict[str, Any]:
    policy_result = payload.get("policy_result")
    if not isinstance(policy_result, dict):
        action_chunk = payload.get("action_chunk")
        if isinstance(action_chunk, dict):
            policy_result = dict(action_chunk)
        else:
            actions = payload.get("actions")
            policy_result = {
                "kind": "action_chunk",
                "action_space": payload.get("action_space")
                or "xlerobot_single_arm_joint",
                "embodiment": payload.get("embodiment") or "xlerobot",
                "horizon": payload.get("horizon")
                or (len(actions) if isinstance(actions, list) else None),
                "dt": payload.get("dt"),
                "actions": actions if isinstance(actions, list) else [],
                "done": bool(payload.get("done", False)),
                "confidence": payload.get("confidence"),
                "valid": bool(payload.get("valid", True)),
                "raw": dict(payload.get("raw", {}) or {}),
            }
    if policy_result.get("kind") != "action_chunk":
        return {
            "success": False,
            "status": "failed",
            "failure_mode": "action_chunk_policy_invalid_response",
            "summary": "VLA policy_result.kind must be action_chunk",
            "metrics": {"policy_result": policy_result},
        }
    actions = policy_result.get("actions")
    if not isinstance(actions, list) or not actions:
        return {
            "success": False,
            "status": "failed",
            "failure_mode": "action_chunk_policy_invalid_response",
            "summary": "VLA action_chunk policy_result requires at least one action",
            "metrics": {"policy_result": policy_result},
        }
    first = dict(actions[0])
    joint_angles = dict(
        first.get("joints")
        or first.get("joint_angles")
        or first.get("single_arm")
        or {}
    )
    gripper_action = first.get("gripper")
    if gripper_action is None:
        gripper_action = first.get("gripper_action")
    done = bool(policy_result.get("done", first.get("done", False)))
    return {
        "success": True,
        "status": "completed",
        "summary": "VLA action chunk produced",
        "metrics": {
            "policy_result": policy_result,
            "action_chunk": policy_result,
            "vla": {
                "joint_angles": joint_angles,
                "gripper_action": gripper_action,
                "task_done": done,
                "backend_mode": "action_chunk_policy",
                "frame_id": observation.get("frame_id"),
                "hardware_ownership": "none",
            },
        },
    }
