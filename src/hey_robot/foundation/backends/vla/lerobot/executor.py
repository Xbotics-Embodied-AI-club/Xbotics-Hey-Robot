"""LeRobot VLA 执行器，用于运行单臂操作 policy。

当前形态：推理、控制循环和硬件访问打包在一起。
目标形态（VLA Step 2）：这里只保留无状态推理，控制循环移动到 Skill OS。
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

# ── 共享 policy 工具 ────────────────────────────────────────────────────────

_RAD_TO_DEG = 180.0 / 3.141592653589793
_DEG_TO_RAD = 3.141592653589793 / 180.0

_CAMERA_KEY_MAP: dict[str, str] = {
    "front": "observation.images.front",
    "handeye": "observation.images.handeye",
    "right_wrist": "observation.images.handeye",
    "left_wrist": "observation.images.handeye",
}

# 这些相机名应跟随 policy 声明的 wrist key。
_WRIST_CAMERA_NAMES = {"handeye", "right_wrist", "left_wrist", "wrist"}


_DEFAULT_IMAGE_SIZE = (256, 256)  # smolvla / pi0 标准输入尺寸

# SO101 机械臂使用的关节顺序；policy 可以只使用其中一部分。
_SO101_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)


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
    """从 LeRobot policy checkpoint 中提取各输入特征的归一化统计量。"""
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
    """把 LeRobot 单臂 VLA 作为模型服务运行。"""

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
    """无状态 VLA 推理执行器：输入一帧图像，输出一个动作。

    支持：
      - mock 模式：返回假的关节动作，用于测试
      - 直接加载和推理 ACT 模型：不需要外部 HTTP server
      - 旧版 HTTP endpoint fallback：action_chunk_endpoint
    """

    def __init__(self, service_id: str, spec: ModelServiceSpec) -> None:
        self.service_id = service_id
        self.spec = spec
        self._policy: Any = None
        self._policy_type: str = ""
        self._dataset_stats: dict[str, Any] = {}
        self._action_mean: torch.Tensor | None = None
        self._action_std: torch.Tensor | None = None
        self._image_size: tuple[int, int] = _DEFAULT_IMAGE_SIZE
        self._tokenizer: Any = None
        self._state_mean: torch.Tensor | None = None
        self._state_std: torch.Tensor | None = None

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

    # -- mock 模式 ----------------------------------------------------------

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
        """返回假的 VLA 推理结果：先伸手再抓取的动作序列。

        多步执行由 Skill OS 中的控制循环驱动；每次调用只返回一组关节目标。
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

        # 模拟三阶段动作：伸手 → 抓取 → 抬升
        if "pick" in task or "grasp" in task or "拿" in task or "抓" in task:
            if step == 0:
                # 伸向物体
                return _vla_result(
                    joint_angles={
                        "shoulder_lift": 0.3,
                        "shoulder_pan": 0.15,
                        "elbow_flex": 0.4,
                        "wrist_flex": 0.2,
                        "wrist_roll": 0.0,
                    },
                    gripper_action=1.0,  # 打开
                    task_done=False,
                )
            if step == 1:
                # 抓取
                return _vla_result(
                    joint_angles={
                        "shoulder_lift": 0.35,
                        "shoulder_pan": 0.15,
                        "elbow_flex": 0.45,
                        "wrist_flex": 0.2,
                        "wrist_roll": 0.0,
                    },
                    gripper_action=0.2,  # 闭合
                    task_done=False,
                )
            # 抬升
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
            return self._direct_policy_inference(
                str(model_path),
                observation=observation,
                payload=payload,
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

    # -- 直接 LeRobot policy 推理 -------------------------------------------

    def _load_policy(self, model_path: str) -> None:
        """懒加载任意 LeRobot policy，并提取动作归一化统计量。

        LeRobot policy 的 ``predict_action_chunk`` 输出都是归一化动作，这里统一手动反归一化。
        统计量来自 ``normalize_inputs`` buffer（旧格式），或 preprocessor safetensors 文件（新格式）。
        """
        if self._policy is not None:
            return

        from lerobot.policies.factory import get_policy_class

        device = str(self.spec.settings.get("policy_device") or "cuda")
        model_dir = Path(model_path)

        config = json.loads((model_dir / "config.json").read_text())

        policy_type = str(
            config.get("type") or self.spec.settings.get("policy_type", "act")
        )
        self._policy_type = policy_type

        # 解析图像尺寸：pi05 声明 image_resolution，其他 policy 使用 settings 或 input_features。
        if (
            isinstance(config.get("image_resolution"), list)
            and len(config["image_resolution"]) == 2
        ):
            h, w = config["image_resolution"]
            self._image_size = (int(w), int(h))
        else:
            image_size_raw = self.spec.settings.get("image_size")
            if isinstance(image_size_raw, (list, tuple)) and len(image_size_raw) == 2:
                self._image_size = (int(image_size_raw[0]), int(image_size_raw[1]))
            elif isinstance(image_size_raw, str) and "x" in image_size_raw.lower():
                w, h = image_size_raw.lower().split("x", 1)
                self._image_size = (int(w), int(h))
            elif isinstance(config.get("input_features"), dict):
                sizes: list[tuple[int, int]] = []
                for v in config["input_features"].values():
                    shape = v.get("shape", [])
                    if isinstance(shape, list) and len(shape) >= 3:
                        sizes.append((int(shape[2]), int(shape[1])))
                if sizes:
                    self._image_size = sizes[0]

        # 加载 policy 权重前先提取 action 统计量；不同格式可能需要单独的 safetensors 文件。
        self._extract_action_stats(model_dir, config, device)
        # Pi05 的离散文本 tokenize 流水线需要 state 统计量。
        if policy_type == "pi05":
            self._extract_state_stats(model_dir, device)

        policy_cls = get_policy_class(policy_type)
        self._policy = policy_cls.from_pretrained(str(model_dir))
        self._policy.to(device)
        self._policy.eval()

        # 根据 policy 输入特征构建动态相机 key 映射。
        self._build_camera_key_map(config)

    def _extract_action_stats(
        self, model_dir: Path, config: dict[str, Any], device: str
    ) -> None:
        """提取用于反归一化的 action mean/std。

        优先尝试新格式 preprocessor safetensors（smolvla、pi0 等），再回退到旧格式
        normalize_inputs buffer。
        """
        from safetensors.torch import load_file as load_safetensors

        logger = __import__("logging").getLogger(__name__)

        # 新格式：模型目录里的 preprocessor safetensors。
        preprocessor_stats: dict[str, torch.Tensor] | None = None
        for fname in sorted(
            model_dir.glob(
                "policy_preprocessor_step_*_normalizer_processor.safetensors"
            )
        ):
            try:
                preprocessor_stats = load_safetensors(str(fname))
                break
            except Exception as exc:
                logger.debug(f"Failed to load preprocessor stats from {fname}: {exc}")

        if preprocessor_stats is not None and "action.mean" in preprocessor_stats:
            self._action_mean = preprocessor_stats["action.mean"].to(device)
            self._action_std = (
                preprocessor_stats["action.std"].to(device).clamp(min=1e-8)
            )
            logger.debug(
                f"Extracted action stats from preprocessor: "
                f"mean={self._action_mean.tolist()}, std={self._action_std.tolist()}"
            )
            return

        # 旧格式：model.safetensors 里的 normalize_inputs buffer。
        model_file = model_dir / "model.safetensors"
        if model_file.exists():
            try:
                state_dict = load_safetensors(str(model_file))
                stats = _extract_input_stats(state_dict)
                action_stat = stats.get("action")
                if action_stat is not None:
                    self._action_mean = action_stat.mean.to(device)
                    self._action_std = action_stat.std.to(device).clamp(min=1e-8)
                    logger.debug(
                        f"Extracted action stats from normalize_inputs: "
                        f"mean={self._action_mean.tolist()}, std={self._action_std.tolist()}"
                    )
                    return
            except Exception as exc:
                logger.debug(f"Failed to extract normalize_inputs stats: {exc}")

        # 回退：单位变换（不做反归一化）。
        logger.debug(f"No action stats found for {config.get('type')}, using identity")
        self._action_mean = torch.zeros(6, device=device)
        self._action_std = torch.ones(6, device=device)

    def _extract_state_stats(self, model_dir: Path, device: str) -> None:
        """从 preprocessor safetensors 中提取 observation.state 的归一化统计量。"""
        from safetensors.torch import load_file as load_safetensors

        logger = __import__("logging").getLogger(__name__)
        for fname in sorted(
            model_dir.glob(
                "policy_preprocessor_step_*_normalizer_processor.safetensors"
            )
        ):
            try:
                preprocessor_stats = load_safetensors(str(fname))
            except Exception as exc:
                logger.debug(f"Failed to load preprocessor stats from {fname}: {exc}")
                continue
            if "observation.state.mean" in preprocessor_stats:
                self._state_mean = preprocessor_stats["observation.state.mean"].to(
                    device
                )
                self._state_std = (
                    preprocessor_stats["observation.state.std"]
                    .to(device)
                    .clamp(min=1e-8)
                )
                return
        logger.debug("No state stats found, using identity")
        self._state_mean = torch.zeros(6, device=device)
        self._state_std = torch.ones(6, device=device)

    def _build_camera_key_map(self, config: dict[str, Any]) -> None:
        """根据 policy config 的输入特征覆盖默认相机 key 映射。

        例如 policy 声明 ``observation.images.wrist`` 而不是默认
        ``observation.images.handeye`` 时，所有 wrist 类相机名（handeye、
        right_wrist、left_wrist、wrist）都会重映射到该 key。
        """
        input_features = config.get("input_features")
        if not isinstance(input_features, dict):
            return
        policy_image_keys = {
            k for k in input_features if k.startswith("observation.images.")
        }
        for policy_key in policy_image_keys:
            suffix = policy_key.rsplit(".", 1)[-1]
            if suffix in _WRIST_CAMERA_NAMES:
                for cam_name in _WRIST_CAMERA_NAMES:
                    if cam_name in _CAMERA_KEY_MAP:
                        _CAMERA_KEY_MAP[cam_name] = policy_key
                _CAMERA_KEY_MAP[suffix] = policy_key
            elif suffix and suffix not in _CAMERA_KEY_MAP:
                _CAMERA_KEY_MAP[suffix] = policy_key

    def _direct_policy_inference(
        self,
        model_path: str,
        *,
        observation: dict[str, Any],
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            self._load_policy(model_path)
        except ImportError as exc:
            return {
                "success": False,
                "status": "failed",
                "failure_mode": "missing_dependency",
                "summary": f"Policy dependencies unavailable: {exc}",
                "error": str(exc),
            }
        except Exception as exc:
            return {
                "success": False,
                "status": "failed",
                "failure_mode": "model_load_failed",
                "summary": f"Failed to load policy from {model_path}: {exc}",
                "error": str(exc),
            }

        try:
            batch = self._preprocess_observation(observation, payload or {})
        except Exception as exc:
            return {
                "success": False,
                "status": "failed",
                "failure_mode": "preprocessing_failed",
                "summary": f"Preprocessing failed: {exc}",
                "error": str(exc),
            }

        try:
            raw = self._run_policy_inference(batch)
        except Exception as exc:
            return {
                "success": False,
                "status": "failed",
                "failure_mode": "inference_failed",
                "summary": f"Policy inference failed: {type(exc).__name__}: {exc}",
                "error": str(exc),
                "metrics": {
                    "vla": {
                        "backend_mode": "action_chunk_policy",
                        "policy_type": self._policy_type,
                        "frame_id": observation.get("frame_id"),
                        "hardware_ownership": "none",
                    }
                },
            }
        actions_out = raw["actions"]
        return {
            "success": True,
            "status": "completed",
            "summary": f"{self._policy_type} inference completed",
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
                    "policy_type": self._policy_type,
                    "frame_id": observation.get("frame_id"),
                    "hardware_ownership": "none",
                },
            },
        }

    def _preprocess_observation(
        self, observation: dict[str, Any], payload: dict[str, Any]
    ) -> dict[str, Any]:
        """把 observation 字典转换为 LeRobot policy batch，不做手动归一化。

        policy 内部的 ``normalize_inputs`` 会处理 mean/std 缩放，因此这里只把图像转换为
        tensor（BCHW，[0,1]），并把 state 转为角度 tensor。
        """
        device = str(self.spec.settings.get("policy_device") or "cuda")
        w, h = self._image_size
        batch: dict[str, Any] = {}

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
                    img = img.resize((w, h), Resampling.BILINEAR)
                    arr = np.array(img, dtype=np.float32) / 255.0
                    tensor = (
                        torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
                    )
                    batch[target_key] = tensor
                except Exception as exc:
                    logger = __import__("logging").getLogger(__name__)
                    logger.debug(f"Failed to decode image from camera {camera}: {exc}")
                    continue

        state = observation.get("proprioception")
        if isinstance(state, list) and len(state) >= 6:
            state_deg = [
                float(v) * _RAD_TO_DEG
                for v in np.asarray(state[:6], dtype=np.float32).flat
            ]
            batch["observation.state"] = torch.tensor(
                state_deg, dtype=torch.float32, device=device
            ).unsqueeze(0)
        elif isinstance(state, (int, float)):
            batch["observation.state"] = torch.tensor(
                [[float(state)]], dtype=torch.float32, device=device
            )

        # 语言条件 policy（smolvla、pi0、pi0.5）需要 task prompt。
        task = (
            payload.get("task")
            or payload.get("task_prompt")
            or payload.get("objective")
            or "manipulate"
        )
        batch["task"] = [str(task)]

        # 为直接接收语言 token 的 policy 对 task 做 tokenize。
        self._tokenize_task(batch, device)

        return batch

    def _tokenize_task(self, batch: dict[str, Any], device: str) -> None:
        """为语言条件 policy 对任务 prompt 做 tokenize。

        支持两条路径：
        - smolvla：内部 vlm_with_expert.processor.tokenizer
        - pi05：PaliGemma tokenizer，并把离散化 state 嵌入 prompt
        """
        if self._policy is None:
            return

        if self._policy_type == "pi05":
            self._tokenize_pi05_task(batch, device)
            return

        processor = getattr(
            getattr(getattr(self._policy, "model", None), "vlm_with_expert", None),
            "processor",
            None,
        )
        if processor is None or not hasattr(processor, "tokenizer"):
            return
        tokenizer = processor.tokenizer
        task_list: list[str] = batch.get("task", [])
        if not task_list:
            return
        tokens = tokenizer(
            task_list,
            return_tensors="pt",
            padding="max_length",
            max_length=48,
            truncation=True,
        )
        batch["observation.language.tokens"] = tokens["input_ids"].to(device)
        batch["observation.language.attention_mask"] = (
            tokens["attention_mask"].bool().to(device)
        )

    def _tokenize_pi05_task(self, batch: dict[str, Any], device: str) -> None:
        """Pi05 专用 tokenize：离散化 state，并嵌入文本 prompt。

        Pi05 把机器人 state 放进语言 prompt，而不是单独使用 ``observation.state`` tensor。
        流程为：
        1. 用数据集统计量把 state 归一化到 [-1, 1]
        2. 离散化到 256 个 bin（0-255）
        3. 构造 prompt："Task: {task}, State: {bins};\\nAction: "
        4. 用 PaliGemma tokenizer 处理（max_length=200）
        """
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer_path = Path("models/paligemma-tokenizer")
            if tokenizer_path.exists():
                self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            else:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    "google/paligemma-3b-pt-224"
                )

        task_list: list[str] = batch.get("task", [])
        state = batch.get("observation.state")
        if state is not None:
            batch.pop("observation.state", None)

        bsz = state.shape[0] if state is not None else 1
        prompts: list[str] = []
        for i in range(bsz):
            task = (task_list[i] if i < len(task_list) else "manipulate").strip()
            cleaned = task.replace("_", " ").replace("\n", " ")
            state_str = ""
            if state is not None:
                state_row = state[i].to(device, dtype=torch.float32)
                # 归一化到 [-1, 1]
                if self._state_mean is not None and self._state_std is not None:
                    s_mean = self._state_mean.to(device)
                    s_std = self._state_std.to(device).clamp(min=1e-8)
                    state_row = (state_row - s_mean) / s_std
                state_row = state_row.clamp(-1, 1)
                # 离散化到 256 个 bin
                bins = np.linspace(-1, 1, 257)[:-1]
                discretised = np.digitize(state_row.cpu().numpy(), bins=bins) - 1
                # 用 0 填充到 max_state_dim（32）
                max_dim = 32
                padded = np.zeros(max_dim, dtype=np.int64)
                padded[: len(discretised)] = discretised[:max_dim]
                state_str = " ".join(map(str, padded))
            prompts.append(f"Task: {cleaned}, State: {state_str};\nAction: ")

        tokens = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            max_length=200,
            truncation=True,
        )
        batch["observation.language.tokens"] = tokens["input_ids"].to(device)
        batch["observation.language.attention_mask"] = (
            tokens["attention_mask"].bool().to(device)
        )

    def _run_policy_inference(self, batch: dict[str, Any]) -> dict[str, Any]:
        """运行通用 LeRobot policy，并把输出转换为关节/夹爪字典。"""
        assert self._policy is not None, "Policy not loaded"
        assert self._action_mean is not None, "Action stats not loaded"
        assert self._action_std is not None, "Action stats not loaded"

        with torch.no_grad():
            action_chunk = self._policy.predict_action_chunk(batch)
            if action_chunk.ndim == 3:
                action_chunk = action_chunk.squeeze(0)

        # 所有 LeRobot policy 都输出归一化动作；这里执行反归一化。
        num_actions = action_chunk.shape[-1]
        action_mean = self._action_mean[:num_actions]
        action_std = self._action_std[:num_actions].clamp(min=1e-8)
        action_chunk = action_chunk * action_std + action_mean

        actions_raw = action_chunk.cpu().tolist()

        actions_out: list[dict[str, Any]] = []
        for action_vec in actions_raw:
            joints: dict[str, float] = {}
            for i, name in enumerate(_SO101_JOINT_NAMES):
                if i < len(action_vec):
                    joints[name] = round(float(action_vec[i]) * _DEG_TO_RAD, 6)
            gripper_idx = min(5, len(action_vec) - 1)
            gripper_pct = max(0.0, min(1.0, float(action_vec[gripper_idx]) / 100.0))
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
