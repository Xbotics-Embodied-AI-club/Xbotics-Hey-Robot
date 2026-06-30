from __future__ import annotations

import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from hey_robot.config import CapabilityServiceSpec
from hey_robot.robot_runtime.media import LocalMediaStore


class VLNPlanningError(RuntimeError):
    def __init__(self, failure_mode: str, message: str) -> None:
        super().__init__(message)
        self.failure_mode = failure_mode


@dataclass(frozen=True)
class VLNPlannerInput:
    rgb: np.ndarray
    depth: np.ndarray | None
    pose: tuple[float, float, float]
    instruction: str
    intrinsic: np.ndarray
    look_down: bool = False
    image_source: str | None = None


@dataclass(frozen=True)
class VLNPlannerResult:
    mode: str
    pixel_goal: list[int] | None = None
    waypoint: list[float] | None = None
    heading_deg: float | None = None
    stop: bool = False
    confidence: float | None = None
    reason: str | None = None
    raw_output: str | None = None
    image_source: str | None = None

    def to_metrics(
        self, *, backend: str, camera: str, control_mode: str
    ) -> dict[str, Any]:
        return {
            "backend": backend,
            "control_mode": control_mode,
            "camera": camera,
            "mode": self.mode,
            "pixel_goal": self.pixel_goal,
            "waypoint": self.waypoint,
            "heading_deg": self.heading_deg,
            "stop": self.stop,
            "confidence": self.confidence,
            "reason": self.reason,
            "raw_output": self.raw_output,
            "image_source": self.image_source,
        }


class InternVLAN1System2Executor:
    """Planner-only VLN executor for InternVLA-N1 System 2.

    The first implementation intentionally supports a mock planner path so the
    Hey Robot capability/skill plumbing can be validated before loading the
    heavy InternNav dependency stack.
    """

    def __init__(self, service_id: str, spec: CapabilityServiceSpec) -> None:
        self.service_id = service_id
        self.spec = spec
        self._cancelled = threading.Event()
        self._model: Any | None = None
        self._model_error: str | None = None

    def health(self) -> dict[str, Any]:
        settings = self.spec.settings
        mock_mode = self._mock_mode()
        missing = self._missing_config()
        loaded = mock_mode or (not missing and self._model_error is None)
        return {
            "name": self.service_id,
            "online": True,
            "loaded": loaded,
            "robot_id": self.spec.robot_id,
            "error": None
            if loaded
            else self._model_error
            or f"missing VLN configuration: {', '.join(missing)}",
            "metrics": {
                "type": self.spec.type,
                "backend": settings.get("backend", "internvla_n1_system2"),
                "model_path": settings.get("model_path"),
                "runtime": "internvla_n1_system2",
                "mock_mode": mock_mode,
                "control_mode": settings.get("control_mode", "planner_only"),
                "camera": settings.get("camera", "front"),
            },
        }

    def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._cancelled.clear()
        started_at = time.time()
        settings = self.spec.settings
        camera = str(settings.get("camera") or "front")
        control_mode = str(settings.get("control_mode") or "planner_only")
        if control_mode != "planner_only":
            return {
                "success": False,
                "status": "failed",
                "failure_mode": "unsupported_control_mode",
                "summary": f"VLN control_mode={control_mode!r} is not enabled",
                "metrics": {"vln": self._base_metrics(camera, control_mode)},
            }

        try:
            result = (
                self._mock_plan(payload)
                if self._mock_mode()
                else self._internvla_plan(payload)
            )
        except VLNPlanningError as exc:
            return {
                "success": False,
                "status": "failed",
                "failure_mode": exc.failure_mode,
                "summary": str(exc),
                "error": str(exc),
                "metrics": {"vln": self._base_metrics(camera, control_mode)},
            }
        except ImportError as exc:
            self._model_error = str(exc)
            return {
                "success": False,
                "status": "failed",
                "failure_mode": "missing_dependency",
                "summary": f"InternVLA-N1 System 2 dependencies are unavailable: {exc}",
                "error": str(exc),
                "metrics": {"vln": self._base_metrics(camera, control_mode)},
            }
        if self._cancelled.is_set():
            return {
                "success": False,
                "status": "cancelled",
                "failure_mode": "cancelled",
                "summary": "VLN planning cancelled",
                "metrics": {
                    "duration_sec": round(time.time() - started_at, 3),
                    "vln": result.to_metrics(
                        backend="internvla_n1_system2",
                        camera=camera,
                        control_mode=control_mode,
                    ),
                },
            }
        return {
            "success": True,
            "status": "completed",
            "summary": f"VLN planner produced {result.mode}",
            "metrics": {
                "duration_sec": round(time.time() - started_at, 3),
                "vln": result.to_metrics(
                    backend="internvla_n1_system2",
                    camera=camera,
                    control_mode=control_mode,
                ),
            },
        }

    def cancel(self) -> None:
        self._cancelled.set()

    def _mock_mode(self) -> bool:
        settings = self.spec.settings
        if "mock_mode" in settings:
            return bool(settings.get("mock_mode"))
        if "use_mock" in settings:
            return bool(settings.get("use_mock"))
        return not bool(settings.get("model_path"))

    def _missing_config(self) -> list[str]:
        if self._mock_mode():
            return []
        return [key for key in ("model_path",) if not self.spec.settings.get(key)]

    def _base_metrics(self, camera: str, control_mode: str) -> dict[str, Any]:
        return {
            "backend": self.spec.settings.get("backend", "internvla_n1_system2"),
            "control_mode": control_mode,
            "camera": camera,
        }

    def _mock_plan(self, payload: dict[str, Any]) -> VLNPlannerResult:
        arguments = dict(payload.get("arguments", {}) or {})
        text = " ".join(
            str(value)
            for value in (
                payload.get("objective"),
                arguments.get("target"),
                arguments.get("instruction"),
            )
            if value
        ).lower()
        if any(token in text for token in ("stop", "停止", "停下", "done", "完成")):
            return VLNPlannerResult(
                mode="stop",
                stop=True,
                confidence=1.0,
                reason="mock planner matched stop-like instruction",
                raw_output="STOP",
            )
        heading = _number_arg(arguments, "heading_deg")
        if heading is None:
            heading = _number_arg(self.spec.settings, "mock_heading_deg")
        if heading is not None:
            return VLNPlannerResult(
                mode="heading",
                heading_deg=float(heading),
                confidence=0.5,
                reason="mock planner returned configured heading",
                raw_output=f"HEADING {float(heading):.1f}",
            )
        width = int(self.spec.settings.get("image_width", 640))
        height = int(self.spec.settings.get("image_height", 480))
        x = int(self.spec.settings.get("mock_pixel_x", width // 2))
        y = int(self.spec.settings.get("mock_pixel_y", height // 2))
        return VLNPlannerResult(
            mode="pixel_goal",
            pixel_goal=[x, y],
            confidence=0.5,
            reason="mock planner returned center pixel goal",
            raw_output=f"({y}, {x})",
        )

    def _internvla_plan(self, payload: dict[str, Any]) -> VLNPlannerResult:
        planner_input = self._build_planner_input(payload)
        model = self._load_model()
        output = model.s2_step(
            planner_input.rgb,
            planner_input.depth,
            planner_input.pose,
            planner_input.instruction,
            planner_input.intrinsic,
            planner_input.look_down,
        )
        return self._planner_result_from_s2_output(
            output,
            image_width=int(planner_input.rgb.shape[1]),
            image_height=int(planner_input.rgb.shape[0]),
            image_source=planner_input.image_source,
        )

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        settings = self.spec.settings
        repo = settings.get("internnav_repo")
        if repo:
            repo_path = str(Path(str(repo)).expanduser().resolve())
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)
        from internnav.model import get_config, get_policy

        policy_name = str(settings.get("policy_name") or "InternVLAN1_Policy")
        policy_cls = get_policy(policy_name)
        config_cls = get_config(policy_name)
        model_settings = {
            "policy_name": policy_name,
            "state_encoder": None,
            "mode": "system2",
            "model_path": settings.get("model_path"),
            "device": settings.get("device", "cuda"),
            "dtype": settings.get("dtype", settings.get("torch_dtype", "auto")),
            "torch_dtype": settings.get("torch_dtype", settings.get("dtype", "auto")),
            "attn_implementation": settings.get("attn_implementation", "auto"),
            "num_history": int(settings.get("num_history", 8)),
            "resize_w": int(settings.get("resize_w", settings.get("image_width", 384))),
            "resize_h": int(
                settings.get("resize_h", settings.get("image_height", 384))
            ),
            "max_new_tokens": int(settings.get("max_new_tokens", 128)),
            "num_frames": int(settings.get("num_frames", 8)),
            "num_future_steps": int(settings.get("num_future_steps", 0)),
            "continuous_traj": bool(settings.get("continuous_traj", False)),
            "n_query": int(settings.get("n_query", 4)),
            "vis_debug": bool(settings.get("vis_debug", False)),
            "vis_debug_path": settings.get("vis_debug_path", "./logs/vln_debug"),
        }
        model = policy_cls(config=config_cls(model_cfg={"model": model_settings}))
        n_query = int(settings.get("n_query", 4))
        if not hasattr(model.model.config, "n_query"):
            model.model.config.n_query = n_query
        if not hasattr(model.model, "latent_queries") or model.model.latent_queries is None:
            model.model.latent_queries = torch.nn.Parameter(
                torch.randn(1, n_query, model.model.config.hidden_size)
            )
        eval_fn = getattr(model, "eval", None)
        if callable(eval_fn):
            eval_fn()
        self._model = model
        return model

    def _build_planner_input(self, payload: dict[str, Any]) -> VLNPlannerInput:
        arguments = dict(payload.get("arguments", {}) or {})
        camera = str(
            arguments.get("camera") or self.spec.settings.get("camera") or "front"
        )
        instruction = _instruction_from_payload(payload, arguments)
        if not instruction:
            raise VLNPlanningError(
                "invalid_request",
                "VLN planning requires arguments.instruction, arguments.target, or objective",
            )

        rgb, image_source = _load_rgb_from_payload(
            payload,
            arguments,
            camera=camera,
            media_root=self.spec.settings.get("media_root"),
        )
        depth = _depth_from_payload(arguments, rgb.shape[:2])
        pose = _pose_from_payload(arguments, payload)
        intrinsic = _intrinsic_from_payload(
            arguments,
            width=int(rgb.shape[1]),
            height=int(rgb.shape[0]),
            hfov=float(self.spec.settings.get("hfov", 90.0)),
        )
        return VLNPlannerInput(
            rgb=rgb,
            depth=depth,
            pose=pose,
            instruction=instruction,
            intrinsic=intrinsic,
            look_down=bool(arguments.get("look_down", False)),
            image_source=image_source,
        )

    def _planner_result_from_s2_output(
        self,
        output: Any,
        *,
        image_width: int,
        image_height: int,
        image_source: str | None = None,
    ) -> VLNPlannerResult:
        raw_output = _public_raw_output(output)
        pixel = getattr(output, "output_pixel", None)
        if pixel is not None:
            parsed = _parse_pixel_goal(pixel)
            if parsed is None:
                raise VLNPlanningError(
                    "vln_parse_failed",
                    "InternVLA-N1 System 2 returned an invalid pixel goal",
                )
            x, y = parsed
            clamped_x = min(max(x, 0), max(image_width - 1, 0))
            clamped_y = min(max(y, 0), max(image_height - 1, 0))
            reason = "InternVLA-N1 System 2 returned output_pixel"
            if (clamped_x, clamped_y) != (x, y):
                reason = (
                    "InternVLA-N1 System 2 output_pixel was clamped to image bounds"
                )
            return VLNPlannerResult(
                mode="pixel_goal",
                pixel_goal=[clamped_x, clamped_y],
                reason=reason,
                raw_output=raw_output,
                image_source=image_source,
            )

        action = getattr(output, "output_action", None)
        if _is_stop_action(action):
            return VLNPlannerResult(
                mode="stop",
                stop=True,
                reason="InternVLA-N1 System 2 returned STOP",
                raw_output=raw_output or str(action),
                image_source=image_source,
            )
        raise VLNPlanningError(
            "vln_no_valid_goal",
            "InternVLA-N1 System 2 did not return output_pixel or STOP",
        )


def _number_arg(source: dict[str, Any], key: str) -> float | None:
    value = source.get(key)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def _instruction_from_payload(
    payload: dict[str, Any], arguments: dict[str, Any]
) -> str:
    value = (
        arguments.get("instruction")
        or arguments.get("target")
        or arguments.get("task")
        or payload.get("objective")
    )
    return str(value or "").strip()


def _load_rgb_from_payload(
    payload: dict[str, Any],
    arguments: dict[str, Any],
    *,
    camera: str,
    media_root: Any,
) -> tuple[np.ndarray, str | None]:
    source = _image_source_from_payload(payload, arguments, camera=camera)
    if source is None:
        raise VLNPlanningError(
            "image_unavailable",
            "VLN planning requires image_path, image_ref, image_uri, rgb, or observation.images",
        )
    rgb = _load_rgb_source(source, media_root=media_root)
    return _as_uint8_rgb(rgb), _image_source_label(source)


def _image_source_from_payload(
    payload: dict[str, Any], arguments: dict[str, Any], *, camera: str
) -> Any | None:
    for key in ("rgb", "rgb_array", "image", "image_path", "image_ref", "image_uri"):
        if arguments.get(key) is not None:
            return arguments[key]
    for container in (
        arguments.get("observation"),
        payload.get("observation"),
        dict(payload.get("metadata", {}) or {}).get("observation"),
    ):
        if not isinstance(container, dict):
            continue
        images = container.get("images")
        if not isinstance(images, list):
            continue
        selected = _select_image_ref(images, camera=camera)
        if selected is not None:
            return selected
    return None


def _select_image_ref(images: list[Any], *, camera: str) -> Any | None:
    first: Any | None = None
    for image in images:
        if first is None:
            first = image
        if isinstance(image, dict) and str(image.get("camera") or "") == camera:
            return image
    return first


def _load_rgb_source(source: Any, *, media_root: Any) -> np.ndarray:
    if isinstance(source, np.ndarray):
        return source
    if isinstance(source, (list, tuple)):
        return np.asarray(source)
    if isinstance(source, dict):
        for key in ("rgb", "rgb_array", "data"):
            if source.get(key) is not None:
                return _load_rgb_source(source[key], media_root=media_root)
        for key in ("path", "image_path", "uri", "image_ref", "image_uri"):
            if source.get(key) is not None:
                return _load_rgb_source(source[key], media_root=media_root)
    if isinstance(source, str):
        path = _path_from_image_reference(source, media_root=media_root)
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))
    raise VLNPlanningError(
        "image_unavailable",
        f"unsupported VLN image source: {type(source).__name__}",
    )


def _path_from_image_reference(value: str, *, media_root: Any) -> Path:
    if value.startswith("media://local/"):
        if not media_root:
            raise VLNPlanningError(
                "image_unavailable",
                "media://local image_ref requires capability setting media_root",
            )
        return LocalMediaStore(str(media_root)).path_for_uri(value)
    path = Path(value).expanduser()
    if not path.is_file():
        raise VLNPlanningError("image_unavailable", f"image file not found: {value}")
    return path


def _image_source_label(source: Any) -> str | None:
    if isinstance(source, str):
        return source
    if isinstance(source, dict):
        value = (
            source.get("uri")
            or source.get("path")
            or source.get("image_path")
            or source.get("image_ref")
            or source.get("image_uri")
        )
        if value:
            return str(value)
    if isinstance(source, np.ndarray):
        return "payload.rgb"
    if isinstance(source, (list, tuple)):
        return "payload.rgb_array"
    return None


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[2] not in {1, 3, 4}:
        raise VLNPlanningError(
            "image_unavailable",
            f"VLN image must be HxW, HxWx1, HxWx3, or HxWx4; got shape={arr.shape}",
        )
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _depth_from_payload(
    arguments: dict[str, Any], image_shape: tuple[int, int]
) -> np.ndarray | None:
    if arguments.get("depth") is not None:
        return np.asarray(arguments["depth"], dtype=np.float32)
    if arguments.get("depth_path") is not None:
        with Image.open(str(arguments["depth_path"])) as image:
            return np.asarray(image, dtype=np.float32)
    height, width = image_shape
    return np.zeros((height, width), dtype=np.float32)


def _pose_from_payload(
    arguments: dict[str, Any], payload: dict[str, Any]
) -> tuple[float, float, float]:
    value = arguments.get("pose") or dict(payload.get("metadata", {}) or {}).get("pose")
    if value is None:
        return (0.0, 0.0, 0.0)
    items = list(value) if isinstance(value, (list, tuple)) else []
    if len(items) < 3:
        raise VLNPlanningError("invalid_request", "pose must contain x, y, yaw")
    return (float(items[0]), float(items[1]), float(items[2]))


def _intrinsic_from_payload(
    arguments: dict[str, Any], *, width: int, height: int, hfov: float
) -> np.ndarray:
    if arguments.get("intrinsic") is not None:
        intrinsic = np.asarray(arguments["intrinsic"], dtype=np.float32)
        if intrinsic.shape not in {(3, 3), (4, 4)}:
            raise VLNPlanningError(
                "invalid_request",
                "intrinsic must be a 3x3 or 4x4 matrix",
            )
        return intrinsic
    fx = (width / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
    fy = fx
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    return np.asarray(
        [
            [fx, 0.0, cx, 0.0],
            [0.0, fy, cy, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _parse_pixel_goal(value: Any) -> tuple[int, int] | None:
    arr = np.asarray(value).reshape(-1)
    if arr.size < 2:
        return None
    return (int(arr[0]), int(arr[1]))


def _is_stop_action(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().upper() == "STOP"
    arr = np.asarray(value).reshape(-1)
    for item in arr:
        if str(item).strip().upper() == "STOP":
            return True
        try:
            if int(item) == 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _public_raw_output(output: Any) -> str | None:
    text = getattr(output, "llm_output", None) or getattr(output, "raw_output", None)
    if text:
        return str(text)
    pixel = getattr(output, "output_pixel", None)
    if pixel is not None:
        return str(np.asarray(pixel).reshape(-1).tolist())
    action = getattr(output, "output_action", None)
    if action is not None:
        return str(np.asarray(action).reshape(-1).tolist())
    return None
