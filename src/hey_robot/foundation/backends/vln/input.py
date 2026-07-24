from __future__ import annotations

import base64
import io
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from hey_robot.foundation.backends.vln.models import (
    VLNPlannerInput,
    VLNPlanningError,
)


def planner_input_from_payload(
    payload: dict[str, Any],
    *,
    camera: str,
    media_root: str | None,
    hfov: float,
) -> VLNPlannerInput:
    arguments = dict(payload.get("arguments", {}) or {})
    selected_camera = str(arguments.get("camera") or camera)
    instruction = _instruction_from_payload(payload, arguments)
    if not instruction:
        raise VLNPlanningError(
            "invalid_request",
            "VLN planning requires arguments.instruction, arguments.target, or objective",
        )
    rgb, image_source = _load_rgb_from_payload(
        payload,
        arguments,
        camera=selected_camera,
        media_root=media_root,
    )
    return VLNPlannerInput(
        rgb=rgb,
        depth=_depth_from_payload(arguments, rgb.shape[:2]),
        pose=_pose_from_payload(arguments, payload),
        instruction=instruction,
        intrinsic=_intrinsic_from_payload(
            arguments,
            width=int(rgb.shape[1]),
            height=int(rgb.shape[0]),
            hfov=hfov,
        ),
        look_down=bool(arguments.get("look_down", False)),
        image_source=image_source,
    )


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
    media_root: str | None,
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


def _load_rgb_source(source: Any, *, media_root: str | None) -> np.ndarray:
    if isinstance(source, np.ndarray):
        return source
    if isinstance(source, list | tuple):
        return np.asarray(source)
    if isinstance(source, dict):
        if _dict_contains_base64_image(source):
            try:
                decoded = base64.b64decode(str(source["data"]), validate=True)
                with Image.open(io.BytesIO(decoded)) as image:
                    return np.asarray(image.convert("RGB"))
            except Exception as exc:
                raise VLNPlanningError(
                    "image_unavailable",
                    f"invalid base64 VLN image payload: {exc}",
                ) from exc
        for key in ("rgb", "rgb_array", "data"):
            if source.get(key) is not None:
                return _load_rgb_source(source[key], media_root=media_root)
        for key in ("path", "image_path", "uri", "image_ref", "image_uri"):
            if source.get(key) is not None:
                return _load_rgb_source(source[key], media_root=media_root)
    if isinstance(source, str):
        path = _path_from_image_reference(source, media_root=media_root)
        try:
            with Image.open(path) as image:
                return np.asarray(image.convert("RGB"))
        except OSError as exc:
            raise VLNPlanningError(
                "image_unavailable", f"cannot read VLN image {source}: {exc}"
            ) from exc
    raise VLNPlanningError(
        "image_unavailable",
        f"unsupported VLN image source: {type(source).__name__}",
    )


def _dict_contains_base64_image(source: dict[str, Any]) -> bool:
    if source.get("data") is None:
        return False
    fmt = str(source.get("format") or "").lower()
    content_type = str(
        source.get("content_type") or source.get("mime_type") or ""
    ).lower()
    encoding = str(source.get("encoding") or "base64").lower()
    if encoding not in {"", "base64"}:
        return False
    return fmt in {
        "jpeg",
        "jpg",
        "png",
        "image/jpeg",
        "image/png",
    } or content_type.startswith("image/")


def _path_from_image_reference(value: str, *, media_root: str | None) -> Path:
    prefix = "media://local/"
    if not value.startswith(prefix):
        path = Path(value).expanduser()
        if not path.is_file():
            raise VLNPlanningError(
                "image_unavailable", f"image file not found: {value}"
            )
        return path
    root_value = str(media_root or "").strip()
    if not root_value:
        raise VLNPlanningError(
            "image_unavailable",
            "media://local image_ref requires model setting media_root",
        )
    relative = PurePosixPath(value.removeprefix(prefix))
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise VLNPlanningError("image_unavailable", "unsafe local media URI")
    root = Path(root_value).expanduser().resolve()
    path = (root / Path(*relative.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise VLNPlanningError(
            "image_unavailable", "local media URI escapes media_root"
        ) from exc
    if not path.is_file():
        raise VLNPlanningError("image_unavailable", f"media object not found: {value}")
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
    if isinstance(source, list | tuple):
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
        try:
            with Image.open(str(arguments["depth_path"])) as image:
                return np.asarray(image, dtype=np.float32)
        except OSError as exc:
            raise VLNPlanningError(
                "image_unavailable", f"cannot read VLN depth image: {exc}"
            ) from exc
    height, width = image_shape
    return np.zeros((height, width), dtype=np.float32)


def _pose_from_payload(
    arguments: dict[str, Any], payload: dict[str, Any]
) -> tuple[float, float, float]:
    value = arguments.get("pose") or dict(payload.get("metadata", {}) or {}).get("pose")
    if value is None:
        return (0.0, 0.0, 0.0)
    items = to_float_list(value)
    if len(items) < 3:
        raise VLNPlanningError("invalid_request", "pose must contain x, y, yaw")
    return (items[0], items[1], items[2])


def _intrinsic_from_payload(
    arguments: dict[str, Any], *, width: int, height: int, hfov: float
) -> np.ndarray:
    if arguments.get("intrinsic") is not None:
        intrinsic = np.asarray(arguments["intrinsic"], dtype=np.float32)
        if intrinsic.shape not in {(3, 3), (4, 4)}:
            raise VLNPlanningError(
                "invalid_request", "intrinsic must be a 3x3 or 4x4 matrix"
            )
        return intrinsic
    fx = (width / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
    return np.asarray(
        [
            [fx, 0.0, (width - 1.0) / 2.0, 0.0],
            [0.0, fx, (height - 1.0) / 2.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def to_float_list(value: Any) -> list[float]:
    if isinstance(value, str):
        return []
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return []
