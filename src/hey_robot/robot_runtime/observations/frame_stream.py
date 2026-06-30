from __future__ import annotations

import io
import json
import struct
from typing import Any

import numpy as np
from PIL import Image


def encode_frame_packet(image: np.ndarray, metadata: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(_as_rgb_uint8(image)).save(
        buffer, format="JPEG", quality=80, optimize=False
    )
    header = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    return struct.pack("!I", len(header)) + header + buffer.getvalue()


def decode_frame_packet(payload: bytes) -> tuple[dict[str, Any], np.ndarray]:
    if len(payload) < 4:
        raise ValueError("camera frame packet is truncated")
    header_size = struct.unpack("!I", payload[:4])[0]
    if header_size <= 0 or len(payload) < 4 + header_size:
        raise ValueError("camera frame packet header is invalid")
    metadata = json.loads(payload[4 : 4 + header_size].decode("utf-8"))
    image = np.asarray(
        Image.open(io.BytesIO(payload[4 + header_size :])).convert("RGB")
    )
    return dict(metadata), image


def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    frame = np.asarray(image)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("camera frame must be an RGB image")
    return frame[:, :, :3]
