from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, cast

import numpy as np


@dataclass(frozen=True)
class OpenCVCameraConfig:
    enabled: bool = True
    device_id: int = 0
    width: int | None = None
    height: int | None = None
    fps: int | None = None
    backend: str = "auto"


class OpenCVCamera:
    """OpenCV camera component that can be attached to any local robot body."""

    def __init__(self, config: OpenCVCameraConfig) -> None:
        self.config = config
        self._capture: Any | None = None
        self._frame_id = 0
        self._last_error: str | None = None
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_at: float | None = None
        self._condition = threading.Condition()
        self._reader_thread: threading.Thread | None = None
        self._stopping = False

    @property
    def frame_id(self) -> int:
        with self._condition:
            return self._frame_id

    def open(self) -> dict:
        if not self.config.enabled:
            return {"success": True, "message": "camera disabled", "enabled": False}
        try:
            import cv2
        except ImportError as exc:
            self._last_error = "opencv-python is required for native camera capture"
            return {
                "success": False,
                "message": self._last_error,
                "error": f"{type(exc).__name__}: {exc}",
            }
        backend = _opencv_backend(cv2, self.config.backend)
        capture = (
            cv2.VideoCapture(self.config.device_id, backend)
            if backend is not None
            else cv2.VideoCapture(self.config.device_id)
        )
        if self.config.width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        if self.config.height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if self.config.fps is not None:
            capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            self._last_error = f"failed to open camera device {self.config.device_id}"
            capture.release()
            return {"success": False, "message": self._last_error}
        with self._condition:
            self._capture = capture
            self._latest_frame = None
            self._latest_frame_at = None
            self._stopping = False
            self._last_error = None
        self._reader_thread = threading.Thread(
            target=self._read_loop,
            name=f"opencv-camera-{self.config.device_id}",
            daemon=True,
        )
        self._reader_thread.start()
        return {
            "success": True,
            "message": "camera opened",
            "device_id": self.config.device_id,
            "backend": self.config.backend,
            "width": self.config.width,
            "height": self.config.height,
            "fps": self.config.fps,
        }

    def capture_frame(
        self, *, timeout_ms: int = 1000
    ) -> tuple[int | None, np.ndarray | None]:
        if not self.config.enabled:
            return None, None
        timeout_sec = max(timeout_ms, 1) / 1000.0
        with self._condition:
            if self._capture is None:
                return None, None
            ready = self._condition.wait_for(
                lambda: self._latest_frame is not None or self._stopping,
                timeout=timeout_sec,
            )
            if not ready or self._latest_frame is None:
                self._last_error = f"no camera frame within {timeout_ms} ms"
                return None, None
            return self._frame_id, self._latest_frame.copy()

    def diagnostics(self) -> dict:
        with self._condition:
            frame_age_ms = (
                max(0.0, (time.monotonic() - self._latest_frame_at) * 1000.0)
                if self._latest_frame_at is not None
                else None
            )
            return {
                "success": (self._capture is not None and self.config.enabled)
                or not self.config.enabled,
                "enabled": self.config.enabled,
                "opened": self._capture is not None,
                "device_id": self.config.device_id,
                "backend": self.config.backend,
                "frame_id": self._frame_id,
                "frame_age_ms": frame_age_ms,
                "error": self._last_error,
            }

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            capture = self._capture
            self._condition.notify_all()
        thread = self._reader_thread
        if thread is not None:
            thread.join(timeout=1.0)
        if capture is not None:
            capture.release()
        with self._condition:
            self._capture = None
            self._reader_thread = None
            self._latest_frame = None
            self._latest_frame_at = None

    def _read_loop(self) -> None:
        while True:
            with self._condition:
                if self._stopping or self._capture is None:
                    return
                capture = self._capture
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                with self._condition:
                    if self._stopping:
                        return
                    self._last_error = "camera frame read failed"
                time.sleep(0.01)
                continue
            frame_rgb = _bgr_to_rgb(frame_bgr)
            with self._condition:
                if self._stopping:
                    return
                self._frame_id += 1
                self._latest_frame = frame_rgb
                self._latest_frame_at = time.monotonic()
                self._last_error = None
                self._condition.notify_all()


def _bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[2] >= 3:
        return frame[:, :, :3][:, :, ::-1].copy()
    return frame


def _opencv_backend(cv2: Any, backend: str) -> int | None:
    normalized = backend.lower().strip()
    if normalized in {"", "auto", "default"}:
        return cast("int | None", cv2.CAP_DSHOW if _is_windows() else None)
    if normalized == "dshow":
        return cast(int, cv2.CAP_DSHOW)
    if normalized == "msmf":
        return cast(int, cv2.CAP_MSMF)
    if normalized == "v4l2":
        return cast(int, cv2.CAP_V4L2)
    return None


def _is_windows() -> bool:
    import sys

    return sys.platform.startswith("win")
