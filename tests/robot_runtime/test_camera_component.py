from __future__ import annotations

import builtins
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from hey_robot.robot_runtime.components.camera import (
    OpenCVCamera,
    OpenCVCameraConfig,
    _bgr_to_rgb,
    _opencv_backend,
)


def test_opencv_camera_disabled_is_safe_noop() -> None:
    camera = OpenCVCamera(OpenCVCameraConfig(enabled=False, device_id=3))

    assert camera.frame_id == 0
    assert camera.open() == {
        "success": True,
        "message": "camera disabled",
        "enabled": False,
    }
    assert camera.capture_frame() == (None, None)
    assert camera.diagnostics() == {
        "success": True,
        "enabled": False,
        "opened": False,
        "device_id": 3,
        "backend": "auto",
        "frame_id": 0,
        "frame_age_ms": None,
        "error": None,
    }


def test_opencv_camera_open_reports_missing_opencv_dependency(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "cv2":
            raise ImportError("cv2 missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    camera = OpenCVCamera(OpenCVCameraConfig())

    result = camera.open()

    assert result["success"] is False
    assert result["message"] == "opencv-python is required for native camera capture"
    assert "ImportError: cv2 missing" in result["error"]


def test_opencv_camera_serves_latest_frame_without_reading_on_caller(
    monkeypatch,
) -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self.released = False
            self.read_count = 0

        def set(self, _property: int, _value: int) -> bool:
            return True

        def isOpened(self) -> bool:  # noqa: N802
            return True

        def read(self):
            time.sleep(0.005)
            self.read_count += 1
            frame = np.array([[[1, 2, self.read_count % 255]]], dtype=np.uint8)
            return True, frame

        def release(self) -> None:
            self.released = True

    capture = FakeCapture()
    fake_cv2 = SimpleNamespace(
        CAP_DSHOW=1,
        CAP_MSMF=2,
        CAP_V4L2=3,
        CAP_PROP_FRAME_WIDTH=4,
        CAP_PROP_FRAME_HEIGHT=5,
        CAP_PROP_FPS=6,
        CAP_PROP_BUFFERSIZE=7,
        VideoCapture=lambda *_args: capture,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    camera = OpenCVCamera(OpenCVCameraConfig(device_id=1, backend="dshow"))

    assert camera.open()["success"] is True
    frame_id, frame = camera.capture_frame(timeout_ms=200)
    started = time.monotonic()
    next_frame_id, next_frame = camera.capture_frame(timeout_ms=200)
    elapsed = time.monotonic() - started

    assert frame_id is not None
    assert frame_id >= 1
    assert next_frame_id is not None
    assert next_frame_id >= frame_id
    assert frame is not None
    assert next_frame is not None
    assert frame[0, 0, 2] == 1
    assert elapsed < 0.02
    assert camera.diagnostics()["frame_age_ms"] is not None

    camera.close()
    assert capture.released is True


def test_opencv_camera_open_failure_releases_capture_and_records_error(
    monkeypatch,
) -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self.released = False
            self.settings: list[tuple[int, int]] = []

        def set(self, property_id: int, value: int) -> bool:
            self.settings.append((property_id, value))
            return True

        def isOpened(self) -> bool:  # noqa: N802
            return False

        def release(self) -> None:
            self.released = True

    capture = FakeCapture()
    fake_cv2 = SimpleNamespace(
        CAP_DSHOW=1,
        CAP_MSMF=2,
        CAP_V4L2=3,
        CAP_PROP_FRAME_WIDTH=4,
        CAP_PROP_FRAME_HEIGHT=5,
        CAP_PROP_FPS=6,
        CAP_PROP_BUFFERSIZE=7,
        VideoCapture=lambda *_args: capture,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    camera = OpenCVCamera(
        OpenCVCameraConfig(
            device_id=7,
            width=640,
            height=480,
            fps=30,
            backend="msmf",
        )
    )

    result = camera.open()

    assert result == {"success": False, "message": "failed to open camera device 7"}
    assert capture.settings == [(4, 640), (5, 480), (6, 30), (7, 1)]
    assert capture.released is True
    assert camera.diagnostics()["error"] == "failed to open camera device 7"


def test_opencv_camera_capture_timeout_reports_last_error() -> None:
    camera = OpenCVCamera(OpenCVCameraConfig())

    assert camera.capture_frame(timeout_ms=1) == (None, None)

    with camera._condition:
        camera._capture = object()

    frame_id, frame = camera.capture_frame(timeout_ms=1)

    assert frame_id is None
    assert frame is None
    assert camera.diagnostics()["error"] == "no camera frame within 1 ms"


def test_opencv_camera_read_loop_records_read_failures_and_exits_on_stop(
    monkeypatch,
) -> None:
    class FailingCapture:
        def read(self):
            return False, None

    camera = OpenCVCamera(OpenCVCameraConfig())
    with camera._condition:
        camera._capture = FailingCapture()

    def stop_after_failure(_delay: float) -> None:
        with camera._condition:
            camera._stopping = True

    monkeypatch.setattr(time, "sleep", stop_after_failure)

    camera._read_loop()

    assert camera.diagnostics()["error"] == "camera frame read failed"


def test_opencv_camera_read_loop_exits_without_overwriting_error_after_stop() -> None:
    camera = OpenCVCamera(OpenCVCameraConfig())

    class FailingCapture:
        def read(self):
            with camera._condition:
                camera._stopping = True
            return False, None

    with camera._condition:
        camera._capture = FailingCapture()

    camera._read_loop()

    assert camera.diagnostics()["error"] is None


def test_opencv_camera_backend_and_color_conversion_helpers(monkeypatch) -> None:
    cv2 = SimpleNamespace(CAP_DSHOW=1, CAP_MSMF=2, CAP_V4L2=3)

    monkeypatch.setattr(sys, "platform", "linux")
    assert _opencv_backend(cv2, "auto") is None
    assert _opencv_backend(cv2, "default") is None
    assert _opencv_backend(cv2, "v4l2") == 3

    monkeypatch.setattr(sys, "platform", "win32")
    assert _opencv_backend(cv2, "auto") == 1
    assert _opencv_backend(cv2, "dshow") == 1
    assert _opencv_backend(cv2, "msmf") == 2
    assert _opencv_backend(cv2, "unknown") is None

    bgr = np.array([[[10, 20, 30, 40]]], dtype=np.uint8)
    gray = np.array([[9]], dtype=np.uint8)

    assert _bgr_to_rgb(bgr).tolist() == [[[30, 20, 10]]]
    assert _bgr_to_rgb(gray) is gray
