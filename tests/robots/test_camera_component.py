from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import numpy as np

from hey_robot.robots.components.camera import OpenCVCamera, OpenCVCameraConfig


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
