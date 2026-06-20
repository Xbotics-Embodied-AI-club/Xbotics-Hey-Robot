from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from hey_robot.capability.sensors import (
    ObservationStreamCamera,
    ObservationStreamCameraConfig,
)
from hey_robot.media import LocalMediaStore
from hey_robot.perception import CameraObservationConsumer
from hey_robot.protocol import Envelope, RobotObservation


def test_observation_stream_camera_reads_latest_frame(tmp_path) -> None:
    store = LocalMediaStore(tmp_path)
    image = np.full((4, 5, 3), 127, dtype=np.uint8)
    image_ref = store.put_image(image, robot_id="xlerobot", frame_id=1, camera="front")
    observation = RobotObservation(
        envelope=Envelope(robot_id="xlerobot"),
        frame_id=1,
        images=[image_ref],
    )
    camera = ObservationStreamCamera(
        CameraObservationConsumer(robot_id="xlerobot"),
        store,
        ObservationStreamCameraConfig(camera_name="front", max_age_ms=1000),
    )

    assert camera.ingest(observation)

    frame = camera.read()

    assert frame.shape == image.shape
    assert frame.dtype == np.uint8


def test_observation_stream_camera_connects_and_disconnects_without_warmup(
    tmp_path,
) -> None:
    class FakeConsumer:
        def __init__(self) -> None:
            self.ingested: list[Any] = []

        def ingest(self, observation) -> bool:
            self.ingested.append(observation)
            return True

        def latest(self, max_age_ms: int | None = None):
            _ = max_age_ms
            return

    camera = ObservationStreamCamera(
        cast(Any, FakeConsumer()), LocalMediaStore(tmp_path)
    )

    camera.connect(warmup=False)
    camera.disconnect()

    assert camera.is_connected is False


def test_observation_stream_camera_rejects_non_matching_camera(tmp_path) -> None:
    store = LocalMediaStore(tmp_path)
    image = np.full((4, 5, 3), 127, dtype=np.uint8)
    image_ref = store.put_image(image, robot_id="xlerobot", frame_id=1, camera="side")
    observation = RobotObservation(
        envelope=Envelope(robot_id="xlerobot"),
        frame_id=1,
        images=[image_ref],
    )
    camera = ObservationStreamCamera(
        CameraObservationConsumer(robot_id="xlerobot"),
        store,
        ObservationStreamCameraConfig(camera_name="front", max_age_ms=1000),
    )

    with pytest.raises(TimeoutError, match="has no 'front' image"):
        camera._resolve_frame(observation)


def test_observation_stream_camera_times_out_without_frame(tmp_path) -> None:
    camera = ObservationStreamCamera(
        CameraObservationConsumer(robot_id="xlerobot"),
        LocalMediaStore(tmp_path),
    )

    with pytest.raises(TimeoutError, match="no camera observation"):
        camera.async_read(timeout_ms=1)
