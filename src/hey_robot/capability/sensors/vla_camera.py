from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from hey_robot.media import LocalMediaStore
from hey_robot.perception import CameraObservationConsumer
from hey_robot.protocol import RobotObservation


@dataclass(frozen=True)
class ObservationStreamCameraConfig:
    camera_name: str = "front"
    max_age_ms: int = 1000


class ObservationStreamCamera:
    """LeRobot-style camera adapter backed by camera.observation frames."""

    def __init__(
        self,
        consumer: CameraObservationConsumer,
        media_store: LocalMediaStore,
        config: ObservationStreamCameraConfig | None = None,
    ) -> None:
        self.consumer = consumer
        self.media_store = media_store
        self.config = config or ObservationStreamCameraConfig()
        self.is_connected = False

    def connect(self, warmup: bool = True) -> None:
        self.is_connected = True
        if warmup:
            self.read_latest(max_age_ms=self.config.max_age_ms)

    def disconnect(self) -> None:
        self.is_connected = False

    def ingest(self, observation: RobotObservation) -> bool:
        return self.consumer.ingest(observation)

    def read(self) -> np.ndarray:
        return self.read_latest(max_age_ms=self.config.max_age_ms)

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        deadline = time.monotonic() + max(1.0, float(timeout_ms)) / 1000.0
        while time.monotonic() <= deadline:
            observation = self.consumer.latest(max_age_ms=self.config.max_age_ms)
            if observation is not None and observation.images:
                return self._resolve_frame(observation)
            time.sleep(0.005)
        raise TimeoutError(f"no camera observation within {timeout_ms:g} ms")

    def read_latest(self, max_age_ms: int | None = None) -> np.ndarray:
        observation = self.consumer.latest(max_age_ms=max_age_ms)
        if observation is None:
            raise TimeoutError("no fresh camera observation available")
        return self._resolve_frame(observation)

    def _resolve_frame(self, observation: RobotObservation) -> np.ndarray:
        image_ref = next(
            (
                image
                for image in observation.images
                if image.camera in {None, self.config.camera_name}
            ),
            None,
        )
        if image_ref is None:
            raise TimeoutError(
                f"camera observation has no {self.config.camera_name!r} image"
            )
        return self.media_store.resolve_image(image_ref)
