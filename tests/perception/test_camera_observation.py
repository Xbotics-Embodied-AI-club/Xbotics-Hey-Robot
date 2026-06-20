from __future__ import annotations

from hey_robot.perception import CameraObservationConsumer
from hey_robot.protocol import Envelope, RobotObservation


class Clock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value


def _observation(robot_id: str, frame_id: int) -> RobotObservation:
    return RobotObservation(
        envelope=Envelope(robot_id=robot_id),
        frame_id=frame_id,
        raw={"perception": {"source": "camera_service"}},
    )


def test_camera_observation_consumer_filters_robot_and_tracks_latest() -> None:
    clock = Clock()
    consumer = CameraObservationConsumer(robot_id="xlerobot", clock=clock)

    assert not consumer.ingest(_observation("other", 1))
    assert consumer.latest() is None

    assert consumer.ingest(_observation("xlerobot", 2))
    latest = consumer.latest()
    assert latest is not None
    assert latest.frame_id == 2


def test_camera_observation_consumer_applies_consumer_side_frequency() -> None:
    clock = Clock()
    consumer = CameraObservationConsumer(robot_id="xlerobot", max_hz=2.0, clock=clock)

    assert consumer.ingest(_observation("xlerobot", 1))
    clock.value += 0.2
    assert not consumer.ingest(_observation("xlerobot", 2))
    latest = consumer.latest()
    assert latest is not None
    assert latest.frame_id == 1

    clock.value += 0.3
    assert consumer.ingest(_observation("xlerobot", 3))
    latest = consumer.latest()
    assert latest is not None
    assert latest.frame_id == 3


def test_camera_observation_consumer_rejects_stale_latest() -> None:
    clock = Clock()
    consumer = CameraObservationConsumer(robot_id="xlerobot", clock=clock)

    assert consumer.ingest(_observation("xlerobot", 1))
    clock.value += 2.0

    assert consumer.latest(max_age_ms=1000) is None
