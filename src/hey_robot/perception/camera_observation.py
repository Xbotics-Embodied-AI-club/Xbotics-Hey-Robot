from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from hey_robot.protocol import RobotObservation


@dataclass(frozen=True)
class CameraObservationSnapshot:
    observation: RobotObservation
    received_at: float


class CameraObservationConsumer:
    """Consumer-side cache and rate gate for camera observation streams."""

    def __init__(
        self,
        *,
        robot_id: str | None = None,
        max_hz: float | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.robot_id = robot_id
        self.min_period_sec = (
            0.0 if max_hz is None or max_hz <= 0 else 1.0 / float(max_hz)
        )
        self._clock = clock or time.time
        self._latest: CameraObservationSnapshot | None = None
        self._last_accept_at = 0.0

    def ingest(self, observation: RobotObservation) -> bool:
        observed_robot_id = observation.envelope.robot_id
        if self.robot_id and observed_robot_id and observed_robot_id != self.robot_id:
            return False

        now = float(self._clock())
        if (
            self.min_period_sec > 0
            and self._last_accept_at > 0
            and now - self._last_accept_at < self.min_period_sec
        ):
            return False

        self._last_accept_at = now
        self._latest = CameraObservationSnapshot(
            observation=observation, received_at=now
        )
        return True

    def latest(self, *, max_age_ms: int | None = None) -> RobotObservation | None:
        snapshot = self._latest
        if snapshot is None:
            return None
        if max_age_ms is not None:
            age_ms = (float(self._clock()) - snapshot.received_at) * 1000.0
            if age_ms > max_age_ms:
                return None
        return snapshot.observation
