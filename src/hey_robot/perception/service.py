from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from hey_robot.media import LocalMediaStore
from hey_robot.perception.observation import DriverObservation
from hey_robot.perception.pipeline import ObservationPipeline
from hey_robot.protocol import RobotObservation


class ObservationDriver(Protocol):
    robot_id: str

    async def observe(self) -> DriverObservation: ...


@dataclass(frozen=True)
class PerceptionSnapshot:
    """Current perception state exposed by the robot runtime."""

    robot_id: str
    observation: RobotObservation
    observed_at: float
    source: str
    reason: str | None = None
    driver_observation: DriverObservation | None = None

    @property
    def age_ms(self) -> int:
        return max(0, int((time.time() - self.observed_at) * 1000))

    @property
    def has_images(self) -> bool:
        perception = self.observation.raw.get("perception")
        if isinstance(perception, dict) and "valid_image_count" in perception:
            return int(perception.get("valid_image_count") or 0) > 0
        return bool(self.observation.images)

    def summary(self) -> dict:
        perception = self.observation.raw.get("perception")
        valid_image_count = (
            int(perception.get("valid_image_count") or 0)
            if isinstance(perception, dict)
            else len(self.observation.images)
        )
        return {
            "robot_id": self.robot_id,
            "frame_id": self.observation.frame_id,
            "image_count": len(self.observation.images),
            "valid_image_count": valid_image_count,
            "artifact_count": len(self.observation.artifacts),
            "age_ms": self.age_ms,
            "source": self.source,
            "reason": self.reason,
            "task": self.observation.task,
        }


class PerceptionService:
    """Single runtime entry point for current robot perception.

    Drivers acquire raw observations from hardware or simulation. This service
    materializes those observations, keeps the latest runtime snapshot, and
    gives agents, policies, VLA, VLN, and future consumers one stable interface
    for the current world state.
    """

    def __init__(
        self,
        driver: ObservationDriver,
        media_store: LocalMediaStore,
        *,
        image_save_every_n: int = 1,
    ) -> None:
        self.driver = driver
        self.pipeline = ObservationPipeline(
            media_store, image_save_every_n=image_save_every_n
        )
        self._latest: PerceptionSnapshot | None = None

    async def refresh(self, *, reason: str | None = None) -> PerceptionSnapshot:
        driver_observation = await self.driver.observe()
        observation = self._annotate(
            self.pipeline.build(driver_observation),
            observed_at=driver_observation.timestamp,
            source="refresh",
            reason=reason,
        )
        snapshot = PerceptionSnapshot(
            robot_id=observation.envelope.robot_id or self.driver.robot_id,
            observation=observation,
            observed_at=driver_observation.timestamp,
            source="refresh",
            reason=reason,
            driver_observation=driver_observation,
        )
        self._latest = snapshot
        return snapshot

    def latest(self, *, max_age_ms: int | None = None) -> PerceptionSnapshot | None:
        snapshot = self._latest
        if snapshot is None:
            return None
        if max_age_ms is not None and snapshot.age_ms > max(0, int(max_age_ms)):
            return None
        return snapshot

    async def current(
        self,
        *,
        max_age_ms: int = 1000,
        refresh_if_stale: bool = True,
        reason: str | None = None,
    ) -> PerceptionSnapshot:
        snapshot = self.latest(max_age_ms=max_age_ms)
        if snapshot is not None:
            return snapshot
        if not refresh_if_stale and self._latest is not None:
            return self._latest
        return await self.refresh(reason=reason or "stale_or_missing")

    def build_observation(self, observation: DriverObservation) -> RobotObservation:
        materialized = self._annotate(
            self.pipeline.build(observation),
            observed_at=observation.timestamp,
            source="external",
            reason="build_observation",
        )
        self._latest = PerceptionSnapshot(
            robot_id=materialized.envelope.robot_id or self.driver.robot_id,
            observation=materialized,
            observed_at=observation.timestamp,
            source="external",
            reason="build_observation",
            driver_observation=observation,
        )
        return materialized

    @staticmethod
    def _annotate(
        observation: RobotObservation,
        *,
        observed_at: float,
        source: str,
        reason: str | None,
    ) -> RobotObservation:
        existing_perception = observation.raw.get("perception")
        perception_data = (
            existing_perception if isinstance(existing_perception, dict) else {}
        )
        image_quality = perception_data.get(
            "image_quality", observation.raw.get("image_quality")
        )
        image_quality = [item for item in image_quality or [] if isinstance(item, dict)]
        invalid_images = [item for item in image_quality if item.get("valid") is False]
        if image_quality:
            valid_image_count = sum(
                1 for item in image_quality if item.get("valid") is True
            )
        elif "valid_image_count" in perception_data:
            valid_image_count = int(perception_data.get("valid_image_count") or 0)
        else:
            valid_image_count = len(observation.images)
        existing_issues = [
            str(item)
            for item in perception_data.get("image_quality_issues", []) or []
            if item
        ]
        image_quality_issues = [
            str(item.get("issue")) for item in invalid_images if item.get("issue")
        ] or existing_issues
        perception = {
            **perception_data,
            "observed_at": observed_at,
            "source": source,
            "reason": reason,
            "image_count": len(observation.images),
            "valid_image_count": valid_image_count,
            "image_quality": image_quality,
            "image_quality_issues": image_quality_issues,
            "artifact_count": len(observation.artifacts),
        }
        return RobotObservation(
            envelope=observation.envelope,
            frame_id=observation.frame_id,
            images=observation.images,
            artifacts=observation.artifacts,
            proprioception=observation.proprioception,
            task=observation.task,
            raw={**observation.raw, "perception": perception},
        )
