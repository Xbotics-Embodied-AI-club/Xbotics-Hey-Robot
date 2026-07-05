from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pose3D:
    x: float
    y: float
    z: float

    def as_list(self) -> list[float]:
        return [self.x, self.y, self.z]


@dataclass(frozen=True)
class Detection:
    label: str
    bbox: tuple[int, int, int, int]
    confidence: float = 1.0


@dataclass(frozen=True)
class TrackedObject:
    track_id: int
    label: str
    bbox_2d: tuple[int, int, int, int]
    pose: Pose3D | None
    confidence: float = 1.0
