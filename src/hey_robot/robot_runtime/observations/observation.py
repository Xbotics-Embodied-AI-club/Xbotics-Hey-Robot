from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from hey_robot.protocol import Envelope, ImageRef


@dataclass(frozen=True)
class ObservationAsset:
    kind: str
    data: Any
    role: str | None = None
    name: str | None = None
    content_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DriverObservation:
    envelope: Envelope
    frame_id: int
    assets: list[ObservationAsset] = field(default_factory=list)
    images: list[ImageRef] = field(default_factory=list)
    proprioception: list[float] = field(default_factory=list)
    task: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
