from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class RemoteImage:
    camera: str
    data: bytes
    content_type: str = "image/jpeg"
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class RemoteObservation:
    episode_id: str
    frame_id: int
    state: list[float]
    images: list[RemoteImage] = field(default_factory=list)
    task: str | None = None
    done: bool = False
    success: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RemoteStep:
    observation: RemoteObservation
    reward: float
    done: bool
    success: bool
    metrics: dict[str, Any] = field(default_factory=dict)


class RemoteEpisodeClient(Protocol):
    async def health(self) -> dict[str, Any]: ...

    async def create_episode(self, *, task: str, seed: int) -> RemoteObservation: ...

    async def observe(self, *, episode_id: str) -> RemoteObservation: ...

    async def step(
        self, *, episode_id: str, action: list[float], expected_frame_id: int
    ) -> RemoteStep: ...

    async def reset(self, *, episode_id: str) -> RemoteObservation: ...

    async def close_episode(self, *, episode_id: str) -> bool: ...

    async def close(self) -> None: ...
