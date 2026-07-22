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
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RemoteStep:
    observation: RemoteObservation
    reward: float
    done: bool
    metrics: dict[str, Any] = field(default_factory=dict)


class RemoteEpisodeClient(Protocol):
    async def health(self) -> dict[str, Any]: ...

    async def begin_trial(
        self,
        *,
        trial_id: str,
        task: str,
        seed: int,
        split: str = "target",
        registries: tuple[str, ...] = ("lightwheel",),
    ) -> RemoteObservation: ...

    async def observe(self) -> RemoteObservation: ...

    async def step(
        self,
        *,
        action: list[float],
        expected_frame_id: int,
        raw_action: list[float] | None = None,
        action_clipped: bool = False,
    ) -> RemoteStep: ...

    async def read_truth(self) -> dict[str, Any]: ...

    async def end_trial(self, *, reason: str = "completed") -> bool: ...

    async def close(self) -> None: ...
