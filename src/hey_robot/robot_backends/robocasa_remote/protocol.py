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
    done: bool
    status: str
    actions_executed: int
    chunks_executed: int
    progress: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


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
        execution_artifact_dir: str | None = None,
    ) -> RemoteObservation: ...

    async def observe(self) -> RemoteObservation: ...

    async def run_option(
        self,
        *,
        session_id: str,
        instruction: str,
        max_actions: int,
        reset_session: bool = False,
    ) -> RemoteStep: ...

    async def step_native(
        self, *, action: list[float], expected_frame_id: int
    ) -> RemoteStep: ...

    async def localize_pixels(
        self,
        *,
        camera: str,
        pixels: list[list[int]],
        expected_frame_id: int,
    ) -> dict[str, Any]: ...

    async def read_truth(self) -> dict[str, Any]: ...

    async def end_trial(self, *, reason: str = "completed") -> bool: ...

    async def close(self) -> None: ...
