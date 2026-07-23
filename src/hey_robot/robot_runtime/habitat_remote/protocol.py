from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class HabitatAsset:
    kind: str
    data: bytes
    role: str | None = None
    name: str | None = None
    content_type: str | None = None
    width: int | None = None
    height: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HabitatObservation:
    episode_id: str
    frame_id: int
    assets: list[HabitatAsset] = field(default_factory=list)
    proprioception: list[float] = field(default_factory=list)
    task: str | None = None
    done: bool = False
    success: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    entities: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HabitatStep:
    observation: HabitatObservation
    reward: float = 0.0
    done: bool = False
    success: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HabitatSkillResult:
    observation: HabitatObservation
    steps: int
    success: bool
    failure_mode: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    cancelled: bool = False


class HabitatRuntimeClient(Protocol):
    async def health(self) -> dict[str, Any]: ...

    async def create_episode(
        self,
        *,
        profile: str,
        task: str,
        split: str,
        requested_dataset_episode_id: str | None,
        seed: int,
        controlled_agent: str,
    ) -> HabitatObservation: ...

    async def observe(self, *, episode_id: str) -> HabitatObservation: ...

    async def step(
        self, *, episode_id: str, action: dict[str, Any], expected_frame_id: int
    ) -> HabitatStep: ...

    async def execute_skill(
        self,
        *,
        episode_id: str,
        operation_id: str,
        skill_name: str,
        arguments: dict[str, Any],
        expected_frame_id: int,
        max_steps: int,
    ) -> HabitatSkillResult: ...

    async def cancel_skill(self, *, episode_id: str, operation_id: str) -> bool: ...

    async def reset(
        self, *, episode_id: str, expected_frame_id: int
    ) -> HabitatObservation: ...

    async def close_episode(
        self, *, episode_id: str, expected_frame_id: int
    ) -> bool: ...

    async def close(self) -> None: ...
