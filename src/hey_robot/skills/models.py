"""Domain models shared by the slow harness and fast skill worker."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from hey_robot.protocol import ArtifactRef, Envelope, ImageRef

if TYPE_CHECKING:
    from hey_robot.skills.context import SkillContext


SkillPhase = Literal[
    "accepted",
    "running",
    "progress",
    "completed",
    "failed",
    "cancelled",
]
SkillStatus = Literal["completed", "failed", "cancelled"]


@dataclass(frozen=True)
class SkillResult:
    success: bool
    summary: str
    status: SkillStatus
    data: dict[str, Any] = field(default_factory=dict)
    evidence_ids: tuple[str, ...] = ()
    observations: tuple[ImageRef, ...] = ()
    artifacts: tuple[ArtifactRef, ...] = ()
    failure_mode: str | None = None
    error: str | None = None


SkillHandler = Callable[["SkillContext", dict[str, Any]], Awaitable[SkillResult]]


@dataclass(frozen=True)
class Skill:
    """One executable robot capability and its agent-facing schema."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: SkillHandler
    resources: tuple[str, ...] = ()
    timeout_sec: float = 60.0
    supported_robots: tuple[str, ...] = ()
    required_actions: tuple[str, ...] = ()
    required_models: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkillCommand:
    envelope: Envelope
    run_id: str
    task_id: str
    robot_id: str
    name: str
    arguments: dict[str, Any]
    deadline_at: float | None = None


@dataclass(frozen=True)
class SkillCancel:
    envelope: Envelope
    run_id: str
    reason: str


@dataclass(frozen=True)
class SkillEvent:
    envelope: Envelope
    run_id: str
    sequence: int
    name: str
    phase: SkillPhase
    timestamp: float
    progress: float | None = None
    frame_id: int | None = None
    summary: str | None = None
    result: SkillResult | None = None
