"""Canonical message types for the deployable Hey Robot runtime.

These dataclasses are the boundary between independently deployed services.
Channels, agents, policies, and robot drivers exchange these shapes instead of
ad hoc dictionaries.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any, get_args, get_origin, get_type_hints

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class Envelope:
    trace_id: str = field(default_factory=lambda: _new_id("tr"))
    episode_id: str | None = None
    turn_id: str | None = None
    channel: str | None = None
    account_id: str | None = None
    user_id: str | None = None
    chat_id: str | None = None
    chat_type: str | None = None
    sender_id: str | None = None
    message_id: str | None = None
    reply_to_id: str | None = None
    robot_id: str | None = None
    agent_id: str | None = None
    deployment_id: str | None = None
    timestamp: float = field(default_factory=time.time)

    def child(self, **updates: Any) -> Envelope:
        data = asdict(self)
        data.update(updates)
        if not data.get("trace_id"):
            data["trace_id"] = _new_id("tr")
        return Envelope(**data)


@dataclass(frozen=True)
class MediaRef:
    uri: str
    media_type: str
    name: str | None = None
    content_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageRef:
    uri: str
    camera: str | None = None
    width: int | None = None
    height: int | None = None
    timestamp: float | None = None
    content_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactRef:
    uri: str
    artifact_type: str
    role: str | None = None
    name: str | None = None
    content_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UserTurn:
    envelope: Envelope
    text: str
    media: list[MediaRef] = field(default_factory=list)
    intent: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentReply:
    envelope: Envelope
    text: str
    media: list[MediaRef] = field(default_factory=list)
    final: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RobotObservation:
    envelope: Envelope
    frame_id: int
    images: list[ImageRef] = field(default_factory=list)
    artifacts: list[ArtifactRef] = field(default_factory=list)
    proprioception: list[float] = field(default_factory=list)
    task: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RobotStatus:
    envelope: Envelope
    frame_id: int | None = None
    state: str = "unknown"
    task: str | None = None
    skill_id: str | None = None
    success: bool | None = None
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillIntent:
    envelope: Envelope
    skill_id: str = field(default_factory=lambda: _new_id("skill"))
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    objective: str = ""
    priority: int = 0
    interrupt: bool = False
    timeout_sec: float | None = None
    feedback_mode: str = "status"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillEvent:
    envelope: Envelope
    skill_id: str
    name: str = ""
    phase: str = "created"
    step: str | None = None
    text: str | None = None
    mode: str | None = None
    policy_id: str | None = None
    steps_executed: int | None = None
    frame_id: int | None = None
    progress: float | None = None
    error: str | None = None
    summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RobotAction:
    envelope: Envelope
    values: list[float]
    action_id: str = field(default_factory=lambda: _new_id("act"))
    skill_id: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillResult:
    envelope: Envelope
    skill_id: str
    name: str = ""
    status: str = "unknown"
    success: bool | None = None
    steps_executed: int = 0
    progress: float = 0.0
    summary: str | None = None
    failure_mode: str | None = None
    frame_id: int | None = None
    error: str | None = None
    observations: list[ImageRef] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def to_payload(message: DataclassInstance) -> dict[str, Any]:
    return asdict(message)


def from_payload[T: DataclassInstance](cls: type[T], payload: dict[str, Any]) -> T:
    kwargs: dict[str, Any] = {}
    hints = get_type_hints(cls)
    for item in fields(cls):
        value = payload.get(item.name)
        if value is None:
            continue
        target = hints.get(item.name, item.type)
        if item.name == "envelope" and isinstance(value, dict):
            kwargs[item.name] = Envelope(**value)
            continue
        origin = get_origin(target)
        args = get_args(target)
        if origin is list and args and isinstance(value, list):
            subtype = args[0]
            if subtype in {MediaRef, ImageRef, ArtifactRef}:
                kwargs[item.name] = [
                    subtype(**entry) if isinstance(entry, dict) else entry
                    for entry in value
                ]
                continue
        kwargs[item.name] = value
    return cls(**kwargs)  # type: ignore[arg-type]
