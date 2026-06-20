from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from hey_robot.protocol import Envelope

NotificationSeverity = Literal["info", "warning", "critical"]
NotificationTargetMode = Literal["episode", "explicit"]


@dataclass(frozen=True)
class NotificationTarget:
    mode: NotificationTargetMode = "episode"
    channel: str | None = None
    chat_id: str | None = None
    sender_id: str | None = None
    message_id: str | None = None
    reply_to_current: bool = False


@dataclass(frozen=True)
class Notification:
    kind: str
    body: str
    severity: NotificationSeverity = "info"
    title: str = ""
    episode_id: str | None = None
    robot_id: str | None = None
    agent_id: str | None = None
    trace_id: str | None = None
    origin_envelope: Envelope | None = None
    target: NotificationTarget = field(default_factory=NotificationTarget)
    dedupe_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
