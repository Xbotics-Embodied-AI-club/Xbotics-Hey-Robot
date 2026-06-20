from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

from hey_robot.protocol import Envelope

DEFAULT_EPISODE_DIMENSIONS = ["channel", "chat", "sender", "robot"]
VALID_DIMENSIONS = {"channel", "account", "user", "chat", "sender", "robot", "agent"}


@dataclass(frozen=True)
class EpisodeScope:
    version: str = "v1"
    agent_id: str = "main"
    dimensions: tuple[str, ...] = ()
    values: dict[str, str] = field(default_factory=dict)

    def canonical_signature(self) -> str:
        parts = [self.version, f"agent={self.agent_id}"]
        parts.extend(f"{dim}={self.values.get(dim, '')}" for dim in self.dimensions)
        return "|".join(parts)

    def key(self) -> str:
        digest = hashlib.sha256(self.canonical_signature().encode("utf-8")).hexdigest()[
            :32
        ]
        return f"ep_{self.version}_{digest}"

    def aliases(self) -> list[str]:
        compact = ":".join(
            f"{dim}={self.values.get(dim, '')}" for dim in self.dimensions
        )
        return (
            [f"agent:{self.agent_id}:{compact}"]
            if compact
            else [f"agent:{self.agent_id}"]
        )


@dataclass(frozen=True)
class EpisodeAllocation:
    scope: EpisodeScope
    episode_id: str
    aliases: list[str]


def allocate_episode(
    envelope: Envelope, *, agent_id: str, dimensions: list[str]
) -> EpisodeAllocation:
    clean_dimensions: list[str] = []
    for dim in dimensions:
        normalized = dim.lower().strip()
        if normalized in VALID_DIMENSIONS and normalized not in clean_dimensions:
            clean_dimensions.append(normalized)
    clean_dimensions.sort()

    values: dict[str, str] = {}
    for dim in clean_dimensions:
        value = _dimension_value(envelope, dim, agent_id)
        if value:
            values[dim] = value

    scope = EpisodeScope(
        agent_id=agent_id, dimensions=tuple(clean_dimensions), values=values
    )
    return EpisodeAllocation(
        scope=scope, episode_id=scope.key(), aliases=scope.aliases()
    )


def _dimension_value(envelope: Envelope, dimension: str, agent_id: str) -> str:
    if dimension == "channel":
        return envelope.channel or "unknown"
    if dimension == "account":
        return envelope.account_id or "default"
    if dimension == "user":
        return envelope.user_id or envelope.sender_id or "anonymous"
    if dimension == "chat":
        chat_id = envelope.chat_id or envelope.sender_id or "direct"
        chat_type = envelope.chat_type or "direct"
        return f"{chat_type}:{chat_id}"
    if dimension == "sender":
        return envelope.sender_id or "anonymous"
    if dimension == "robot":
        return envelope.robot_id or "none"
    if dimension == "agent":
        return agent_id
    return ""


def scope_to_dict(scope: EpisodeScope) -> dict:
    return asdict(scope)
