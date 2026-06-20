from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hey_robot.interaction.intent import classify_user_interaction
from hey_robot.protocol import AgentReply, UserTurn


@dataclass
class InteractionState:
    episode_id: str
    active_task_id: str | None = None
    active_channel: str | None = None
    last_user_intent: str | None = None
    pending_confirmation: dict[str, Any] | None = None
    confirmation_reason: str | None = None
    last_spoken_status: str | None = None
    preferred_reply_mode: str = "text"
    linked_channels: tuple[str, ...] = ()
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["linked_channels"] = list(self.linked_channels)
        return data


class InteractionStateStore:
    """Durable multi-channel task interaction state.

    This is deliberately a product-facing query/write model. It does not make
    execution decisions and never talks to robot hardware.
    """

    def __init__(self, root: str | Path = "runtime/interaction") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "states.json"
        self._states = self._load()

    def get(self, episode_id: str | None) -> InteractionState | None:
        if not episode_id:
            return None
        return self._states.get(episode_id)

    def list_recent(self, limit: int = 50) -> list[InteractionState]:
        states = sorted(
            self._states.values(), key=lambda item: item.updated_at, reverse=True
        )
        return states[: max(1, int(limit))]

    def record_turn(
        self,
        turn: UserTurn,
        *,
        active_task_id: str | None = None,
        pending_confirmation: dict[str, Any] | None = None,
        robot_busy: bool = False,
    ) -> InteractionState | None:
        episode_id = turn.envelope.episode_id
        if not episode_id:
            return None
        intent = classify_user_interaction(turn.text, robot_busy=robot_busy)
        state = self._states.get(episode_id) or InteractionState(episode_id=episode_id)
        state.active_task_id = active_task_id or state.active_task_id
        state.active_channel = turn.envelope.channel or state.active_channel
        state.last_user_intent = intent.kind
        state.pending_confirmation = (
            dict(pending_confirmation)
            if isinstance(pending_confirmation, dict)
            else None
        )
        state.confirmation_reason = _confirmation_reason(state.pending_confirmation)
        state.preferred_reply_mode = _reply_mode_for_channel(turn.envelope.channel)
        state.linked_channels = _append_unique(
            state.linked_channels, turn.envelope.channel
        )
        state.updated_at = time.time()
        self._states[episode_id] = state
        self._write()
        return state

    def record_reply(self, reply: AgentReply) -> InteractionState | None:
        episode_id = reply.envelope.episode_id
        if not episode_id:
            return None
        state = self._states.get(episode_id) or InteractionState(episode_id=episode_id)
        state.active_channel = reply.envelope.channel or state.active_channel
        if reply.envelope.channel == "voice" and reply.final and reply.text.strip():
            state.last_spoken_status = reply.text.strip()
            state.preferred_reply_mode = "voice"
        state.linked_channels = _append_unique(
            state.linked_channels, reply.envelope.channel
        )
        state.updated_at = time.time()
        self._states[episode_id] = state
        self._write()
        return state

    def set_pending_confirmation(
        self,
        episode_id: str | None,
        pending_confirmation: dict[str, Any] | None,
    ) -> InteractionState | None:
        if not episode_id:
            return None
        state = self._states.get(episode_id) or InteractionState(episode_id=episode_id)
        state.pending_confirmation = (
            dict(pending_confirmation)
            if isinstance(pending_confirmation, dict)
            else None
        )
        state.confirmation_reason = _confirmation_reason(state.pending_confirmation)
        state.updated_at = time.time()
        self._states[episode_id] = state
        self._write()
        return state

    def _load(self) -> dict[str, InteractionState]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        states: dict[str, InteractionState] = {}
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict) or not item.get("episode_id"):
                    continue
                item["linked_channels"] = tuple(item.get("linked_channels") or ())
                states[str(item["episode_id"])] = InteractionState(**item)
        return states

    def _write(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        payload = [
            state.to_dict()
            for state in sorted(self._states.values(), key=lambda item: item.updated_at)
        ]
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(self.path)


def _reply_mode_for_channel(channel: str | None) -> str:
    if channel == "voice":
        return "voice"
    if channel in {"web", "cli"}:
        return "text"
    if channel == "feishu":
        return "notification"
    return "text"


def _append_unique(items: tuple[str, ...], item: str | None) -> tuple[str, ...]:
    if not item or item in items:
        return items
    return (*items, item)


def _confirmation_reason(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    reason = payload.get("reason") or payload.get("objective")
    return str(reason) if reason is not None else None
