from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hey_robot.protocol import UserTurn
from hey_robot.protocol.messages import from_payload, to_payload


@dataclass
class RobotAgentCheckpoint:
    episode_id: str
    phase: str = "idle"
    skill_id: str | None = None
    updated_at: float = field(default_factory=time.time)
    pending_turns: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def pending_user_turns(self) -> list[UserTurn]:
        turns: list[UserTurn] = []
        for payload in self.pending_turns:
            if isinstance(payload, dict):
                with suppress(Exception):
                    turns.append(from_payload(UserTurn, payload))
        return turns


class RobotAgentCheckpointStore:
    """Small durable store for in-flight robot-agent turn state."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root) / "agent_checkpoints"
        self.root.mkdir(parents=True, exist_ok=True)

    def load(self, episode_id: str) -> RobotAgentCheckpoint | None:
        path = self._path(episode_id)
        if not path.exists():
            return None
        try:
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            return RobotAgentCheckpoint(**data)
        except (OSError, TypeError, json.JSONDecodeError):
            return None

    def save(self, checkpoint: RobotAgentCheckpoint) -> RobotAgentCheckpoint:
        checkpoint.updated_at = time.time()
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(checkpoint.episode_id)
        tmp = self.root / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(
                checkpoint.to_dict(),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        tmp.replace(path)
        return checkpoint

    def mark_phase(
        self,
        episode_id: str,
        *,
        phase: str,
        skill_id: str | None = None,
    ) -> RobotAgentCheckpoint:
        checkpoint = self.load(episode_id) or RobotAgentCheckpoint(
            episode_id=episode_id
        )
        checkpoint.phase = phase
        if skill_id is not None:
            checkpoint.skill_id = skill_id
        return self.save(checkpoint)

    def enqueue_pending_turn(
        self,
        turn: UserTurn,
        *,
        reason: str,
        intent: dict[str, Any] | None = None,
        active_skill_id: str | None = None,
    ) -> RobotAgentCheckpoint | None:
        episode_id = turn.envelope.episode_id
        if not episode_id:
            return None
        checkpoint = self.load(episode_id) or RobotAgentCheckpoint(
            episode_id=episode_id
        )
        payload = to_payload(turn)
        payload.setdefault("metadata", {})
        payload["metadata"] = {
            **dict(payload.get("metadata") or {}),
            "_pending_reason": reason,
            "_interaction_intent": intent or {"kind": reason},
            "_active_skill_id": active_skill_id,
            "_queued_at": time.time(),
        }
        checkpoint.pending_turns.append(payload)
        checkpoint.pending_turns = checkpoint.pending_turns[-20:]
        if active_skill_id:
            checkpoint.skill_id = active_skill_id
        return self.save(checkpoint)

    def pop_pending_turn(self, episode_id: str) -> UserTurn | None:
        checkpoint = self.load(episode_id)
        if checkpoint is None or not checkpoint.pending_turns:
            return None
        payload = checkpoint.pending_turns.pop(0)
        self.save(checkpoint)
        try:
            return from_payload(UserTurn, payload)
        except Exception:
            return None

    def pending_turns(self, episode_id: str | None) -> list[UserTurn]:
        if not episode_id:
            return []
        checkpoint = self.load(episode_id)
        return checkpoint.pending_user_turns() if checkpoint is not None else []

    def mark_execution_feedback(
        self,
        episode_id: str,
        *,
        skill_id: str,
        success: bool,
        summary: str,
        metadata: dict[str, Any] | None = None,
    ) -> RobotAgentCheckpoint:
        checkpoint = self.load(episode_id) or RobotAgentCheckpoint(
            episode_id=episode_id
        )
        checkpoint.skill_id = skill_id
        checkpoint.phase = "confirmed" if success else "feedback_failed"
        del summary, metadata
        return self.save(checkpoint)

    def reset_for_external_turn(self, episode_id: str) -> RobotAgentCheckpoint | None:
        checkpoint = self.load(episode_id)
        if checkpoint is None:
            return None
        checkpoint.phase = "idle"
        checkpoint.skill_id = None
        checkpoint.pending_turns = []
        return self.save(checkpoint)

    def list_recent(self, limit: int = 100) -> list[RobotAgentCheckpoint]:
        checkpoints: list[RobotAgentCheckpoint] = []
        for path in self.root.glob("*.agent_checkpoint.json"):
            try:
                with path.open(encoding="utf-8") as handle:
                    checkpoints.append(RobotAgentCheckpoint(**json.load(handle)))
            except (OSError, TypeError, json.JSONDecodeError):
                continue
        return sorted(checkpoints, key=lambda item: item.updated_at, reverse=True)[
            : max(1, limit)
        ]

    def clear_if_terminal(self, episode_id: str) -> None:
        checkpoint = self.load(episode_id)
        if checkpoint is None or checkpoint.pending_turns:
            return
        self.clear(episode_id)

    def clear(self, episode_id: str) -> None:
        path = self._path(episode_id)
        try:
            path.unlink()
        except FileNotFoundError:
            return

    def _path(self, episode_id: str) -> Path:
        return self.root / f"{_sanitize(episode_id)}.agent_checkpoint.json"


def _sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value)
