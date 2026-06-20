from __future__ import annotations

import json
import secrets
import string
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hey_robot.config import IdentitySpec
from hey_robot.protocol import Envelope


@dataclass(frozen=True)
class IdentityResolution:
    user_id: str | None
    matched_key: str | None = None


@dataclass(frozen=True)
class PendingBinding:
    code: str
    user_id: str
    source_channel: str
    source_sender_id: str | None
    source_chat_id: str | None
    created_at: float
    expires_at: float

    @property
    def expired(self) -> bool:
        return time.time() >= self.expires_at


@dataclass(frozen=True)
class ClaimedBinding:
    code: str
    user_id: str
    source_channel: str
    source_sender_id: str | None
    source_chat_id: str | None
    target_channel: str
    target_sender_id: str | None
    target_chat_id: str | None
    created_at: float
    expires_at: float
    claimed_at: float


class IdentityResolver:
    """Resolve a stable internal user_id from channel-facing envelope fields."""

    def __init__(
        self, spec: IdentitySpec, *, state_path: str | Path | None = None
    ) -> None:
        self.spec = spec
        self._static_bindings = {
            str(key): str(value) for key, value in spec.bindings.items()
        }
        self._dynamic_bindings: dict[str, str] = {}
        self._pending_bindings: dict[str, PendingBinding] = {}
        self._claimed_bindings: dict[str, ClaimedBinding] = {}
        self._state_path = Path(state_path) if state_path is not None else None
        self._load_state()

    def resolve(self, envelope: Envelope) -> IdentityResolution:
        explicit = str(envelope.user_id or "").strip()
        if explicit:
            return IdentityResolution(user_id=explicit, matched_key="envelope.user_id")
        if not self.spec.enabled:
            return IdentityResolution(user_id=None, matched_key=None)

        for key in self._candidate_keys(envelope):
            user_id = self._dynamic_bindings.get(key) or self._static_bindings.get(key)
            if user_id:
                return IdentityResolution(user_id=user_id, matched_key=key)

        default_user_id = str(self.spec.default_user_id or "").strip() or None
        return IdentityResolution(
            user_id=default_user_id,
            matched_key="identity.default_user_id" if default_user_id else None,
        )

    def create_binding(
        self, envelope: Envelope, *, ttl_sec: float = 600.0
    ) -> PendingBinding:
        resolved = self.resolve(envelope)
        user_id = resolved.user_id or self._new_user_id()
        self._bind_envelope(envelope, user_id)
        binding = PendingBinding(
            code=self._new_binding_code(),
            user_id=user_id,
            source_channel=str(envelope.channel or "web"),
            source_sender_id=str(envelope.sender_id or "").strip() or None,
            source_chat_id=str(envelope.chat_id or "").strip() or None,
            created_at=time.time(),
            expires_at=time.time() + max(30.0, float(ttl_sec)),
        )
        self._pending_bindings[binding.code] = binding
        self._save_state()
        return binding

    def claim_binding(self, code: str, envelope: Envelope) -> PendingBinding | None:
        binding = self._pending_bindings.get(code.strip().upper())
        if binding is None:
            return None
        if binding.expired:
            self._pending_bindings.pop(binding.code, None)
            self._save_state()
            return None
        self._bind_envelope(envelope, binding.user_id)
        self._claimed_bindings[binding.code] = ClaimedBinding(
            code=binding.code,
            user_id=binding.user_id,
            source_channel=binding.source_channel,
            source_sender_id=binding.source_sender_id,
            source_chat_id=binding.source_chat_id,
            target_channel=str(envelope.channel or ""),
            target_sender_id=str(envelope.sender_id or "").strip() or None,
            target_chat_id=str(envelope.chat_id or "").strip() or None,
            created_at=binding.created_at,
            expires_at=binding.expires_at,
            claimed_at=time.time(),
        )
        self._pending_bindings.pop(binding.code, None)
        self._save_state()
        return binding

    def binding_status(self, code: str) -> PendingBinding | ClaimedBinding | None:
        normalized = code.strip().upper()
        binding = self._pending_bindings.get(normalized)
        if binding is None:
            return self._claimed_bindings.get(normalized)
        if binding.expired:
            self._pending_bindings.pop(binding.code, None)
            self._save_state()
            return None
        return binding

    def bindings_snapshot(self) -> dict[str, Any]:
        return {
            "static_bindings": dict(self._static_bindings),
            "dynamic_bindings": dict(self._dynamic_bindings),
            "pending_bindings": [
                asdict(item)
                for item in self._pending_bindings.values()
                if not item.expired
            ],
            "claimed_bindings": [
                asdict(item) for item in self._claimed_bindings.values()
            ],
        }

    def linked_channel_targets(
        self, user_id: str, channel: str | None = None
    ) -> list[Envelope]:
        normalized_user_id = str(user_id or "").strip()
        if not normalized_user_id:
            return []
        preferred_channel = str(channel or "").strip() or None
        ordered = sorted(
            self._claimed_bindings.values(),
            key=lambda item: item.claimed_at,
            reverse=True,
        )
        seen: set[tuple[str, str, str | None]] = set()
        targets: list[Envelope] = []

        def add_target(
            target_channel: str | None, chat_id: str | None, sender_id: str | None
        ) -> None:
            resolved_channel = str(target_channel or "").strip()
            resolved_chat_id = str(chat_id or "").strip()
            resolved_sender_id = str(sender_id or "").strip() or None
            if not resolved_channel or not resolved_chat_id:
                return
            if preferred_channel is not None and resolved_channel != preferred_channel:
                return
            dedupe_key = (resolved_channel, resolved_chat_id, resolved_sender_id)
            if dedupe_key in seen:
                return
            seen.add(dedupe_key)
            targets.append(
                Envelope(
                    channel=resolved_channel,
                    chat_id=resolved_chat_id,
                    sender_id=resolved_sender_id,
                    user_id=normalized_user_id,
                )
            )

        for binding in ordered:
            if binding.user_id != normalized_user_id:
                continue
            add_target(
                binding.target_channel, binding.target_chat_id, binding.target_sender_id
            )
            add_target(
                binding.source_channel, binding.source_chat_id, binding.source_sender_id
            )
        return targets

    def known_channels(self, user_id: str) -> list[str]:
        normalized_user_id = str(user_id or "").strip()
        if not normalized_user_id:
            return []
        channels: set[str] = set()
        for key, value in {**self._static_bindings, **self._dynamic_bindings}.items():
            if value != normalized_user_id:
                continue
            channel = _channel_from_binding_key(key)
            if channel:
                channels.add(channel)
        for binding in self._claimed_bindings.values():
            if binding.user_id != normalized_user_id:
                continue
            if binding.source_channel:
                channels.add(binding.source_channel)
            if binding.target_channel:
                channels.add(binding.target_channel)
        return sorted(channels)

    def _bind_envelope(self, envelope: Envelope, user_id: str) -> None:
        for key in self._candidate_keys(envelope):
            self._dynamic_bindings[key] = user_id

    def _load_state(self) -> None:
        if self._state_path is None or not self._state_path.exists():
            return
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        bindings = payload.get("dynamic_bindings", {})
        pending = payload.get("pending_bindings", [])
        claimed = payload.get("claimed_bindings", [])
        if isinstance(bindings, dict):
            self._dynamic_bindings = {
                str(key): str(value) for key, value in bindings.items()
            }
        loaded_pending: dict[str, PendingBinding] = {}
        for item in pending if isinstance(pending, list) else []:
            if not isinstance(item, dict):
                continue
            try:
                binding = PendingBinding(**item)
            except TypeError:
                continue
            if not binding.expired:
                loaded_pending[binding.code] = binding
        self._pending_bindings = loaded_pending
        loaded_claimed: dict[str, ClaimedBinding] = {}
        for item in claimed if isinstance(claimed, list) else []:
            if not isinstance(item, dict):
                continue
            try:
                claimed_binding = ClaimedBinding(**item)
            except TypeError:
                continue
            loaded_claimed[claimed_binding.code] = claimed_binding
        self._claimed_bindings = loaded_claimed

    def _save_state(self) -> None:
        if self._state_path is None:
            return
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dynamic_bindings": self._dynamic_bindings,
            "pending_bindings": [
                asdict(item)
                for item in self._pending_bindings.values()
                if not item.expired
            ],
            "claimed_bindings": [
                asdict(item) for item in self._claimed_bindings.values()
            ],
        }
        self._state_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _new_user_id() -> str:
        return f"user_{secrets.token_hex(4)}"

    @staticmethod
    def _new_binding_code(length: int = 6) -> str:
        alphabet = string.ascii_uppercase + string.digits
        return "".join(secrets.choice(alphabet) for _ in range(length))

    @staticmethod
    def _candidate_keys(envelope: Envelope) -> list[str]:
        channel = str(envelope.channel or "").strip()
        account = str(envelope.account_id or "").strip()
        sender = str(envelope.sender_id or "").strip()
        chat = str(envelope.chat_id or "").strip()
        keys: list[str] = []
        if channel and sender:
            keys.append(f"{channel}:sender:{sender}")
        if channel and chat:
            keys.append(f"{channel}:chat:{chat}")
        if channel and account:
            keys.append(f"{channel}:account:{account}")
        if sender:
            keys.append(f"sender:{sender}")
        if chat:
            keys.append(f"chat:{chat}")
        if account:
            keys.append(f"account:{account}")
        return keys


def _channel_from_binding_key(key: str) -> str | None:
    normalized = str(key or "").strip()
    if not normalized:
        return None
    head, sep, _tail = normalized.partition(":")
    if not sep:
        return None
    if head in {"sender", "chat", "account"}:
        return None
    return head or None
