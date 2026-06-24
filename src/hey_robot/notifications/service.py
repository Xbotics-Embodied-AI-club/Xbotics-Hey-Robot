from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from hey_robot.episode import JsonlEpisodeStore
from hey_robot.notifications.models import Notification, NotificationTarget
from hey_robot.notifications.policy import NotificationPolicy
from hey_robot.notifications.presentation import present_notification_text
from hey_robot.protocol import AgentReply, Envelope

ReplyPublisher = Callable[[AgentReply], Awaitable[None]]
LinkedTargetProvider = Callable[[str, str | None], list[Envelope]]


class NotificationService:
    """Small notification router for proactive operator/user updates."""

    def __init__(
        self,
        episodes: JsonlEpisodeStore,
        publish_reply: ReplyPublisher,
        *,
        dedupe_window_sec: float = 300.0,
        policy: NotificationPolicy | None = None,
        linked_target_provider: LinkedTargetProvider | None = None,
    ) -> None:
        self.episodes = episodes
        self.publish_reply = publish_reply
        self.dedupe_window_sec = max(0.0, float(dedupe_window_sec))
        self.policy = policy or NotificationPolicy()
        self.linked_target_provider = linked_target_provider
        self._recent: dict[tuple[str, str, str], float] = {}

    async def publish(self, notification: Notification) -> bool:
        published = False
        for routed in self.policy.expand(notification):
            for envelope in self._resolve_targets(routed):
                if envelope is None or not envelope.channel or not envelope.chat_id:
                    continue
                dedupe_key = self._dedupe_key(routed, envelope)
                if self._should_suppress(envelope, dedupe_key):
                    continue
                reply = AgentReply(
                    envelope=envelope,
                    text=self._render_text(routed),
                    metadata={
                        "proactive": True,
                        "notification": True,
                        "notification_kind": routed.kind,
                        "severity": routed.severity,
                        **dict(routed.metadata),
                    },
                )
                await self.publish_reply(reply)
                self._remember(envelope, dedupe_key)
                published = True
        return published

    def _resolve_targets(self, notification: Notification) -> list[Envelope]:
        target = notification.target
        if target.mode == "explicit":
            return self._explicit_targets(notification, target)
        envelope = self._episode_target(notification)
        return [envelope] if envelope is not None else []

    def _explicit_targets(
        self, notification: Notification, target: NotificationTarget
    ) -> list[Envelope]:
        base = notification.origin_envelope or Envelope()
        channel = target.channel or base.channel
        chat_id = target.chat_id or base.chat_id
        if channel and not chat_id:
            episode_target = self._episode_target(notification)
            if (
                episode_target is not None
                and episode_target.channel == channel
                and episode_target.chat_id
            ):
                return [episode_target]
            user_id = self._notification_user_id(notification)
            if user_id and self.linked_target_provider is not None:
                return self.linked_target_provider(user_id, channel)
        if not channel or not chat_id:
            return []
        same_target = channel == base.channel and chat_id == base.chat_id
        return [
            base.child(
                channel=channel,
                chat_id=chat_id,
                sender_id=target.sender_id or base.sender_id,
                message_id=(
                    target.message_id
                    if target.message_id is not None
                    else (
                        base.message_id
                        if same_target and target.reply_to_current
                        else None
                    )
                ),
                reply_to_id=base.reply_to_id
                if same_target and target.reply_to_current
                else None,
                episode_id=notification.episode_id or base.episode_id,
                robot_id=notification.robot_id or base.robot_id,
                agent_id=notification.agent_id or base.agent_id,
                trace_id=notification.trace_id or base.trace_id,
                user_id=base.user_id or self._notification_user_id(notification),
            )
        ]

    def _notification_user_id(self, notification: Notification) -> str | None:
        origin = notification.origin_envelope
        if origin is not None and origin.user_id:
            return origin.user_id
        episode_id = notification.episode_id
        if not episode_id:
            return None
        for record in reversed(self.episodes.history(episode_id, limit=50)):
            payload = record.payload if isinstance(record.payload, dict) else {}
            envelope_payload = payload.get("envelope")
            if not isinstance(envelope_payload, dict):
                continue
            user_id = str(envelope_payload.get("user_id") or "").strip()
            if user_id:
                return user_id
        return None

    def _episode_target(self, notification: Notification) -> Envelope | None:
        if notification.origin_envelope is not None:
            origin = notification.origin_envelope
            if origin.channel and origin.chat_id:
                return origin.child(
                    episode_id=notification.episode_id or origin.episode_id,
                    robot_id=notification.robot_id or origin.robot_id,
                    agent_id=notification.agent_id or origin.agent_id,
                    trace_id=notification.trace_id or origin.trace_id,
                )
        episode_id = notification.episode_id
        if not episode_id:
            return None
        for record in reversed(self.episodes.history(episode_id, limit=50)):
            payload = record.payload if isinstance(record.payload, dict) else {}
            envelope_payload = payload.get("envelope")
            if not isinstance(envelope_payload, dict):
                continue
            try:
                envelope = Envelope(**envelope_payload)
            except TypeError:
                continue
            if envelope.channel and envelope.chat_id:
                return envelope.child(
                    episode_id=episode_id,
                    robot_id=notification.robot_id or envelope.robot_id,
                    agent_id=notification.agent_id or envelope.agent_id,
                    trace_id=notification.trace_id or envelope.trace_id,
                )
        return None

    @staticmethod
    def _render_text(notification: Notification) -> str:
        title = notification.title.strip()
        body = present_notification_text(notification.body)
        if title and body:
            return f"{title}\n{body}"
        return title or body

    def _dedupe_key(self, notification: Notification, envelope: Envelope) -> str:
        raw = notification.dedupe_key or notification.metadata.get("dedupe_key")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        parts = [
            notification.kind,
            notification.severity,
            notification.episode_id or envelope.episode_id or "",
            notification.robot_id or envelope.robot_id or "",
            notification.body.strip(),
        ]
        return "|".join(parts)

    def _should_suppress(self, envelope: Envelope, dedupe_key: str) -> bool:
        if self.dedupe_window_sec <= 0:
            return False
        now = time.time()
        cache_key = (envelope.channel or "", envelope.chat_id or "", dedupe_key)
        last = self._recent.get(cache_key)
        return last is not None and now - last < self.dedupe_window_sec

    def _remember(self, envelope: Envelope, dedupe_key: str) -> None:
        if self.dedupe_window_sec <= 0:
            return
        now = time.time()
        self._recent[(envelope.channel or "", envelope.chat_id or "", dedupe_key)] = now
        cutoff = now - self.dedupe_window_sec
        self._recent = {
            key: value for key, value in self._recent.items() if value >= cutoff
        }
