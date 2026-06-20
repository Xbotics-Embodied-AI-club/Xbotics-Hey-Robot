from __future__ import annotations

from dataclasses import replace

from hey_robot.config import NotificationSpec
from hey_robot.notifications.models import (
    Notification,
    NotificationSeverity,
    NotificationTarget,
)


class NotificationPolicy:
    def __init__(self, spec: NotificationSpec | None = None) -> None:
        self.spec = spec or NotificationSpec()

    def expand(self, notification: Notification) -> list[Notification]:
        normalized = replace(notification, severity=self._severity(notification))
        if normalized.target.mode == "explicit":
            return [normalized]
        route_channels = self._route_channels(normalized)
        if not route_channels:
            return [normalized]
        routed: list[Notification] = []
        for channel in route_channels:
            target = self._target_for_channel(normalized, channel)
            if target is None:
                continue
            routed.append(replace(normalized, target=target))
        return routed

    def _severity(self, notification: Notification) -> NotificationSeverity:
        kind_cfg = self.spec.kinds.get(notification.kind, {})
        raw = kind_cfg.get(
            "severity", self.spec.defaults.get("severity", notification.severity)
        )
        value = str(raw or notification.severity).strip().lower()
        if value == "info":
            return "info"
        if value == "warning":
            return "warning"
        if value == "critical":
            return "critical"
        return notification.severity

    def _route_channels(self, notification: Notification) -> list[str]:
        kind_cfg = self.spec.kinds.get(notification.kind, {})
        raw = kind_cfg.get("channels", self.spec.defaults.get("channels", []))
        if not isinstance(raw, list):
            return []
        ordered: list[str] = []
        seen: set[str] = set()
        for item in raw:
            channel = str(item or "").strip()
            if not channel or channel in seen:
                continue
            ordered.append(channel)
            seen.add(channel)
        return ordered

    def _target_for_channel(
        self, notification: Notification, channel: str
    ) -> NotificationTarget | None:
        if (
            notification.origin_envelope is not None
            and notification.origin_envelope.channel == channel
        ):
            base_cfg = self.spec.channels.get(channel, {})
            return NotificationTarget(
                mode="episode",
                reply_to_current=bool(
                    base_cfg.get(
                        "reply_to_current", notification.target.reply_to_current
                    )
                ),
            )
        channel_cfg = dict(self.spec.channels.get(channel, {}) or {})
        kind_cfg = self.spec.kinds.get(notification.kind, {})
        target_overrides = dict(
            dict(kind_cfg.get("targets", {}) or {}).get(channel, {}) or {}
        )
        merged = {**channel_cfg, **target_overrides}
        return NotificationTarget(
            mode="explicit",
            channel=channel,
            chat_id=str(merged.get("chat_id") or "").strip() or None,
            sender_id=str(merged.get("sender_id") or "").strip() or None,
            message_id=str(merged.get("message_id") or "").strip() or None,
            reply_to_current=bool(merged.get("reply_to_current", False)),
        )
