from __future__ import annotations

import re
import time

from hey_robot.audio import VoiceInteractionLoop, voice_config_from_settings
from hey_robot.channels.base import ChannelContext, InboundHandler
from hey_robot.events import RuntimeEvent
from hey_robot.notifications import (
    format_notification_text,
    should_deliver_notification,
)
from hey_robot.protocol import AgentReply, Envelope, UserTurn
from hey_robot.user_reply import (
    looks_like_internal_user_reply,
    present_runtime_event_for_user,
)


class VoiceChannel:
    """Local voice channel for embodied interaction.

    The channel owns only the user-facing audio interface. It converts local
    speech into normal UserTurn messages and optionally speaks AgentReply text
    back to the operator.
    """

    def __init__(self, context: ChannelContext) -> None:
        self.context = context
        self.name = context.name
        self.config = voice_config_from_settings(context.spec.settings)
        self.loop = VoiceInteractionLoop(self.config)
        configured_levels = context.spec.settings.get(
            "notification_levels", ["warning", "critical"]
        )
        levels = {
            str(level).strip().lower()
            for level in configured_levels or []
            if str(level).strip()
        }
        self._notification_levels = levels or {"warning", "critical"}
        self._spoken_event_keys: set[str] = set()

    async def start(self, handler: InboundHandler) -> None:
        async def on_text(text: str, metadata: dict) -> None:
            if _is_nonsense_asr(text):
                return
            if _needs_voice_clarification(text):
                await self.loop.speak("请说清楚要观察哪里，或要我做哪个动作。")
                return
            envelope = Envelope(
                channel=self.name,
                account_id=self.context.spec.account_id or self.name,
                user_id=(
                    str(self.context.spec.settings.get("user_id")).strip() or None
                    if self.context.spec.settings.get("user_id") is not None
                    else None
                ),
                chat_id=self.config.chat_id,
                chat_type="voice",
                sender_id=self.config.sender_id,
                deployment_id=self.context.deployment_id,
                timestamp=time.time(),
            )
            await handler(UserTurn(envelope=envelope, text=text, metadata=metadata))

        await self.loop.start(on_text)

    async def send(self, reply: AgentReply) -> None:
        if reply.metadata.get("notification"):
            if not should_deliver_notification(reply, self._notification_levels):
                return
            if reply.envelope.channel not in {None, self.name}:
                return
            await self.loop.speak(format_notification_text(reply))
            return
        if reply.envelope.channel != self.name:
            return
        if not reply.final:
            return
        if _is_internal_or_technical_reply(reply):
            return
        await self.loop.speak(reply.text)

    async def on_event(self, event: RuntimeEvent) -> None:
        if not isinstance(event, RuntimeEvent):
            return
        if event.channel not in {None, self.name}:
            return
        payload = event.payload if isinstance(event.payload, dict) else {}
        text = present_runtime_event_for_user(kind=event.kind, payload=payload)
        if text is None:
            return
        key = _voice_event_key(event)
        if key in self._spoken_event_keys:
            return
        self._spoken_event_keys.add(key)
        if len(self._spoken_event_keys) > 200:
            self._spoken_event_keys = set(list(self._spoken_event_keys)[-100:])
        await self.loop.speak(text)

    async def stop(self) -> None:
        await self.loop.stop()


def _is_internal_or_technical_reply(reply: AgentReply) -> bool:
    text = reply.text.strip()
    if text.startswith("Execution feedback for skill "):
        return True
    if looks_like_internal_user_reply(text):
        return True
    return (
        reply.metadata.get("tool") == "request_capability"
        and "subgoal_success:" in text
    )


_REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{2,}")


def _is_nonsense_asr(text: str) -> bool:
    """Drop ASR output that is clearly noise artifacts, not real speech.

    Catches patterns like ``请请请`` or ``啊啊啊`` where the ASR engine
    hallucinates repeated characters from ambient sound.
    """
    return bool(_REPEATED_CHAR_PATTERN.search(str(text or "")))


def _needs_voice_clarification(text: str) -> bool:
    normalized = "".join(str(text or "").lower().split())
    if not normalized:
        return False
    if _is_specific_action_intent(normalized):
        return False
    vague_markers = ("想想", "看看吧", "好吧", "行吧", "随便")
    if not any(marker in normalized for marker in vague_markers):
        return False
    specific_markers = (
        "前方",
        "前面",
        "左",
        "右",
        "桌",
        "门",
        "地上",
        "行李箱",
        "靠近",
        "前进",
        "后退",
        "转",
        "停止",
        "停下",
        "状态",
        "电池",
        "什么",
        "看到",
        "看见",
        "吗",
        "做",
        "能",
        "知道",
        "怎么",
        "哪里",
        "在哪",
        "有没有",
        "是不是",
    )
    return not any(marker in normalized for marker in specific_markers)


def _is_specific_action_intent(text: str) -> bool:
    """Return True when the text carries actionable intent despite voice vagueness."""
    markers = (
        "跟随",
        "跟着",
        "启动",
        "确认",
        "好的",
    )
    return any(marker in text for marker in markers)


def _voice_event_key(event: RuntimeEvent) -> str:
    payload = event.payload if isinstance(event.payload, dict) else {}
    return ":".join(
        str(item or "")
        for item in (
            event.episode_id,
            payload.get("skill_id"),
            payload.get("phase"),
            payload.get("step"),
            payload.get("summary") or payload.get("error"),
        )
    )
