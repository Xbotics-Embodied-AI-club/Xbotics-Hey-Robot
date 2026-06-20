from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from hey_robot.audio.config import VoiceActivationConfig

_LEADING_TRAILING_PUNCTUATION = " \t\r\n，,。.!！?？:：；;"

_EMERGENCY_STOP_PHRASES = frozenset(
    {
        "停止",
        "停下",
        "停",
        "别动",
        "别走",
        "快停",
        "停下来",
        "别跟着",
        "stop",
        "halt",
        "freeze",
    }
)


def _is_emergency_stop(text: str) -> bool:
    stripped = text.strip(_LEADING_TRAILING_PUNCTUATION)
    return any(phrase in stripped for phrase in _EMERGENCY_STOP_PHRASES)


@dataclass(frozen=True)
class VoiceRouteDecision:
    accepted: bool
    text: str
    reason: str
    session_active: bool
    wake_word: str = ""


class VoiceSessionRouter:
    """Routes ASR text through wake-word activation and a short voice session."""

    def __init__(
        self,
        config: VoiceActivationConfig,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.config = config
        self._clock = clock or time.monotonic
        self._active_until = 0.0

    def route(self, raw_text: str) -> VoiceRouteDecision:
        text = str(raw_text or "").strip()
        if not text:
            return self._drop("empty", text="")

        if not self.config.enabled or not self.config.wake_words:
            return self._accept(text, reason="activation_disabled")

        wake_match = _match_wake_prefix(text, self.config.wake_words)
        if wake_match is not None:
            wake_word, span = wake_match
            routed_text = text
            if self.config.strip_wake_word:
                routed_text = text[span:].strip(_LEADING_TRAILING_PUNCTUATION)
            self._extend_session()
            if not routed_text:
                return VoiceRouteDecision(
                    accepted=False,
                    text="",
                    reason="activated_without_command",
                    session_active=True,
                    wake_word=wake_word,
                )
            if len(routed_text) < self.config.min_route_chars:
                return VoiceRouteDecision(
                    accepted=False,
                    text=routed_text,
                    reason="too_short_after_wake",
                    session_active=True,
                    wake_word=wake_word,
                )
            return self._accept(
                routed_text,
                reason="wake_word",
                wake_word=wake_word,
            )

        if self._is_active():
            if len(text) < self.config.min_route_chars:
                return self._drop("too_short_in_session", text=text)
            self._extend_session()
            return self._accept(text, reason="active_session")

        if _is_emergency_stop(text):
            return self._accept(text, reason="emergency_phrase")

        return self._drop("wake_word_required", text=text)

    def _accept(
        self, text: str, *, reason: str, wake_word: str = ""
    ) -> VoiceRouteDecision:
        return VoiceRouteDecision(
            accepted=True,
            text=text,
            reason=reason,
            session_active=self._is_active(),
            wake_word=wake_word,
        )

    def _drop(self, reason: str, *, text: str) -> VoiceRouteDecision:
        return VoiceRouteDecision(
            accepted=False,
            text=text,
            reason=reason,
            session_active=self._is_active(),
        )

    def _extend_session(self) -> None:
        timeout = max(0.0, float(self.config.session_timeout_sec))
        self._active_until = self._clock() + timeout

    def _is_active(self) -> bool:
        return self._clock() < self._active_until


def _match_wake_prefix(text: str, wake_words: list[str]) -> tuple[str, int] | None:
    normalized = text.strip()
    for wake_word in wake_words:
        word = str(wake_word or "").strip()
        if not word:
            continue
        if normalized.startswith(word):
            return word, len(word)
        span = _match_repeated_first_char_prefix(normalized, word)
        if span is not None:
            return word, span
    return None


def _match_repeated_first_char_prefix(text: str, wake_word: str) -> int | None:
    if len(wake_word) < 2 or not text.startswith(wake_word[0] * 2):
        return None
    collapsed = wake_word[0] + text[2:]
    if collapsed.startswith(wake_word):
        return len(wake_word) + 1
    return None
