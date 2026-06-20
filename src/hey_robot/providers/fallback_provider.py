from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from hey_robot.providers.base import BaseReasoningProvider
from hey_robot.providers.types import (
    ReasoningMessage,
    ReasoningProvider,
    ReasoningResponse,
)


@dataclass(frozen=True)
class FallbackCandidate:
    provider: ReasoningProvider
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    reasoning_effort: str | None = None


class FallbackReasoningProvider(BaseReasoningProvider):
    _failure_threshold = 3
    _cooldown_sec = 60.0

    def __init__(
        self, primary: ReasoningProvider, fallbacks: list[FallbackCandidate]
    ) -> None:
        super().__init__()
        self.primary = primary
        self.fallbacks = list(fallbacks)
        self._primary_failures = 0
        self._primary_tripped_at: float | None = None

    async def chat(
        self,
        *,
        messages: list[ReasoningMessage],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ReasoningResponse:
        if self._primary_available():
            response = await self.primary.chat(
                messages=messages,
                tools=tools,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                tool_choice=tool_choice,
            )
            if response.finish_reason != "error" or not self._should_fallback(response):
                self._primary_failures = (
                    0 if response.finish_reason != "error" else self._primary_failures
                )
                return response
            self._primary_failures += 1
            if self._primary_failures >= self._failure_threshold:
                self._primary_tripped_at = time.monotonic()

        last: ReasoningResponse | None = None
        for fallback in self.fallbacks:
            response = await fallback.provider.chat(
                messages=messages,
                tools=tools,
                model=fallback.model,
                max_tokens=fallback.max_tokens
                if fallback.max_tokens is not None
                else max_tokens,
                temperature=fallback.temperature
                if fallback.temperature is not None
                else temperature,
                reasoning_effort=(
                    fallback.reasoning_effort
                    if fallback.reasoning_effort is not None
                    else reasoning_effort
                ),
                tool_choice=tool_choice,
            )
            if response.finish_reason != "error":
                return response
            last = response
        return last or ReasoningResponse(
            content="all model providers failed", finish_reason="error"
        )

    def get_default_model(self) -> str:
        return self.primary.get_default_model()

    def _primary_available(self) -> bool:
        if self._primary_tripped_at is None:
            return True
        return time.monotonic() - self._primary_tripped_at >= self._cooldown_sec

    @staticmethod
    def _should_fallback(response: ReasoningResponse) -> bool:
        if response.error_should_retry is False:
            return False
        if response.error_status_code is not None:
            return (
                response.error_status_code in {408, 409, 429}
                or response.error_status_code >= 500
            )
        text = (response.content or "").lower()
        return any(
            token in text
            for token in ("rate limit", "timeout", "overloaded", "server error")
        )
