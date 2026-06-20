from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any

from hey_robot.providers.types import (
    ReasoningMessage,
    ReasoningProvider,
    ReasoningResponse,
)


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float = 0.1
    max_tokens: int = 2048
    reasoning_effort: str | None = None


class BaseReasoningProvider(ReasoningProvider):
    _retry_delays = (1.0, 2.0, 4.0)
    _retryable_status_codes = frozenset({408, 409, 429})
    _retryable_error_kinds = frozenset({"timeout", "connection"})
    _transient_markers = (
        "rate limit",
        "too many requests",
        "overloaded",
        "timeout",
        "timed out",
        "connection",
        "server error",
        "temporarily unavailable",
    )

    def __init__(self, *, generation: GenerationSettings | None = None) -> None:
        self.generation = generation or GenerationSettings()

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
        del (
            messages,
            tools,
            model,
            max_tokens,
            temperature,
            reasoning_effort,
            tool_choice,
        )
        raise NotImplementedError

    async def chat_with_retry(
        self,
        *,
        messages: list[ReasoningMessage],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        retry_mode: str = "standard",
        on_retry_wait: Callable[[str], Awaitable[None]] | None = None,
    ) -> ReasoningResponse:
        attempt = 0
        last: ReasoningResponse | None = None
        while True:
            attempt += 1
            try:
                response = await self.chat(
                    messages=messages,
                    tools=tools,
                    model=model,
                    max_tokens=self.generation.max_tokens
                    if max_tokens is None
                    else max_tokens,
                    temperature=self.generation.temperature
                    if temperature is None
                    else temperature,
                    reasoning_effort=(
                        self.generation.reasoning_effort
                        if reasoning_effort is None
                        else reasoning_effort
                    ),
                    tool_choice=tool_choice,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                response = ReasoningResponse(
                    content=f"Error calling model provider: {type(exc).__name__}: {exc}",
                    finish_reason="error",
                    error_kind="exception",
                )
            if response.finish_reason != "error":
                return response
            last = response
            if not self._is_transient_response(response):
                return response
            if retry_mode != "persistent" and attempt > len(self._retry_delays):
                return response
            delay = response.error_retry_after_s or self._extract_retry_after(
                response.content
            )
            if delay is None:
                delay = self._retry_delays[
                    min(attempt - 1, len(self._retry_delays) - 1)
                ]
            if on_retry_wait is not None:
                await on_retry_wait(f"model provider retry in {delay:.0f}s")
            await asyncio.sleep(max(0.1, delay))

        return last or ReasoningResponse(
            content="model provider failed", finish_reason="error"
        )

    @classmethod
    def _is_transient_response(cls, response: ReasoningResponse) -> bool:
        if response.error_should_retry is not None:
            return bool(response.error_should_retry)
        if response.error_status_code is not None:
            status = int(response.error_status_code)
            if status in cls._retryable_status_codes or status >= 500:
                return True
        if (response.error_kind or "").lower() in cls._retryable_error_kinds:
            return True
        text = (response.content or "").lower()
        return any(marker in text for marker in cls._transient_markers)

    @staticmethod
    def _extract_retry_after(content: str | None) -> float | None:
        text = (content or "").strip()
        if not text:
            return None
        match = re.search(r"retry after\s+(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
        try:
            retry_at = parsedate_to_datetime(text)
        except Exception:
            return None
        return max(0.1, (retry_at - retry_at.now(retry_at.tzinfo)).total_seconds())


class UnconfiguredReasoningProvider(BaseReasoningProvider):
    """Strict placeholder provider for code paths that should never plan."""

    def __init__(self, message: str) -> None:
        super().__init__(generation=GenerationSettings(temperature=0.0, max_tokens=64))
        self._message = message

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
        del (
            messages,
            tools,
            model,
            max_tokens,
            temperature,
            reasoning_effort,
            tool_choice,
        )
        return ReasoningResponse(
            content=self._message,
            finish_reason="error",
            error_kind="configuration",
            error_should_retry=False,
        )

    def get_default_model(self) -> str:
        return "unconfigured"
