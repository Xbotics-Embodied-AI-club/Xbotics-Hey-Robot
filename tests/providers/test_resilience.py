from __future__ import annotations

import asyncio
import time
from typing import cast

from hey_robot.providers.base import BaseReasoningProvider, GenerationSettings
from hey_robot.providers.fallback_provider import (
    FallbackCandidate,
    FallbackReasoningProvider,
)
from hey_robot.providers.registry import find_provider
from hey_robot.providers.types import ReasoningMessage, ReasoningResponse


class SequencedProvider(BaseReasoningProvider):
    def __init__(
        self,
        responses: list[ReasoningResponse | Exception],
        *,
        generation: GenerationSettings | None = None,
        model: str = "test-model",
    ) -> None:
        super().__init__(generation=generation)
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []
        self.model = model

    async def chat(self, **kwargs) -> ReasoningResponse:  # type: ignore[override]
        self.calls.append(kwargs)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return cast(ReasoningResponse, item)

    def get_default_model(self) -> str:
        return self.model


def test_base_provider_chat_with_retry_retries_transient_response(monkeypatch) -> None:
    provider = SequencedProvider(
        [
            ReasoningResponse(
                content="retry after 0.2",
                finish_reason="error",
                error_status_code=429,
            ),
            ReasoningResponse(content="ok", finish_reason="stop"),
        ],
        generation=GenerationSettings(
            temperature=0.4, max_tokens=111, reasoning_effort="medium"
        ),
    )
    waits: list[str] = []
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    async def on_retry_wait(message: str) -> None:
        waits.append(message)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    response = asyncio.run(
        provider.chat_with_retry(
            messages=[ReasoningMessage(role="user", content="hello")],
            retry_mode="standard",
            on_retry_wait=on_retry_wait,
        )
    )

    assert response.content == "ok"
    assert len(provider.calls) == 2
    assert provider.calls[0]["max_tokens"] == 111
    assert provider.calls[0]["temperature"] == 0.4
    assert provider.calls[0]["reasoning_effort"] == "medium"
    assert waits == ["model provider retry in 0s"]
    assert delays == [0.2]


def test_base_provider_chat_with_retry_returns_non_transient_errors_and_parses_retry_after() -> (
    None
):
    provider = SequencedProvider([ValueError("bad request")])

    response = asyncio.run(
        provider.chat_with_retry(
            messages=[ReasoningMessage(role="user", content="hello")]
        )
    )

    assert response.finish_reason == "error"
    assert response.error_kind == "exception"
    assert "ValueError" in (response.content or "")
    assert provider._extract_retry_after("Retry After 3.5") == 3.5
    assert provider._extract_retry_after("not-a-date") is None


def test_fallback_provider_uses_fallback_after_primary_failures() -> None:
    primary = SequencedProvider(
        [
            ReasoningResponse(
                content="rate limit", finish_reason="error", error_status_code=429
            ),
            ReasoningResponse(
                content="overloaded", finish_reason="error", error_status_code=503
            ),
            ReasoningResponse(
                content="timeout", finish_reason="error", error_kind="timeout"
            ),
        ],
        model="primary-model",
    )
    fallback = SequencedProvider(
        [
            ReasoningResponse(content="fallback-1", finish_reason="stop"),
            ReasoningResponse(content="fallback-2", finish_reason="stop"),
            ReasoningResponse(content="fallback-3", finish_reason="stop"),
        ],
        model="fb-model",
    )
    provider = FallbackReasoningProvider(
        primary, [FallbackCandidate(provider=fallback, max_tokens=55)]
    )

    first = asyncio.run(
        provider.chat(messages=[ReasoningMessage(role="user", content="hi")])
    )
    second = asyncio.run(
        provider.chat(messages=[ReasoningMessage(role="user", content="again")])
    )
    response = asyncio.run(
        provider.chat(messages=[ReasoningMessage(role="user", content="third")])
    )

    assert first.content == "fallback-1"
    assert second.content == "fallback-2"
    assert response.content == "fallback-3"
    assert provider.get_default_model() == "primary-model"
    assert provider._primary_tripped_at is not None

    fallback._responses.append(
        ReasoningResponse(content="cooldown fallback", finish_reason="stop")
    )
    during_cooldown = asyncio.run(
        provider.chat(messages=[ReasoningMessage(role="user", content="cooldown")])
    )

    assert during_cooldown.content == "cooldown fallback"
    assert len(primary.calls) == 3

    provider._primary_tripped_at = time.monotonic() - provider._cooldown_sec - 1
    primary._responses.append(
        ReasoningResponse(content="primary restored", finish_reason="stop")
    )
    restored = asyncio.run(
        provider.chat(messages=[ReasoningMessage(role="user", content="restored")])
    )

    assert restored.content == "primary restored"

    no_fallback = FallbackReasoningProvider(
        SequencedProvider(
            [
                ReasoningResponse(
                    content="validation failed",
                    finish_reason="error",
                    error_should_retry=False,
                )
            ]
        ),
        [
            FallbackCandidate(
                provider=SequencedProvider(
                    [ReasoningResponse(content="unused", finish_reason="stop")]
                )
            )
        ],
    )
    direct = asyncio.run(
        no_fallback.chat(messages=[ReasoningMessage(role="user", content="hi")])
    )

    assert direct.content == "validation failed"


def test_find_provider_resolves_by_name_api_base_and_model() -> None:
    assert find_provider(name="qwen").name == "dashscope"
    assert (
        find_provider(api_base="https://openrouter.ai/api/v1/chat").name == "openrouter"
    )
    assert find_provider(model="deepseek-chat").name == "deepseek"
    assert find_provider(name="openai-compatible").name == "custom"
    assert find_provider().name == "openai"
