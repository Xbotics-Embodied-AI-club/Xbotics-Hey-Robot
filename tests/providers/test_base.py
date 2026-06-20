from __future__ import annotations

import asyncio
from datetime import UTC

import pytest

from hey_robot.providers.base import BaseReasoningProvider, GenerationSettings
from hey_robot.providers.types import ReasoningResponse


class FakeProvider(BaseReasoningProvider):
    """Concrete provider for testing BaseReasoningProvider."""

    def __init__(
        self,
        responses: list[ReasoningResponse],
        *,
        generation: GenerationSettings | None = None,
    ) -> None:
        super().__init__(generation=generation)
        self.responses = responses
        self.call_count = 0

    def get_default_model(self) -> str:
        return "fake-model"

    async def chat(self, **kwargs: object) -> ReasoningResponse:  # noqa: ARG002
        self.call_count += 1
        if not self.responses:
            return ReasoningResponse(content="fallback", finish_reason="stop")
        return self.responses.pop(0)


def make_error(content: str = "server error", **kwargs: object) -> ReasoningResponse:
    return ReasoningResponse(content=content, finish_reason="error", **kwargs)  # type: ignore[arg-type]


def make_ok(content: str = "OK") -> ReasoningResponse:
    return ReasoningResponse(content=content, finish_reason="stop")


class TestBaseReasoningProvider:
    async def test_chat_raises_not_implemented(self) -> None:
        class ChatOnlyProvider(BaseReasoningProvider):
            def get_default_model(self) -> str:
                return "test"

        provider = ChatOnlyProvider()
        with pytest.raises(NotImplementedError):
            await provider.chat(messages=[])

    async def test_retry_succeeds_after_transient_error(self) -> None:
        provider = FakeProvider(
            [
                make_error("overloaded, retry after 0.1"),
                make_ok("success"),
            ]
        )
        response = await provider.chat_with_retry(messages=[])
        assert response.content == "success"
        assert provider.call_count == 2

    async def test_retry_returns_error_if_not_transient(self) -> None:
        provider = FakeProvider(
            [
                make_error("invalid api key"),
            ]
        )
        response = await provider.chat_with_retry(messages=[])
        assert response.finish_reason == "error"
        assert provider.call_count == 1

    async def test_retry_exhausts_standard_retries(self) -> None:
        provider = FakeProvider([make_error("timeout") for _ in range(10)])
        response = await provider.chat_with_retry(messages=[])
        assert response.finish_reason == "error"
        assert provider.call_count == 4  # 1 initial + 3 retries

    async def test_persistent_retry_keeps_trying(self) -> None:
        provider = FakeProvider(
            [make_error("timeout") for _ in range(5)] + [make_ok("finally")]
        )
        response = await provider.chat_with_retry(messages=[], retry_mode="persistent")
        assert response.content == "finally"
        assert provider.call_count == 6

    async def test_cancelled_error_propagates(self) -> None:
        provider = FakeProvider([])

        async def cancel_soon() -> None:
            await asyncio.sleep(0.01)
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await asyncio.gather(cancel_soon(), provider.chat_with_retry(messages=[]))

    async def test_exception_caught_as_error_response(self) -> None:
        class ExplodingProvider(FakeProvider):
            async def chat(self, **kwargs: object) -> ReasoningResponse:  # noqa: ARG002
                raise ValueError("boom")

        provider = ExplodingProvider([])
        response = await provider.chat_with_retry(messages=[])
        assert response.finish_reason == "error"
        assert response.content is not None
        assert "ValueError" in response.content

    async def test_retry_after_from_error_header(self) -> None:
        provider = FakeProvider(
            [
                make_error("overloaded", error_retry_after_s=0.05),
                make_ok("done"),
            ]
        )
        response = await provider.chat_with_retry(messages=[])
        assert response.content == "done"

    async def test_extract_retry_after_from_content(self) -> None:
        provider = FakeProvider(
            [
                make_error("rate limit exceeded, retry after 0.05"),
                make_ok("done"),
            ]
        )
        response = await provider.chat_with_retry(messages=[])
        assert response.content == "done"

    async def test_is_transient_with_status_code(self) -> None:
        resp = ReasoningResponse(
            content="err", finish_reason="error", error_status_code=429
        )
        assert BaseReasoningProvider._is_transient_response(resp) is True

    async def test_is_transient_with_500_status(self) -> None:
        resp = ReasoningResponse(
            content="err", finish_reason="error", error_status_code=502
        )
        assert BaseReasoningProvider._is_transient_response(resp) is True

    async def test_is_transient_with_error_kind(self) -> None:
        resp = ReasoningResponse(
            content="err", finish_reason="error", error_kind="timeout"
        )
        assert BaseReasoningProvider._is_transient_response(resp) is True

    async def test_is_transient_with_transient_marker(self) -> None:
        resp = ReasoningResponse(content="rate limit hit", finish_reason="error")
        assert BaseReasoningProvider._is_transient_response(resp) is True

    async def test_is_transient_with_error_should_retry_false(self) -> None:
        resp = ReasoningResponse(
            content="err", finish_reason="error", error_should_retry=False
        )
        assert BaseReasoningProvider._is_transient_response(resp) is False

    async def test_extract_retry_after_empty_content(self) -> None:
        assert BaseReasoningProvider._extract_retry_after("") is None
        assert BaseReasoningProvider._extract_retry_after(None) is None

    async def test_extract_retry_after_from_regex(self) -> None:
        delay = BaseReasoningProvider._extract_retry_after("Retry after 2.5 seconds")
        assert delay == 2.5

    async def test_extract_retry_after_from_http_date(self) -> None:
        import datetime

        future = datetime.datetime.now(UTC) + datetime.timedelta(seconds=5)
        delay = BaseReasoningProvider._extract_retry_after(
            future.strftime("%a, %d %b %Y %H:%M:%S GMT")
        )
        assert delay is not None
        assert 0.1 <= delay <= 10.0

    async def test_extract_retry_after_invalid_text(self) -> None:
        delay = BaseReasoningProvider._extract_retry_after("just some error message")
        assert delay is None


class TestGenerationSettings:
    def test_defaults(self) -> None:
        gs = GenerationSettings()
        assert gs.temperature == 0.1
        assert gs.max_tokens == 2048
        assert gs.reasoning_effort is None

    def test_custom(self) -> None:
        gs = GenerationSettings(
            temperature=0.5, max_tokens=1024, reasoning_effort="high"
        )
        assert gs.temperature == 0.5
        assert gs.max_tokens == 1024
        assert gs.reasoning_effort == "high"

    def test_provider_uses_defaults(self) -> None:
        provider = FakeProvider([])
        assert provider.generation.temperature == 0.1

    def test_provider_accepts_custom(self) -> None:
        gs = GenerationSettings(temperature=0.7)
        provider = FakeProvider([], generation=gs)
        assert provider.generation.temperature == 0.7
