from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy.typing as npt

ModelRole = Literal["system", "user", "assistant", "tool"]
TextDeltaCallback = Callable[[str], Awaitable[None]]


@dataclass(frozen=True)
class ModelImage:
    data: npt.NDArray
    media_type: str = "image/jpeg"
    detail: str = "high"
    name: str | None = None


@dataclass(frozen=True)
class ModelMessage:
    role: ModelRole
    content: str
    images: list[ModelImage] = field(default_factory=list)
    tool_call_id: str | None = None
    tool_calls: list[ModelToolCall] = field(default_factory=list)


@dataclass(frozen=True)
class ModelToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResponse:
    content: str | None = None
    tool_calls: list[ModelToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    error_kind: str | None = None


class ModelClientLike(Protocol):
    async def chat(
        self,
        *,
        messages: list[ModelMessage],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_text_delta: TextDeltaCallback | None = None,
    ) -> ModelResponse: ...
