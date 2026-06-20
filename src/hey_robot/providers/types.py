from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy.typing as npt

ReasoningRole = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ReasoningImage:
    data: npt.NDArray
    media_type: str = "image/jpeg"
    detail: str = "high"
    name: str | None = None


@dataclass(frozen=True)
class ReasoningMessage:
    role: ReasoningRole
    content: str
    images: list[ReasoningImage] = field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_calls: list[ReasoningToolCall] = field(default_factory=list)


@dataclass(frozen=True)
class ReasoningToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReasoningResponse:
    content: str | None = None
    tool_calls: list[ReasoningToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    reasoning_content: str | None = None
    error_status_code: int | None = None
    error_kind: str | None = None
    error_type: str | None = None
    error_code: str | None = None
    error_retry_after_s: float | None = None
    error_should_retry: bool | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def should_execute_tools(self) -> bool:
        return self.has_tool_calls and self.finish_reason in {"tool_calls", "stop"}


class ReasoningProvider(Protocol):
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
    ) -> ReasoningResponse: ...

    def get_default_model(self) -> str: ...
