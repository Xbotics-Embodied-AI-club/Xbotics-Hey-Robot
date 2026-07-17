"""由对话和 Goal 执行共享的一次受限模型决策。"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Literal, Protocol

from hey_robot.protocol import FailurePayload
from hey_robot.providers import ReasoningMessage, ReasoningProvider


class ToolRegistryLike(Protocol):
    """纯模型工具边界；实现不得执行 IO。"""

    @property
    def definitions(self) -> list[dict[str, object]]: ...

    def proposal(self, name: str, arguments: dict[str, object]) -> object: ...


@dataclass(frozen=True)
class AgentTurnRequest:
    messages: tuple[ReasoningMessage, ...]
    allowed_tools: frozenset[str]
    deadline: float
    run_id: str


@dataclass(frozen=True)
class AgentToolCallRecord:
    tool_call_id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class AgentTurnResult:
    status: Literal["returned", "action_proposed", "failed"]
    final_text: str | None
    stop_reason: str
    tool_calls: tuple[AgentToolCallRecord, ...] = ()
    proposal: object | None = None
    failure: FailurePayload | None = None
    usage: dict[str, int] = field(default_factory=dict)


class AgentRunner:
    """返回文本或一个带类型的提案；绝不执行外部 IO。"""

    def __init__(self, provider: ReasoningProvider, tools: ToolRegistryLike) -> None:
        self._provider = provider
        self._tools = tools

    async def run(self, request: AgentTurnRequest) -> AgentTurnResult:
        if not request.messages or any(
            not isinstance(message, ReasoningMessage) for message in request.messages
        ):
            return self._failure(
                "CONTEXT_BUILD",
                "INVALID_MODEL_MESSAGES",
                "messages must contain ReasoningMessage values",
            )
        definitions = self._allowed_definitions(request.allowed_tools)
        if definitions is None:
            return self._failure(
                "CONTEXT_BUILD",
                "INVALID_TOOL_SET",
                "allowed tools are not present in the configured registry",
            )
        if time.monotonic() >= request.deadline:
            return self._failure(
                "MODEL_REQUEST", "PROVIDER_TIMEOUT", "decision deadline elapsed"
            )
        try:
            response = await asyncio.wait_for(
                self._provider.chat(messages=list(request.messages), tools=definitions),
                timeout=max(0.001, request.deadline - time.monotonic()),
            )
        except TimeoutError:
            return self._failure(
                "MODEL_REQUEST", "PROVIDER_TIMEOUT", "provider request timed out"
            )
        except Exception as exc:
            return self._failure("MODEL_REQUEST", "PROVIDER_ERROR", str(exc))

        if response.finish_reason == "error":
            return self._failure(
                "MODEL_REQUEST",
                "PROVIDER_ERROR",
                response.content or "provider returned an error",
            )
        if len(response.tool_calls) > 1:
            return self._failure(
                "MODEL_PROTOCOL",
                "MULTIPLE_ACTION_PROPOSALS",
                "model returned multiple tool calls",
            )
        if response.tool_calls:
            call = response.tool_calls[0]
            record = AgentToolCallRecord(call.id, call.name, dict(call.arguments))
            if call.name not in request.allowed_tools:
                return self._failure(
                    "TOOL_VALIDATION", "UNKNOWN_TOOL", call.name, (record,)
                )
            try:
                proposal = self._tools.proposal(call.name, dict(call.arguments))
            except (KeyError, TypeError, ValueError) as exc:
                return self._failure(
                    "TOOL_VALIDATION", "INVALID_TOOL_ARGUMENTS", str(exc), (record,)
                )
            return AgentTurnResult(
                "action_proposed",
                None,
                "stop_slice",
                (record,),
                proposal,
                usage=dict(response.usage),
            )

        text = (response.content or "").strip()
        if not text:
            return self._failure(
                "MODEL_PROTOCOL",
                "EMPTY_MODEL_RESPONSE",
                "provider returned neither text nor a tool call",
            )
        return AgentTurnResult(
            "returned", text, "model_returned", usage=dict(response.usage)
        )

    def _allowed_definitions(
        self, allowed_tools: frozenset[str]
    ) -> list[dict[str, object]] | None:
        by_name: dict[str, dict[str, object]] = {}
        for definition in self._tools.definitions:
            function = definition.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if isinstance(name, str):
                by_name[name] = definition
        if not allowed_tools.issubset(by_name):
            return None
        return [
            definition for name, definition in by_name.items() if name in allowed_tools
        ]

    @staticmethod
    def _failure(
        stage: str,
        code: str,
        message: str,
        calls: tuple[AgentToolCallRecord, ...] = (),
    ) -> AgentTurnResult:
        return AgentTurnResult(
            "failed",
            None,
            code,
            calls,
            failure=FailurePayload(stage, code, "AgentRunner", message),
        )
