"""One model request, one proposal at most, and no physical IO."""

from __future__ import annotations

import asyncio
import time
from typing import cast

from hey_robot.cognition.runtime.result import (
    AgentRunRequest,
    AgentRunResult,
    ToolCallRecord,
)
from hey_robot.cognition.tools.autonomous import AutonomousToolRegistry
from hey_robot.protocol import FailurePayload
from hey_robot.providers import ReasoningMessage, ReasoningProvider


class StrictAgentRunner:
    def __init__(
        self, provider: ReasoningProvider, tools: AutonomousToolRegistry
    ) -> None:
        self._provider = provider
        self._tools = tools

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        if request.allowed_tools != self._tools.names:
            return self._failure(
                "CONTEXT_BUILD",
                "INVALID_TOOL_SET",
                "allowed tools differ from the autonomous surface",
            )
        if time.monotonic() >= request.deadline:
            return self._failure(
                "MODEL_REQUEST", "PROVIDER_TIMEOUT", "deliberation deadline elapsed"
            )
        raw = list(request.messages)
        if not all(isinstance(message, ReasoningMessage) for message in raw):
            return self._failure(
                "CONTEXT_BUILD", "INVALID_MODEL_RESPONSE", "invalid reasoning message"
            )
        messages = cast(list[ReasoningMessage], raw)
        try:
            response = await asyncio.wait_for(
                self._provider.chat(messages=messages, tools=self._tools.definitions),
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
        if response.tool_calls and len(response.tool_calls) != 1:
            return self._failure(
                "MODEL_PROTOCOL",
                "MULTIPLE_ACTION_PROPOSALS",
                "model returned multiple tool calls",
            )
        if response.tool_calls:
            call = response.tool_calls[0]
            record = ToolCallRecord(call.id, call.name, dict(call.arguments))
            if call.name not in request.allowed_tools:
                return self._failure(
                    "TOOL_VALIDATION", "UNKNOWN_TOOL", call.name, (record,)
                )
            try:
                proposal = self._tools.proposal(call.name, dict(call.arguments))
            except (KeyError, ValueError, TypeError) as exc:
                return self._failure(
                    "TOOL_VALIDATION", "INVALID_TOOL_ARGUMENTS", str(exc), (record,)
                )
            return AgentRunResult(
                "action_proposed",
                None,
                "stop_slice",
                (record,),
                proposal,
                usage=dict(response.usage),
            )
        if not (response.content or "").strip():
            return self._failure(
                "MODEL_PROTOCOL",
                "EMPTY_MODEL_RESPONSE",
                "provider returned neither text nor a tool call",
            )
        return AgentRunResult(
            "returned", response.content, "model_returned", usage=dict(response.usage)
        )

    @staticmethod
    def _failure(
        stage: str, code: str, message: str, calls: tuple[ToolCallRecord, ...] = ()
    ) -> AgentRunResult:
        return AgentRunResult(
            "failed",
            None,
            code,
            calls,
            failure=FailurePayload(stage, code, "StrictAgentRunner", message),
        )
