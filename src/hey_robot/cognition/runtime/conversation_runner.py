"""Bounded, nanobot-style tool loop for one user conversation turn."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Protocol

from hey_robot.protocol import ActionProposal, ToolOutcome
from hey_robot.providers import ReasoningMessage, ReasoningProvider, ReasoningToolCall

ExecuteProposal = Callable[[ActionProposal], Awaitable[ToolOutcome | dict[str, object]]]


class ToolRegistry(Protocol):
    @property
    def definitions(self) -> list[dict[str, object]]: ...

    @property
    def instructions(self) -> str: ...

    def proposal(self, name: str, arguments: dict[str, object]) -> ActionProposal: ...


class ConversationToolRunner:
    """Let the model reason over tool results until it returns a final answer.

    Tool calls are executed serially.  This preserves the normal Agent Loop
    while ensuring physical operations do not race each other.
    """

    def __init__(
        self,
        provider: ReasoningProvider,
        tools: ToolRegistry,
        *,
        max_tool_rounds: int = 6,
    ) -> None:
        self._provider = provider
        self._tools = tools
        self._max_tool_rounds = max(1, max_tool_rounds)

    async def run(
        self, messages: list[ReasoningMessage], execute: ExecuteProposal
    ) -> str:
        transcript = list(messages)
        last_outcome: dict[str, object] = {}
        for _ in range(self._max_tool_rounds):
            response = await self._provider.chat(
                messages=transcript, tools=self._tools.definitions
            )
            if not response.tool_calls:
                return (response.content or _fallback(last_outcome)).strip()
            transcript.append(
                ReasoningMessage(
                    role="assistant",
                    content=response.content or "",
                    tool_calls=[
                        ReasoningToolCall(
                            call.id,
                            call.name,
                            dict(call.arguments),
                            call.provider_metadata,
                        )
                        for call in response.tool_calls
                    ],
                )
            )
            for call in response.tool_calls:
                outcome = await self._execute_call(call, execute)
                last_outcome = outcome
                transcript.append(
                    ReasoningMessage(
                        role="tool",
                        tool_call_id=call.id,
                        tool_name=call.name,
                        content=json.dumps(outcome, ensure_ascii=False),
                    )
                )
        return _fallback(last_outcome) or "已达到本轮工具调用上限，请说明下一步。"

    async def _execute_call(
        self, call: ReasoningToolCall, execute: ExecuteProposal
    ) -> dict[str, object]:
        try:
            proposal = self._tools.proposal(call.name, dict(call.arguments))
        except (KeyError, TypeError, ValueError) as exc:
            return {
                "status": "failed",
                "user_summary": f"工具调用参数无效：{exc}",
                "retryable": True,
            }
        return _outcome_payload(await execute(proposal))


def _outcome_payload(raw: ToolOutcome | dict[str, object]) -> dict[str, object]:
    if isinstance(raw, ToolOutcome):
        return {
            "status": raw.status,
            "user_summary": raw.user_summary,
            "data": raw.data,
            "operation_id": raw.operation_id,
            "goal_id": raw.goal_id,
            "retryable": raw.retryable,
        }
    return raw


def _fallback(outcome: dict[str, object]) -> str:
    summary = outcome.get("user_summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    status = outcome.get("status")
    if status in {"accepted", "waiting"}:
        return "请求已接收，正在等待机器人执行结果。"
    if status == "failed":
        return "这次操作没有完成。"
    return ""
