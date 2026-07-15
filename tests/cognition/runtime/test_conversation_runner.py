from __future__ import annotations

import asyncio

from hey_robot.cognition.runtime.conversation_runner import ConversationToolRunner
from hey_robot.cognition.tools.autonomous import (
    AgentToolDependencies,
    build_agent_tools,
)
from hey_robot.protocol import ToolOutcome
from hey_robot.providers import ReasoningMessage, ReasoningResponse, ReasoningToolCall
from hey_robot.skill_os.base import SkillCatalog


class _Provider:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return ReasoningResponse(
                tool_calls=[
                    ReasoningToolCall(
                        "call-1", "request_observation", {"question": "前方有什么？"}
                    )
                ]
            )
        return ReasoningResponse(content="前方是一张桌子。")


def test_tool_result_is_returned_to_same_conversation_loop() -> None:
    provider = _Provider()
    tools = build_agent_tools(AgentToolDependencies(SkillCatalog(())))
    runner = ConversationToolRunner(provider, tools)

    async def execute(_proposal):
        return ToolOutcome("completed", "前方是一张桌子。")

    text = asyncio.run(
        runner.run([ReasoningMessage(role="user", content="你看到了什么？")], execute)
    )

    assert text == "前方是一张桌子。"
    assert provider.calls[1]["tools"] == tools.definitions
    assert provider.calls[1]["messages"][-1].role == "tool"


def test_accepted_outcome_is_not_presented_as_completion() -> None:
    provider = _Provider()
    tools = build_agent_tools(AgentToolDependencies(SkillCatalog(())))
    runner = ConversationToolRunner(provider, tools)

    async def execute(_proposal):
        return ToolOutcome("accepted")

    async def chat(**kwargs):
        provider.calls.append(kwargs)
        if len(provider.calls) == 1:
            return ReasoningResponse(
                tool_calls=[
                    ReasoningToolCall(
                        "call-1", "request_observation", {"question": "前方有什么？"}
                    )
                ]
            )
        return ReasoningResponse()

    provider.chat = chat
    text = asyncio.run(
        runner.run([ReasoningMessage(role="user", content="看一下")], execute)
    )

    assert "已接收" in text
