from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from hey_robot.cognition.tools.registry import ToolRegistry
from hey_robot.providers import (
    ReasoningMessage,
    ReasoningProvider,
    ReasoningResponse,
)


@dataclass
class ModelLoop:
    """Builds provider requests for a tool-using reasoning turn."""

    provider: ReasoningProvider
    tools: ToolRegistry

    async def request(
        self,
        messages: list[ReasoningMessage],
        *,
        allowed_tools: set[str] | None = None,
    ) -> ReasoningResponse:
        chat_with_retry = getattr(self.provider, "chat_with_retry", None)
        tools = self.tools.list_tools()
        if allowed_tools is not None:
            tools = [tool for tool in tools if tool["name"] in allowed_tools]
        kwargs = {
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        }
        if callable(chat_with_retry):
            return cast(ReasoningResponse, await chat_with_retry(**kwargs))
        return await self.provider.chat(
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
