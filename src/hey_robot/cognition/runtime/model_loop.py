from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

from hey_robot.cognition.tools.registry import ToolRegistry
from hey_robot.providers import (
    ReasoningMessage,
    ReasoningProvider,
    ReasoningResponse,
)

logger = logging.getLogger("hey_robot.cognition.model_loop")


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
        tool_names = [str(tool.get("name") or "") for tool in tools]
        logger.info(
            "model request: messages=%s tools=%s allowed_tools=%s tool_choice=auto",
            len(messages),
            tool_names,
            sorted(allowed_tools) if allowed_tools is not None else None,
        )
        kwargs = {
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        }
        if callable(chat_with_retry):
            response = cast(ReasoningResponse, await chat_with_retry(**kwargs))
        else:
            response = await self.provider.chat(
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
        logger.info(
            "model response: finish_reason=%s error_kind=%s tool_calls=%s content_len=%s",
            response.finish_reason,
            getattr(response, "error_kind", None),
            [call.name for call in response.tool_calls],
            len(response.content or ""),
        )
        return response
