from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from hey_robot.agents.runtime.registry import ToolSpec


@dataclass(frozen=True)
class ToolHookContext:
    tool_call_id: str
    tool: ToolSpec
    arguments: dict


class ToolHook(Protocol):
    async def before_tool(self, context: ToolHookContext) -> None: ...

    async def after_tool(self, context: ToolHookContext, result: str) -> None: ...

    async def on_tool_error(
        self, context: ToolHookContext, error: Exception
    ) -> None: ...


@dataclass(slots=True)
class AgentRuntimeHookContext:
    iteration: int
    messages: list[Any]
    response: Any | None = None
    tool_results: list[Any] = field(default_factory=list)
    final_result: Any | None = None
    turn_trace: dict[str, Any] | None = None


class AgentRuntimeHook:
    async def before_iteration(self, _context: AgentRuntimeHookContext) -> None:
        return None

    async def after_model_response(self, _context: AgentRuntimeHookContext) -> None:
        return None

    async def before_execute_tools(self, _context: AgentRuntimeHookContext) -> None:
        return None

    async def after_tool_results(self, _context: AgentRuntimeHookContext) -> None:
        return None

    async def after_iteration(self, _context: AgentRuntimeHookContext) -> None:
        return None
