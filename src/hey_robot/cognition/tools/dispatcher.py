"""Small typed-call router used by the existing Agent loop."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from hey_robot.cognition.tools.models import PreparedToolCall

DispatchHandler = Callable[[PreparedToolCall, Any], Any | Awaitable[Any]]


class ToolDispatcher:
    """Route a prepared call by its exact type; own no Agent or Skill state."""

    def __init__(self, handlers: Mapping[type[object], DispatchHandler]) -> None:
        self._handlers = dict(handlers)

    async def dispatch(self, call: PreparedToolCall, context: Any) -> Any:
        handler = self._handlers.get(type(call))
        if handler is None:
            raise TypeError(f"unsupported prepared Tool call: {type(call)!r}")
        result = handler(call, context)
        if inspect.isawaitable(result):
            return await result
        return result
