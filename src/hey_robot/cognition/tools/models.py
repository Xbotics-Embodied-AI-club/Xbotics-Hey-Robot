"""Typed contracts shared by Agent-facing tools."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from hey_robot.protocol import ToolOutcome
from hey_robot.tool_schema import validate_arguments


@dataclass(frozen=True)
class ToolSpec:
    """One model-visible function without execution ownership."""

    name: str
    description: str
    parameters: dict[str, Any]

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }


@dataclass(frozen=True)
class PhysicalToolCall:
    """Validated call that must cross the durable physical boundary."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AgentResponseCall:
    """Validated user response with an atomic sustained-task state."""

    task_state: Literal["none", "wait", "complete", "cancel"]
    message: str


HarnessToolHandler = Callable[
    [dict[str, Any]],
    ToolOutcome | Awaitable[ToolOutcome],
]


@dataclass(frozen=True)
class HarnessToolCall:
    """A validated non-physical call bound to a constrained handler."""

    name: str
    arguments: dict[str, Any]
    handler: HarnessToolHandler

    async def execute(self) -> ToolOutcome:
        result = self.handler(dict(self.arguments))
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, ToolOutcome):
            raise TypeError(
                f"Harness Tool {self.name!r} returned unsupported result: "
                f"{type(result)!r}"
            )
        return result


PreparedToolCall = AgentResponseCall | HarnessToolCall | PhysicalToolCall


class AgentTool(Protocol):
    """Minimal contract implemented by every model-visible tool."""

    name: str

    @property
    def spec(self) -> ToolSpec: ...

    @property
    def schema(self) -> dict[str, Any]: ...

    def prepare(self, arguments: dict[str, Any]) -> PreparedToolCall: ...


class HarnessTool:
    """A non-physical Tool with an independently owned schema and handler."""

    def __init__(self, spec: ToolSpec, handler: HarnessToolHandler) -> None:
        if not spec.name:
            raise ValueError("Harness Tool name must be non-empty")
        self.spec = spec
        self.name = spec.name
        self._handler = handler

    @property
    def schema(self) -> dict[str, Any]:
        return self.spec.definition

    def prepare(self, arguments: dict[str, Any]) -> HarnessToolCall:
        normalized = validate_arguments(self.spec.parameters, arguments)
        return HarnessToolCall(self.name, normalized, self._handler)
