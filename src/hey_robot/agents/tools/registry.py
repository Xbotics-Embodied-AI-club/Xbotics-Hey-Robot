"""Class-based tool registry with cast+validate pipeline.

The single :class:`ToolRegistry` for robot agent tools.  Stores
:class:`~hey_robot.agents.tools.base.Tool` instances and exposes
:class:`~hey_robot.agents.runtime.registry.ToolSpec` objects to the
capability / permission / execution pipeline.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Literal

from hey_robot.agents.tools.base import Tool

if TYPE_CHECKING:
    from hey_robot.agents.runtime.registry import ToolSpec


class ToolRegistry:
    """Registry for robot agent tools with schema-driven validation.

    Stores ``Tool`` instances and produces ``ToolSpec`` objects
    for downstream consumers.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._spec_cache: dict[str, ToolSpec] | None = None
        self._cached_definitions: list[dict[str, Any]] | None = None

    # Registration.

    def register(self, tool: Tool) -> None:
        self._validate_tool_definition(tool)
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool
        self._spec_cache = None
        self._cached_definitions = None

    def register_simple(
        self,
        name: str,
        func: Any,
        *,
        description: str | None = None,
        input_schema: dict[str, Any] | None = None,
        read_only: bool | None = None,
        destructive: bool = False,
        safety_level: str = "normal",
        timeout_sec: float | None = None,
        exclusive: bool = False,
        resources: tuple[str, ...] | None = None,
        result_policy: Literal[
            "continue_reasoning", "require_final_answer", "return_direct"
        ] = "continue_reasoning",
        source: str = "local",
    ) -> None:
        """Register a plain callable as a tool without writing a :class:`Tool` subclass.

        Useful for tests and dynamic tool registration.
        """
        self.register(
            _AdHocTool(
                name=name,
                description=description or name,
                func=func,
                input_schema=input_schema
                or {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                read_only=read_only
                if read_only is not None
                else (safety_level == "observe"),
                destructive=destructive,
                safety_level=safety_level,
                timeout_sec=timeout_sec,
                exclusive=exclusive,
                resources=resources or (),
                result_policy=result_policy,
                source=source,
            )
        )

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)
        self._spec_cache = None
        self._cached_definitions = None

    # Lookups.

    def get_tool(self, name: str) -> ToolSpec:
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")
        return self._build_spec(tool)

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def iter_specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._build_spec(t) for t in self._tools.values())

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.parameters,
                "annotations": {
                    "readOnlyHint": tool.read_only,
                    "destructiveHint": tool.destructive,
                    "safetyLevel": tool.safety_level,
                    "source": tool.source,
                    "exclusiveHint": tool.exclusive,
                    "concurrencySafeHint": tool.concurrency_safe,
                    "resources": list(tool.resources),
                    "resultPolicy": tool.result_policy,
                },
            }
            for tool in self._tools.values()
        ]

    def get_definitions(self) -> list[dict[str, Any]]:
        if self._cached_definitions is not None:
            return self._cached_definitions
        definitions = sorted(
            (tool.to_schema() for tool in self._tools.values()),
            key=lambda d: d["function"]["name"],
        )
        self._cached_definitions = definitions
        return definitions

    # Execution.

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> str:
        del context
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        tool, casted, error = self.prepare_call(name, dict(arguments or {}))
        if error:
            raise ValueError(error)
        assert tool is not None
        execute = getattr(tool, "execute", None)
        if execute is None:
            raise ValueError(f"Tool {name!r} has no execute method")
        result = execute(**casted)
        if inspect.isawaitable(result):
            result = await result
        return str(result)

    def prepare_call(
        self,
        name: str,
        params: dict[str, Any],
    ) -> tuple[Tool | None, dict[str, Any], str | None]:
        tool = self._tools.get(name)
        if not tool:
            return (
                None,
                params,
                (
                    f"Error: Tool '{name}' not found. Available: {', '.join(self.tool_names)}"
                ),
            )
        cast_params = tool.cast_params(params)
        errors = tool.validate_params(cast_params)
        if errors:
            return (
                tool,
                cast_params,
                (f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors)),
            )
        return tool, cast_params, None

    # Helpers.

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    # Internal.

    @staticmethod
    def _validate_tool_definition(tool: Tool) -> None:
        errors = tool.validate_definition()
        if errors:
            raise ValueError("; ".join(errors))

    def _build_spec(self, tool: Tool) -> ToolSpec:
        if self._spec_cache is None:
            self._spec_cache = {}
        if tool.name not in self._spec_cache:
            self._spec_cache[tool.name] = tool.to_spec()
        return self._spec_cache[tool.name]


class _AdHocTool(Tool):
    """Minimal :class:`Tool` for :meth:`ToolRegistry.register_simple`."""

    _plugin_discoverable = False

    def __init__(
        self,
        *,
        name: str,
        description: str,
        func: Any,
        input_schema: dict[str, Any],
        read_only: bool,
        destructive: bool,
        safety_level: str,
        timeout_sec: float | None,
        exclusive: bool,
        resources: tuple[str, ...],
        result_policy: Literal[
            "continue_reasoning", "require_final_answer", "return_direct"
        ],
        source: str,
    ) -> None:
        self.name = name
        self.description = description
        self._func = func
        self._parameters = input_schema or {
            "type": "object",
            "properties": {},
            "required": [],
        }
        self.read_only = read_only
        self.destructive = destructive
        self.safety_level = safety_level
        self.timeout_sec = timeout_sec
        self.exclusive = exclusive
        self.resources = resources
        self.result_policy = result_policy
        self.source = source

    @property
    def parameters(self) -> dict[str, Any]:  # type: ignore[override]
        return self._parameters

    @classmethod
    def create(cls, ctx: Any) -> _AdHocTool:
        raise NotImplementedError("_AdHocTool is not discoverable by ToolLoader")

    async def execute(self, **kwargs: Any) -> str:
        import inspect

        result = self._func(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        return str(result)
