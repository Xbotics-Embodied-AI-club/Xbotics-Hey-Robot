"""Abstract base class for robot agent tools.

Each tool is a :class:`Tool` subclass in its own file.  The framework
discovers tools via :class:`~hey_robot.agents.tools.loader.ToolLoader` and
registers them in a :class:`~hey_robot.agents.tools.registry.ToolRegistry`.

Runtime adapter
---------------

:meth:`Tool.to_spec` produces a
:class:`~hey_robot.agents.runtime.registry.ToolSpec` for the current
``CapabilityResolver`` / ``PermissionManager`` / ``ToolExecutor`` pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from hey_robot.agents.tools.schema import Schema

if TYPE_CHECKING:
    from hey_robot.agents.runtime.registry import ToolSpec

_BOOL_TRUE = frozenset(("true", "1", "yes"))
_BOOL_FALSE = frozenset(("false", "0", "no"))
_JSON_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


class Tool(ABC):
    """One agent capability: submit a skill, read state, write memory, etc.

    Subclasses must provide:

    * ``name`` (class-level or property)
    * ``description`` (class-level or property)
    * ``parameters`` via the :func:`tool_parameters` decorator
    * ``execute(**kwargs)`` async method
    * ``safety_level`` class attribute
    * ``create(ctx)`` factory classmethod
    """

    # Required subclass attributes.
    name: str = ""
    description: str = ""
    safety_level: str = (
        "normal"  # "observe" | "actuate" | "communicate" | "memory_write"
    )

    # Optional class-level overrides.
    read_only: bool = False
    destructive: bool = False
    exclusive: bool = False
    timeout_sec: float | None = None
    source: str = "local"
    resources: tuple[str, ...] = ()
    result_policy: Literal[
        "continue_reasoning", "require_final_answer", "return_direct"
    ] = "continue_reasoning"
    _plugin_discoverable: bool = True
    _scopes: ClassVar[set[str]] = {"core"}

    # JSON Schema, usually overridden by @tool_parameters.
    parameters: dict[str, Any] = {"type": "object", "properties": {}, "required": []}  # noqa: RUF012

    @property
    def concurrency_safe(self) -> bool:
        """Whether this tool can run alongside other concurrency-safe tools."""
        return self.read_only and not self.destructive and not self.exclusive

    # JSON Schema coercion / validation.

    @staticmethod
    def _resolve_type(t: Any) -> str | None:
        return Schema.resolve_json_schema_type(t)

    @classmethod
    def enabled(cls, ctx: Any) -> bool:
        """Whether this tool should be registered for the current runtime context."""
        del ctx
        return True

    @classmethod
    @abstractmethod
    def create(cls, ctx: Any) -> Tool:
        """Factory: build this tool from a :class:`ToolContext`."""
        ...

    def _cast_object(self, obj: Any, schema: dict[str, Any]) -> Any:
        if not isinstance(obj, dict):
            return obj
        props = schema.get("properties", {})
        return {
            k: self._cast_value(v, props[k]) if k in props else v
            for k, v in obj.items()
        }

    def cast_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Apply safe schema-driven casts before validation."""
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            return params
        casted = self._cast_object(params, schema)
        return casted if isinstance(casted, dict) else params

    def _cast_value(self, val: Any, schema: dict[str, Any]) -> Any:
        t = self._resolve_type(schema.get("type"))
        if t == "boolean" and isinstance(val, bool):
            return val
        if t == "integer" and isinstance(val, int) and not isinstance(val, bool):
            return val
        if t in _JSON_TYPE_MAP and t not in ("boolean", "integer", "array", "object"):
            expected = _JSON_TYPE_MAP[t]
            if isinstance(val, expected):
                if (
                    t == "string"
                    and not val
                    and isinstance(schema.get("type"), list)
                    and "null" in schema["type"]
                ):
                    return None
                return val
        if isinstance(val, str) and t in ("integer", "number"):
            try:
                return int(val) if t == "integer" else float(val)
            except ValueError:
                return val
        if t == "string":
            if val is None:
                return None
            s = str(val)
            schema_type = schema.get("type")
            if not s and isinstance(schema_type, list) and "null" in schema_type:
                return None
            return s
        if t == "boolean" and isinstance(val, str):
            low = val.lower()
            if low in _BOOL_TRUE:
                return True
            if low in _BOOL_FALSE:
                return False
            return val
        if t == "array" and isinstance(val, list):
            items = schema.get("items")
            return [self._cast_value(x, items) for x in val] if items else val
        if t == "object" and isinstance(val, dict):
            return self._cast_object(val, schema)
        return val

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """Validate against JSON schema; empty list means valid."""
        if not isinstance(params, dict):
            return [f"parameters must be an object, got {type(params).__name__}"]
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            raise ValueError(f"Schema must be object type, got {schema.get('type')!r}")
        return Schema.validate_json_schema_value(
            params, {**schema, "type": "object"}, ""
        )

    def validate_definition(self) -> list[str]:
        """Validate static tool metadata and input schema before registration."""
        errors: list[str] = []
        if not isinstance(self.name, str) or not self.name.strip():
            errors.append("tool name must not be empty")
        if not isinstance(self.description, str) or not self.description.strip():
            errors.append(
                f"tool {self.name or '<unnamed>'} description must not be empty"
            )
        if not isinstance(self.parameters, dict):
            errors.append(
                f"tool {self.name or '<unnamed>'} parameters must be a JSON object schema"
            )
        elif self.parameters.get("type", "object") != "object":
            errors.append(
                f"tool {self.name or '<unnamed>'} parameters schema must have type='object'"
            )
        return errors

    def to_schema(self) -> dict[str, Any]:
        """OpenAI-compatible function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    # Runtime ToolSpec adapter.

    def to_spec(self) -> ToolSpec:
        """Produce a runtime ``ToolSpec`` consumable by the safety pipeline.

        The returned spec wraps :meth:`execute` for the runtime executor.
        """
        from hey_robot.agents.runtime.registry import ToolSpec

        async def _execute_bound(**kwargs: Any) -> str:
            execute = getattr(self, "execute")
            result = execute(**kwargs)
            import inspect

            if inspect.isawaitable(result):
                result = await result
            return str(result)

        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.parameters,
            func=_execute_bound,
            read_only=self.read_only,
            destructive=self.destructive,
            safety_level=self.safety_level,
            timeout_sec=self.timeout_sec,
            source=self.source,
            exclusive=self.exclusive,
            resources=self.resources,
            result_policy=self.result_policy,
        )


def tool_parameters(schema: dict[str, Any]):
    """Class decorator: attach a JSON Schema to a :class:`Tool` subclass.

    Use on :class:`Tool` subclasses instead of setting ``parameters`` manually.

    Example::

        @tool_parameters(tool_parameters_schema(
            reason=StringSchema("Why wait"),
            required=["reason"],
        ))
        class WaitTool(Tool):
            ...
    """

    def decorator(cls: type[Tool]) -> type[Tool]:
        cls.parameters = deepcopy(schema)
        return cls

    return decorator
