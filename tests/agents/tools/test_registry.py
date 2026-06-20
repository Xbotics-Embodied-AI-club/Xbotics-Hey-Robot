from __future__ import annotations

import pytest

from hey_robot.agents.tools.base import Tool, tool_parameters
from hey_robot.agents.tools.registry import ToolRegistry
from hey_robot.agents.tools.schema import StringSchema, tool_parameters_schema


@tool_parameters(
    tool_parameters_schema(
        text=StringSchema("message"),
        required=["text"],
    )
)
class _EchoTool(Tool):
    name = "echo"
    description = "Echo text back."
    safety_level = "normal"

    @classmethod
    def create(cls, ctx):  # noqa: ARG003
        return cls()

    async def execute(self, text: str) -> str:
        return text


@tool_parameters(
    tool_parameters_schema(
        count=StringSchema("Number of items", min_length=1),
    )
)
class _CountTool(Tool):
    name = "count"
    description = "Count something."
    read_only = True
    safety_level = "observe"

    @classmethod
    def create(cls, ctx):  # noqa: ARG003
        return cls()

    async def execute(self, count: str = "0") -> str:
        return str(len(count))


class TestRegistryRegistration:
    def test_register_and_get_tool(self):
        reg = ToolRegistry()
        reg.register(_EchoTool())
        spec = reg.get_tool("echo")
        assert spec.name == "echo"
        assert spec.description == "Echo text back."

    def test_has_and_contains(self):
        reg = ToolRegistry()
        reg.register(_EchoTool())
        assert reg.has("echo") is True
        assert ("echo" in reg) is True
        assert reg.has("nonexistent") is False
        assert ("nonexistent" in reg) is False

    def test_get_raw_instance(self):
        reg = ToolRegistry()
        tool = _EchoTool()
        reg.register(tool)
        assert reg.get("echo") is tool

    def test_get_unknown_raises(self):
        reg = ToolRegistry()
        with pytest.raises(ValueError, match="Unknown tool"):
            reg.get_tool("missing")

    def test_len(self):
        reg = ToolRegistry()
        assert len(reg) == 0
        reg.register(_EchoTool())
        assert len(reg) == 1

    def test_unregister(self):
        reg = ToolRegistry()
        reg.register(_EchoTool())
        reg.unregister("echo")
        assert len(reg) == 0
        assert reg.has("echo") is False

    def test_tool_names(self):
        reg = ToolRegistry()
        reg.register(_EchoTool())
        reg.register(_CountTool())
        assert set(reg.tool_names) == {"echo", "count"}

    def test_register_rejects_empty_name(self):
        class _BadTool(_EchoTool):
            name = ""

        reg = ToolRegistry()
        with pytest.raises(ValueError, match="name must not be empty"):
            reg.register(_BadTool())

    def test_register_rejects_empty_description(self):
        class _BadTool(_EchoTool):
            description = ""

        reg = ToolRegistry()
        with pytest.raises(ValueError, match="description must not be empty"):
            reg.register(_BadTool())

    def test_register_rejects_non_object_schema(self):
        class _BadTool(_EchoTool):
            parameters = {"type": "string"}  # noqa: RUF012

        reg = ToolRegistry()
        with pytest.raises(ValueError, match="type='object'"):
            reg.register(_BadTool())

    def test_register_rejects_duplicate_name(self):
        reg = ToolRegistry()
        reg.register(_EchoTool())
        with pytest.raises(ValueError, match="already registered"):
            reg.register(_EchoTool())


class TestRegistryListTools:
    def test_list_tools_format(self):
        reg = ToolRegistry()
        reg.register(_EchoTool())
        tools = reg.list_tools()
        assert len(tools) == 1
        t = tools[0]
        assert t["name"] == "echo"
        assert "inputSchema" in t
        assert "annotations" in t
        assert t["annotations"]["safetyLevel"] == "normal"

    def test_get_definitions_sorted(self):
        reg = ToolRegistry()
        reg.register(_CountTool())
        reg.register(_EchoTool())
        defs = reg.get_definitions()
        assert len(defs) == 2
        names = [d["function"]["name"] for d in defs]
        assert names == sorted(names)

    def test_iter_specs(self):
        reg = ToolRegistry()
        reg.register(_EchoTool())
        specs = reg.iter_specs()
        assert len(specs) == 1
        assert specs[0].name == "echo"


class TestRegistryPrepareCall:
    def test_prepare_call_success(self):
        reg = ToolRegistry()
        reg.register(_EchoTool())
        tool, casted, error = reg.prepare_call("echo", {"text": "hello"})
        assert error is None
        assert tool is not None
        assert casted["text"] == "hello"

    def test_prepare_call_unknown_tool(self):
        reg = ToolRegistry()
        tool, casted, error = reg.prepare_call("missing", {})  # noqa: RUF059
        assert tool is None
        assert error is not None
        assert "not found" in error.lower()

    def test_prepare_call_invalid_params(self):
        reg = ToolRegistry()
        reg.register(_EchoTool())
        tool, casted, error = reg.prepare_call("echo", {})  # noqa: RUF059
        assert error is not None
        assert "invalid" in error.lower()

    def test_prepare_call_type_coercion(self):
        reg = ToolRegistry()
        reg.register(_CountTool())
        tool, casted, error = reg.prepare_call("count", {"count": 42})  # noqa: RUF059
        assert error is None
        assert casted["count"] == "42"


class TestRegistryCallTool:
    async def test_call_tool_executes(self):
        reg = ToolRegistry()
        reg.register(_EchoTool())
        result = await reg.call_tool("echo", {"text": "hello"})
        assert result == "hello"

    async def test_call_tool_validates_arguments(self):
        reg = ToolRegistry()
        reg.register(_EchoTool())
        with pytest.raises(ValueError, match="Invalid parameters"):
            await reg.call_tool("echo", {})

    async def test_call_tool_applies_type_coercion(self):
        reg = ToolRegistry()
        reg.register(_CountTool())
        result = await reg.call_tool("count", {"count": 42})
        assert result == "2"

    async def test_call_tool_unknown_raises(self):
        reg = ToolRegistry()
        with pytest.raises(ValueError, match="Unknown tool"):
            await reg.call_tool("missing", {})


class TestRegistryCacheInvalidation:
    def test_cache_cleared_on_register(self):
        reg = ToolRegistry()
        reg.register(_EchoTool())
        # force cache population
        reg.get_definitions()
        reg.iter_specs()
        # add another tool 鈥?caches must invalidate
        reg.register(_CountTool())
        defs = reg.get_definitions()
        assert len(defs) == 2

    def test_cache_cleared_on_unregister(self):
        reg = ToolRegistry()
        reg.register(_EchoTool())
        reg.register(_CountTool())
        reg.get_definitions()
        reg.unregister("echo")
        defs = reg.get_definitions()
        assert len(defs) == 1
