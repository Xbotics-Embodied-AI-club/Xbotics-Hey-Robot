from __future__ import annotations

from unittest.mock import patch

from hey_robot.agents.tools import loader as loader_module
from hey_robot.agents.tools.base import Tool
from hey_robot.agents.tools.loader import ToolLoader
from hey_robot.agents.tools.registry import ToolRegistry


def test_discover_finds_all_tools():
    loader = ToolLoader()
    discovered = loader.discover()

    tool_names = {cls.name for cls in discovered}
    expected = {
        "get_robot_status",
        "get_task_context",
        "propose_capability",
        "request_capability",
        "request_perception",
        "search_memory",
        "wait",
        "write_memory",
    }
    assert tool_names == expected, (
        f"Expected 8 tools, got {len(discovered)}: {sorted(tool_names)}"
    )
    assert len(discovered) == 8


def test_discover_all_are_robot_tool_subclasses():
    loader = ToolLoader()
    for cls in loader.discover():
        assert issubclass(cls, Tool)
        assert cls.name, f"{cls.__name__} has no name"
        assert cls.description, f"{cls.__name__} has no description"


def test_discover_is_idempotent():
    loader = ToolLoader()
    first = loader.discover()
    second = loader.discover()
    assert first is second  # cached


def test_discover_skips_foundation_modules():
    loader = ToolLoader()
    discovered = loader.discover()
    module_names = {cls.__module__.split(".")[-1] for cls in discovered}
    assert "base" not in module_names
    assert "schema" not in module_names
    assert "registry" not in module_names
    assert "context" not in module_names
    assert "loader" not in module_names


def test_discover_with_test_classes():
    from hey_robot.agents.tools.base import tool_parameters
    from hey_robot.agents.tools.schema import tool_parameters_schema

    @tool_parameters(tool_parameters_schema())
    class _FakeTool(Tool):
        name = "fake_test_tool"
        description = "Test tool"

        @classmethod
        def create(cls, ctx):  # noqa: ARG003
            return cls()

        async def execute(self, **kwargs) -> str:  # noqa: ARG002
            return "fake"

    loader = ToolLoader(test_classes=[_FakeTool])
    result = loader.discover()
    assert len(result) == 1
    assert result[0].name == "fake_test_tool"


def test_load_skips_disabled_tool():
    from hey_robot.agents.tools.base import tool_parameters
    from hey_robot.agents.tools.schema import tool_parameters_schema

    @tool_parameters(tool_parameters_schema())
    class _DisabledTool(Tool):
        name = "disabled_test_tool"
        description = "Disabled test tool."

        @classmethod
        def enabled(cls, ctx):  # noqa: ARG003
            return False

        @classmethod
        def create(cls, ctx):  # noqa: ARG003
            raise AssertionError("disabled tools should not be instantiated")

        async def execute(self, **kwargs) -> str:  # noqa: ARG002
            return "disabled"

    loader = ToolLoader(test_classes=[_DisabledTool])
    registry = ToolRegistry()

    names = loader.load(object(), registry)  # type: ignore[arg-type]

    assert names == []
    assert not registry.has("disabled_test_tool")


def test_load_skips_invalid_tool_and_keeps_valid_tool():
    from hey_robot.agents.tools.base import tool_parameters
    from hey_robot.agents.tools.schema import tool_parameters_schema

    @tool_parameters(tool_parameters_schema())
    class _ValidTool(Tool):
        name = "valid_test_tool"
        description = "Valid test tool."

        @classmethod
        def create(cls, ctx):  # noqa: ARG003
            return cls()

        async def execute(self, **kwargs) -> str:  # noqa: ARG002
            return "valid"

    @tool_parameters(tool_parameters_schema())
    class _InvalidTool(Tool):
        name = ""
        description = "Invalid test tool."

        @classmethod
        def create(cls, ctx):  # noqa: ARG003
            return cls()

        async def execute(self, **kwargs) -> str:  # noqa: ARG002
            return "invalid"

    loader = ToolLoader(test_classes=[_InvalidTool, _ValidTool])
    registry = ToolRegistry()

    with patch.object(loader_module.logger, "exception") as log_exception:
        names = loader.load(object(), registry)  # type: ignore[arg-type]

    assert names == ["valid_test_tool"]
    assert registry.has("valid_test_tool")
    log_exception.assert_called_once()


def test_load_does_not_overwrite_duplicate_tool_name():
    from hey_robot.agents.tools.base import tool_parameters
    from hey_robot.agents.tools.schema import tool_parameters_schema

    @tool_parameters(tool_parameters_schema())
    class _FirstTool(Tool):
        name = "duplicate_test_tool"
        description = "First test tool."

        @classmethod
        def create(cls, ctx):  # noqa: ARG003
            return cls()

        async def execute(self, **kwargs) -> str:  # noqa: ARG002
            return "first"

    @tool_parameters(tool_parameters_schema())
    class _SecondTool(Tool):
        name = "duplicate_test_tool"
        description = "Second test tool."

        @classmethod
        def create(cls, ctx):  # noqa: ARG003
            return cls()

        async def execute(self, **kwargs) -> str:  # noqa: ARG002
            return "second"

    loader = ToolLoader(test_classes=[_FirstTool, _SecondTool])
    registry = ToolRegistry()

    with patch.object(loader_module.logger, "exception") as log_exception:
        names = loader.load(object(), registry)  # type: ignore[arg-type]

    assert names == ["duplicate_test_tool"]
    tool = registry.get("duplicate_test_tool")
    assert tool is not None
    assert tool.description == "First test tool."
    log_exception.assert_called_once()


def test_load_registers_all_tools():
    # Use the real tool suite through discover + a real registry
    loader = ToolLoader()
    registry = ToolRegistry()

    # Build a minimal ToolContext-like stand-in
    class _FakeCtx:
        def __init__(self):
            self.io = None
            self.spec = None
            self.memory = None
            self.autonomy = None
            self.long_horizon = None
            self.task_planner = None
            self.skill_catalog = None
            self.skill_catalog_runtime = None
            self.runtime = None
            self.runtime_state = None
            self.pending_skills = {}
            self.robot_type = None
            self.turn_context = None
            self._current_envelope = lambda: None
            self._configured_robot_type = lambda: None
            self._get_task = lambda: ""
            self._get_robot_status = lambda: ""
            self._activate_task_plan = lambda plan: None  # noqa: ARG005
            self._max_plan_robot_skills = lambda: 3

    # load() calls create(ctx) for each tool. Some tools need working
    # dependencies to pass __init__, so we limit to tools with trivial __init__.
    # Full integration tests cover load with a real ToolContext.
    ctx = _FakeCtx()
    names = loader.load(ctx, registry)  # type: ignore[arg-type]
    assert len(names) > 0
    assert "request_capability" in names
    assert "request_perception" in names
    for name in names:
        assert registry.has(name)
