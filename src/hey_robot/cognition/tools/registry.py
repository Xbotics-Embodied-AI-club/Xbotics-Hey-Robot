"""The single registry for ordinary Harness Tools and projected robot Skills."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

from hey_robot.cognition.tools.models import AgentTool, PreparedToolCall
from hey_robot.cognition.tools.skill_tools import SkillTool
from hey_robot.cognition.tools.task_tools import CompleteTaskTool, ControlTaskTool


class SkillCatalogView(Protocol):
    def list(self) -> tuple[Any, ...]: ...


@dataclass(frozen=True)
class ToolDependencies:
    skill_catalog: SkillCatalogView
    extra_tools: tuple[AgentTool, ...] = ()


class ToolRegistry:
    """Publish schemas and prepare typed calls without executing external IO."""

    def __init__(self, deps: ToolDependencies) -> None:
        core_tools: dict[str, AgentTool] = {
            spec.name: SkillTool(spec) for spec in deps.skill_catalog.list()
        }
        task_tools: dict[str, AgentTool] = {
            CompleteTaskTool.name: cast(AgentTool, CompleteTaskTool()),
            ControlTaskTool.name: cast(AgentTool, ControlTaskTool()),
        }
        overlap = set(core_tools) & set(task_tools)
        if overlap:
            raise ValueError(f"skill name conflicts with task tool: {sorted(overlap)}")
        core_tools.update(task_tools)
        for tool in deps.extra_tools:
            name = getattr(tool, "name", "")
            if not isinstance(name, str) or not name or name in core_tools:
                raise ValueError(f"invalid or duplicate Agent tool: {name!r}")
            if not callable(getattr(tool, "prepare", None)):
                raise ValueError(f"Agent tool does not implement prepare(): {name!r}")
            core_tools[name] = tool
        self._tools = core_tools

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return [tool.schema for tool in self._tools.values()]

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._tools)

    def prepare(self, name: str, arguments: dict[str, Any]) -> PreparedToolCall:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(name)
        return cast(PreparedToolCall, tool.prepare(arguments))
