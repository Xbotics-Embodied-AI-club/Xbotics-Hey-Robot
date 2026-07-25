"""The single registry for inline Tools and agent-visible physical Skills."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from hey_robot.cognition.tools.models import (
    AgentTool,
    PhysicalToolCall,
    PreparedToolCall,
    ToolSpec,
)
from hey_robot.tool_schema import validate_arguments


@dataclass(frozen=True)
class ToolDependencies:
    skills: tuple[Any, ...]
    extra_tools: tuple[AgentTool, ...] = ()


class ToolRegistry:
    """Publish schemas and prepare typed calls without executing external IO."""

    def __init__(self, deps: ToolDependencies) -> None:
        self._skills = {str(skill.name): skill for skill in deps.skills}
        if len(self._skills) != len(deps.skills):
            raise ValueError("duplicate skill tool name")
        core_tools: dict[str, AgentTool] = {}
        for tool in deps.extra_tools:
            name = getattr(tool, "name", "")
            if (
                not isinstance(name, str)
                or not name
                or name in core_tools
                or name in self._skills
            ):
                raise ValueError(f"invalid or duplicate Agent tool: {name!r}")
            if not callable(getattr(tool, "prepare", None)):
                raise ValueError(f"Agent tool does not implement prepare(): {name!r}")
            core_tools[name] = tool
        self._tools = core_tools

    @property
    def definitions(self) -> list[dict[str, Any]]:
        skill_definitions = [
            ToolSpec(
                str(skill.name),
                str(skill.description),
                dict(skill.parameters),
            ).definition
            for skill in self._skills.values()
        ]
        return skill_definitions + [tool.schema for tool in self._tools.values()]

    @property
    def names(self) -> frozenset[str]:
        return frozenset((*self._skills, *self._tools))

    def prepare(self, name: str, arguments: dict[str, Any]) -> PreparedToolCall:
        skill = self._skills.get(name)
        if skill is not None:
            normalized = validate_arguments(dict(skill.parameters), arguments)
            return PhysicalToolCall(name, normalized)
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(name)
        return cast(PreparedToolCall, tool.prepare(arguments))
