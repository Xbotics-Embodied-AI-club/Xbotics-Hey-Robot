"""单一 Robot Agent 的规范化、仅提案式工具接口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from hey_robot.cognition.tools.skill_tools import SkillCallProposal, SkillTool
from hey_robot.cognition.tools.task_tools import (
    CompleteTaskProposal,
    CompleteTaskTool,
    ControlTaskProposal,
    ControlTaskTool,
)


class SkillCatalogView(Protocol):
    def get(self, name: str) -> Any: ...

    def list(self) -> tuple[Any, ...]: ...


@dataclass(frozen=True)
class ToolDependencies:
    skill_catalog: SkillCatalogView
    extra_tools: tuple[Any, ...] = ()


class ToolRegistry:
    """唯一面向模型的工具注册表；工具只返回提案，不执行 IO。"""

    def __init__(self, deps: ToolDependencies) -> None:
        core_tools: dict[str, Any] = {
            spec.name: SkillTool(spec) for spec in deps.skill_catalog.list()
        }
        task_tools: dict[str, Any] = {
            CompleteTaskTool.name: CompleteTaskTool(),
            ControlTaskTool.name: ControlTaskTool(),
        }
        overlap = set(core_tools) & set(task_tools)
        if overlap:
            raise ValueError(f"skill name conflicts with task tool: {sorted(overlap)}")
        core_tools.update(task_tools)
        for tool in deps.extra_tools:
            name = getattr(tool, "name", "")
            if not isinstance(name, str) or not name or name in core_tools:
                raise ValueError(f"invalid or duplicate Robot Agent tool: {name!r}")
            core_tools[name] = tool
        self._tools = core_tools
        self._catalog = deps.skill_catalog

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return [tool.schema for tool in self._tools.values()]

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._tools)

    @property
    def instructions(self) -> str:
        return "机器人能力以独立工具提供。只调用当前 Tool schema 中存在的能力。"

    def proposal(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(name)
        proposal = tool.proposal(arguments)
        if not isinstance(
            proposal, SkillCallProposal | CompleteTaskProposal | ControlTaskProposal
        ):
            raise TypeError(f"unsupported Robot Agent proposal: {type(proposal)!r}")
        return proposal
