"""The only tools exposed to the autonomous model.

They are adapters from validated model arguments to a typed proposal.  They
do not hold IO, a bus connection, or a skill gateway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from hey_robot.protocol import ActionProposal


class SkillCatalogView(Protocol):
    def get(self, name: str) -> Any: ...


@dataclass(frozen=True)
class AgentToolDependencies:
    skill_catalog: SkillCatalogView


class RequestObservationTool:
    name = "request_observation"
    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": "Request one fresh scene observation.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    }

    def proposal(self, arguments: dict[str, Any]) -> ActionProposal:
        question = arguments.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        return ActionProposal(
            "observation",
            "inspect_scene",
            question.strip(),
            {"question": question.strip()},
        )


class RequestSkillTool:
    name = "request_skill"
    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": "Request one non-observation robot skill.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill": {"type": "string"},
                    "objective": {"type": "string"},
                    "slots": {"type": "object"},
                },
                "required": ["skill", "objective"],
                "additionalProperties": False,
            },
        },
    }

    def __init__(self, catalog: SkillCatalogView) -> None:
        self._catalog = catalog

    def proposal(self, arguments: dict[str, Any]) -> ActionProposal:
        skill = arguments.get("skill")
        objective = arguments.get("objective")
        slots = arguments.get("slots", {})
        if not isinstance(skill, str) or not skill.strip():
            raise ValueError("skill must be a non-empty string")
        if not isinstance(objective, str) or not objective.strip():
            raise ValueError("objective must be a non-empty string")
        if not isinstance(slots, dict):
            raise ValueError("slots must be an object")
        try:
            spec = self._catalog.get(skill.strip())
        except KeyError:
            raise ValueError(f"unknown skill: {skill}")
        category = str(getattr(spec, "category", ""))
        if skill.strip() == "inspect_scene" or category in {"observe", "perception"}:
            raise ValueError("observation skills must use request_observation")
        return ActionProposal("skill", skill.strip(), objective.strip(), dict(slots))


class AutonomousToolRegistry:
    def __init__(self, deps: AgentToolDependencies) -> None:
        self._tools = {
            RequestObservationTool.name: RequestObservationTool(),
            RequestSkillTool.name: RequestSkillTool(deps.skill_catalog),
        }

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return [tool.schema for tool in self._tools.values()]

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._tools)

    def proposal(self, name: str, arguments: dict[str, Any]) -> ActionProposal:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(name)
        return tool.proposal(arguments)


def build_agent_tools(deps: AgentToolDependencies) -> AutonomousToolRegistry:
    return AutonomousToolRegistry(deps)
