"""The only tools exposed to the autonomous model.

They are adapters from validated model arguments to a typed proposal.  They
do not hold IO, a bus connection, or a skill gateway.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from hey_robot.protocol import ActionProposal


class SkillCatalogView(Protocol):
    def get(self, name: str) -> Any: ...

    def list(self) -> tuple[Any, ...]: ...


@dataclass(frozen=True)
class AgentToolDependencies:
    skill_catalog: SkillCatalogView


class RequestObservationTool:
    name: ClassVar[str] = "request_observation"
    schema: ClassVar[dict[str, Any]] = {
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
    name: ClassVar[str] = "request_skill"
    schema: ClassVar[dict[str, Any]] = {
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
                "required": ["skill"],
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
        if not isinstance(slots, dict):
            raise ValueError("slots must be an object")
        try:
            spec = self._catalog.get(skill.strip())
        except KeyError as err:
            raise ValueError(f"unknown skill: {skill}") from err
        category = str(getattr(spec, "category", ""))
        if skill.strip() == "inspect_scene" or category in {"observe", "perception"}:
            raise ValueError("observation skills must use request_observation")
        _validate_slots(dict(getattr(spec, "input_schema", {}) or {}), slots)
        normalized_objective = (
            objective.strip()
            if isinstance(objective, str) and objective.strip()
            else f"execute {skill.strip()}"
        )
        return ActionProposal("skill", skill.strip(), normalized_objective, dict(slots))


def _validate_slots(schema: dict[str, Any], slots: dict[str, Any]) -> None:
    """Small schema check at the tool boundary; the Supervisor checks again."""
    required = schema.get("required", [])
    for field in required if isinstance(required, list) else []:
        if field not in slots:
            raise ValueError(f"missing required slot: {field}")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return
    for name, value in slots.items():
        field = properties.get(name)
        if not isinstance(field, dict):
            continue
        expected = field.get("type")
        if expected == "string" and not isinstance(value, str):
            raise ValueError(f"slot {name} must be a string")
        if expected == "number" and (
            not isinstance(value, int | float) or isinstance(value, bool)
        ):
            raise ValueError(f"slot {name} must be a number")
        if expected == "integer" and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            raise ValueError(f"slot {name} must be an integer")
        if expected == "boolean" and not isinstance(value, bool):
            raise ValueError(f"slot {name} must be a boolean")
        if expected == "object" and not isinstance(value, dict):
            raise ValueError(f"slot {name} must be an object")


class ToolProtocol(Protocol):
    @property
    def schema(self) -> dict[str, Any]: ...

    def proposal(self, arguments: dict[str, Any]) -> ActionProposal: ...


class AutonomousToolRegistry:
    def __init__(self, deps: AgentToolDependencies) -> None:
        self._tools: dict[str, ToolProtocol] = {
            RequestObservationTool.name: RequestObservationTool(),
            RequestSkillTool.name: RequestSkillTool(deps.skill_catalog),
        }

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return [tool.schema for tool in self._tools.values()]

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._tools)

    @property
    def instructions(self) -> str:
        """Expose enabled skill contracts to the model without per-skill code."""
        request_skill = self._tools[RequestSkillTool.name]
        assert isinstance(request_skill, RequestSkillTool)
        contracts = [
            {
                "name": spec.name,
                "description": spec.description,
                "category": spec.category,
                "input_schema": spec.input_schema,
            }
            for spec in request_skill._catalog.list()
            if spec.name != "inspect_scene"
            and spec.category not in {"observe", "perception"}
        ]
        return (
            "For request_skill, choose a skill from these contracts and put its "
            "arguments exactly in slots. Ask the user to clarify if a required "
            "slot is unknown.\n" + json.dumps(contracts, ensure_ascii=False)
        )

    def proposal(self, name: str, arguments: dict[str, Any]) -> ActionProposal:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(name)
        return tool.proposal(arguments)


def build_agent_tools(deps: AgentToolDependencies) -> AutonomousToolRegistry:
    return AutonomousToolRegistry(deps)
