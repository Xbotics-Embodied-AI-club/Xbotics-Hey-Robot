"""Agent-facing projections of registered robot skills."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from hey_robot.skills.runner import validate_arguments


@dataclass(frozen=True)
class SkillCallProposal:
    """Cognition-internal proposal for one bounded skill call."""

    intent_kind: Literal["skill", "observation"]
    name: str
    objective: str
    arguments: dict[str, Any]

    @property
    def skill_name(self) -> str:
        return self.name


class SkillTool:
    """Expose one registered skill as one typed model tool."""

    def __init__(self, spec: Any) -> None:
        self._spec = spec
        self.name = str(spec.name)
        self._parameters = _parameters_for(spec)
        self.schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": str(spec.description),
                "parameters": self._parameters,
            },
        }

    def proposal(self, arguments: dict[str, Any]) -> SkillCallProposal:
        normalized = validate_arguments(self._parameters, arguments)
        category = str(getattr(self._spec, "category", ""))
        intent_kind: Literal["skill", "observation"] = (
            "observation"
            if self.name == "inspect_scene" or category in {"observe", "perception"}
            else "skill"
        )
        objective = _objective(self.name, normalized)
        return SkillCallProposal(intent_kind, self.name, objective, normalized)


def skill_call_payload(proposal: SkillCallProposal) -> dict[str, Any]:
    return {
        "intent_kind": proposal.intent_kind,
        "name": proposal.name,
        "skill_name": proposal.name,
        "objective": proposal.objective,
        "arguments": dict(proposal.arguments),
    }


def skill_call_from_payload(payload: dict[str, Any]) -> SkillCallProposal:
    name = payload.get("name", payload.get("skill_name"))
    if not isinstance(name, str) or not name:
        raise ValueError("skill proposal payload must include name")
    intent_kind = payload.get("intent_kind")
    if intent_kind not in {"skill", "observation"}:
        raise ValueError("skill proposal payload has invalid intent_kind")
    return SkillCallProposal(
        intent_kind,
        name,
        payload["objective"],
        dict(payload.get("arguments", {})),
    )


def _objective(name: str, arguments: dict[str, Any]) -> str:
    question = arguments.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()
    task_prompt = arguments.get("task_prompt") or arguments.get("objective")
    if isinstance(task_prompt, str) and task_prompt.strip():
        return task_prompt.strip()
    return f"execute {name}"


def _parameters_for(spec: Any) -> dict[str, Any]:
    parameters = getattr(spec, "parameters", None)
    if not isinstance(parameters, dict):
        parameters = getattr(spec, "input_schema", {})
    return dict(parameters or {})
