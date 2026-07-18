"""单一 Robot Agent 的规范化、仅提案式工具接口。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from hey_robot.protocol import ActionProposal


class SkillCatalogView(Protocol):
    def get(self, name: str) -> Any: ...

    def list(self) -> tuple[Any, ...]: ...


@dataclass(frozen=True)
class ToolDependencies:
    skill_catalog: SkillCatalogView
    extra_tools: tuple[Any, ...] = ()


@dataclass(frozen=True)
class CompleteTaskProposal:
    recap: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class ControlTaskProposal:
    action: str
    reason: str


class RequestObservationTool:
    name: ClassVar[str] = "request_observation"
    schema: ClassVar[dict[str, Any]] = {
        "type": "function",
        "function": {
            "name": name,
            "description": "为回答问题或继续任务，请求一次新的场景观察。",
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
            "description": (
                "为当前用户目标提出一个明确、有界的机器人技能。运行时会把第一次"
                "机器人操作自动纳入任务，并在每个结果返回后继续审议。"
            ),
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
        schema = dict(getattr(spec, "input_schema", {}) or {})
        slots = _apply_schema_defaults(schema, slots)
        _validate_slots(schema, slots)
        normalized_objective = (
            objective.strip()
            if isinstance(objective, str) and objective.strip()
            else f"execute {skill.strip()}"
        )
        return ActionProposal("skill", skill.strip(), normalized_objective, dict(slots))


class CompleteTaskTool:
    name: ClassVar[str] = "complete_task"
    schema: ClassVar[dict[str, Any]] = {
        "type": "function",
        "function": {
            "name": name,
            "description": "引用当前任务证据并提议结束 active task。",
            "parameters": {
                "type": "object",
                "properties": {
                    "recap": {"type": "string"},
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["recap", "evidence_ids"],
                "additionalProperties": False,
            },
        },
    }

    def proposal(self, arguments: dict[str, Any]) -> CompleteTaskProposal:
        recap = arguments.get("recap")
        evidence_ids = arguments.get("evidence_ids")
        if not isinstance(recap, str) or not recap.strip():
            raise ValueError("recap must be a non-empty string")
        if not isinstance(evidence_ids, list) or not evidence_ids:
            raise ValueError("evidence_ids must be a non-empty array")
        normalized = tuple(
            item.strip()
            for item in evidence_ids
            if isinstance(item, str) and item.strip()
        )
        if len(normalized) != len(evidence_ids):
            raise ValueError("evidence_ids must contain only non-empty strings")
        return CompleteTaskProposal(recap.strip(), normalized)


class ControlTaskTool:
    name: ClassVar[str] = "control_task"
    schema: ClassVar[dict[str, Any]] = {
        "type": "function",
        "function": {
            "name": name,
            "description": "取消、阻塞确认或紧急停止当前持续任务。",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["cancel", "block", "emergency_stop"],
                    },
                    "reason": {
                        "type": "string",
                        "description": "面向用户的简短原因。",
                    },
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        },
    }

    def proposal(self, arguments: dict[str, Any]) -> ControlTaskProposal:
        action = arguments.get("action")
        if action not in {"cancel", "block", "emergency_stop"}:
            raise ValueError("action is invalid")
        reason = arguments.get("reason", "")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("reason must be a string")
        return ControlTaskProposal(
            action, reason.strip() if isinstance(reason, str) else ""
        )


class ToolRegistry:
    """唯一面向模型的工具注册表；工具只返回提案，不执行 IO。"""

    def __init__(self, deps: ToolDependencies) -> None:
        core_tools: dict[str, Any] = {
            RequestObservationTool.name: RequestObservationTool(),
            RequestSkillTool.name: RequestSkillTool(deps.skill_catalog),
            CompleteTaskTool.name: CompleteTaskTool(),
            ControlTaskTool.name: ControlTaskTool(),
        }
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
    def instructions(self) -> str:
        contracts = [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_schema,
            }
            for spec in self._catalog.list()
            if spec.name != "inspect_scene"
            and spec.category not in {"observe", "perception"}
        ]
        return (
            "以下 JSON 是 request_skill 可以使用的 Skill 契约。只能选择其中存在的 name，"
            "并按对应 input_schema 提供 slots：\n"
            + json.dumps(contracts, ensure_ascii=False)
        )

    def proposal(
        self, name: str, arguments: dict[str, Any]
    ) -> ActionProposal | CompleteTaskProposal | ControlTaskProposal:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(name)
        proposal = tool.proposal(arguments)
        if not isinstance(
            proposal,
            ActionProposal | CompleteTaskProposal | ControlTaskProposal,
        ):
            raise TypeError(f"unsupported Robot Agent proposal: {type(proposal)!r}")
        return proposal


def _validate_slots(schema: dict[str, Any], slots: dict[str, Any]) -> None:
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
        if (
            expected == "number"
            and isinstance(value, int | float)
            and not isinstance(value, bool)
        ):
            minimum = field.get("minimum")
            maximum = field.get("maximum")
            if isinstance(minimum, int | float) and value < minimum:
                raise ValueError(f"slot {name} must be >= {minimum}")
            if isinstance(maximum, int | float) and value > maximum:
                raise ValueError(f"slot {name} must be <= {maximum}")
        if expected == "integer" and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            raise ValueError(f"slot {name} must be an integer")
        if expected == "boolean" and not isinstance(value, bool):
            raise ValueError(f"slot {name} must be a boolean")
        if expected == "object" and not isinstance(value, dict):
            raise ValueError(f"slot {name} must be an object")


def _apply_schema_defaults(
    schema: dict[str, Any], slots: dict[str, Any]
) -> dict[str, Any]:
    """Fill declared skill defaults before validating a bounded operation."""
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return dict(slots)
    resolved = dict(slots)
    for name, field in properties.items():
        if name not in resolved and isinstance(field, dict) and "default" in field:
            resolved[name] = field["default"]
    return resolved
