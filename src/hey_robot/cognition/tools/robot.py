"""单一 Robot Agent 的规范化、仅提案式工具接口。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from hey_robot.cognition.conversation_goal import (
    DEFAULT_GOAL_TEMPLATES,
    GoalControlProposal,
    GoalProposal,
)
from hey_robot.protocol import ActionProposal


class SkillCatalogView(Protocol):
    def get(self, name: str) -> Any: ...

    def list(self) -> tuple[Any, ...]: ...


@dataclass(frozen=True)
class ToolDependencies:
    skill_catalog: SkillCatalogView
    goal_kinds: tuple[str, ...] = tuple(item.name for item in DEFAULT_GOAL_TEMPLATES)
    extra_tools: tuple[Any, ...] = ()


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
                "只提出一个明确、有界的机器人技能。不得用它拼接多个动作来实现"
                "长程世界目标；此类请求必须使用 request_goal。"
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
        _validate_slots(dict(getattr(spec, "input_schema", {}) or {}), slots)
        normalized_objective = (
            objective.strip()
            if isinstance(objective, str) and objective.strip()
            else f"execute {skill.strip()}"
        )
        return ActionProposal("skill", skill.strip(), normalized_objective, dict(slots))


class RequestGoalTool:
    name: ClassVar[str] = "request_goal"

    def __init__(self, goal_kinds: tuple[str, ...]) -> None:
        self._goal_kinds = frozenset(goal_kinds)
        self.schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "创建持续执行、由证据验证的世界目标。target 应使用当前上下文"
                    "中的精确实体 ID。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "goal_kind": {
                            "type": "string",
                            "enum": sorted(self._goal_kinds),
                        },
                        "objective": {"type": "string"},
                        "target": {"type": "string"},
                        "destination": {"type": "string"},
                    },
                    "required": ["goal_kind", "objective", "target"],
                    "additionalProperties": False,
                },
            },
        }

    def proposal(self, arguments: dict[str, Any]) -> GoalProposal:
        kind = arguments.get("goal_kind")
        objective = arguments.get("objective")
        target = arguments.get("target")
        destination = arguments.get("destination")
        if not isinstance(kind, str) or kind not in self._goal_kinds:
            raise ValueError("goal_kind is invalid")
        if not isinstance(objective, str) or not objective.strip():
            raise ValueError("objective must be a non-empty string")
        if not isinstance(target, str) or not target.strip():
            raise ValueError("target must be a non-empty string")
        if destination is not None and not isinstance(destination, str):
            raise ValueError("destination must be a string")
        return GoalProposal(
            kind,
            objective.strip(),
            target.strip(),
            destination.strip() if destination else None,
        )


class ControlGoalTool:
    name: ClassVar[str] = "control_goal"
    schema: ClassVar[dict[str, Any]] = {
        "type": "function",
        "function": {
            "name": name,
            "description": "取消、紧急停止或确认当前持续 Goal。",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["cancel", "emergency_stop", "confirm"],
                    },
                    "condition_id": {
                        "type": "string",
                        "description": "确认 waiting_condition 时必须提供。",
                    },
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        },
    }

    def proposal(self, arguments: dict[str, Any]) -> GoalControlProposal:
        action = arguments.get("action")
        if action not in {"cancel", "emergency_stop", "confirm"}:
            raise ValueError("action is invalid")
        condition_id = str(arguments.get("condition_id", "")).strip() or None
        if action == "confirm" and condition_id is None:
            raise ValueError("condition_id is required when action is confirm")
        return GoalControlProposal(action, condition_id)


class ToolRegistry:
    """唯一面向模型的工具注册表；工具只返回提案，不执行 IO。"""

    def __init__(self, deps: ToolDependencies) -> None:
        core_tools: dict[str, Any] = {
            RequestObservationTool.name: RequestObservationTool(),
            RequestSkillTool.name: RequestSkillTool(deps.skill_catalog),
            RequestGoalTool.name: RequestGoalTool(deps.goal_kinds),
            ControlGoalTool.name: ControlGoalTool(),
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
    ) -> ActionProposal | GoalProposal | GoalControlProposal:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(name)
        proposal = tool.proposal(arguments)
        if not isinstance(
            proposal, ActionProposal | GoalProposal | GoalControlProposal
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
        if expected == "integer" and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            raise ValueError(f"slot {name} must be an integer")
        if expected == "boolean" and not isinstance(value, bool):
            raise ValueError(f"slot {name} must be a boolean")
        if expected == "object" and not isinstance(value, dict):
            raise ValueError(f"slot {name} must be an object")
