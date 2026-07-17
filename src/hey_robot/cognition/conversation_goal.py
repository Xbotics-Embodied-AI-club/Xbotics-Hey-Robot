"""将对话目标通过可注册模板编译为任务契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from hey_robot.protocol import GoalBudgets, SceneEntity, SuccessCriterion

_CRITERION_TYPE = Literal["robot_state", "object_relation", "evidence_present"]
_PREDICATE = Literal["equals", "at", "near", "inside", "held_by", "observed"]
_ENTITY_SOURCE = Literal["robot", "target", "destination"]


@dataclass(frozen=True)
class GoalProposal:
    """用户表达的期望世界状态，而不是可直接执行的机器人动作。"""

    goal_kind: str
    objective: str
    target: str
    destination: str | None = None


@dataclass(frozen=True)
class GoalControlProposal:
    action: Literal["cancel", "emergency_stop", "confirm"]
    condition_id: str | None = None


@dataclass(frozen=True)
class GoalTemplate:
    """一种 Goal 的声明式编译规则，可由领域插件注册。"""

    name: str
    criterion_type: _CRITERION_TYPE
    predicate: _PREDICATE
    subject_source: _ENTITY_SOURCE
    object_source: _ENTITY_SOURCE
    target_relation: str | None = None


class GoalTemplateRegistry:
    """Goal 模板注册表；扩展模板不需要修改编译器控制流。"""

    def __init__(self, templates: tuple[GoalTemplate, ...] = ()) -> None:
        self._templates: dict[str, GoalTemplate] = {}
        for template in templates:
            self.register(template)

    def register(self, template: GoalTemplate) -> None:
        name = template.name.strip()
        if not name or name in self._templates:
            raise ValueError(f"无效或重复的 Goal 模板：{template.name}")
        self._templates[name] = template

    def get(self, name: str) -> GoalTemplate:
        try:
            return self._templates[name]
        except KeyError as exc:
            raise ValueError(f"不支持的任务类型：{name}") from exc

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._templates)


DEFAULT_GOAL_TEMPLATES = (
    GoalTemplate("reach", "object_relation", "at", "robot", "target"),
    GoalTemplate(
        "enter",
        "object_relation",
        "inside",
        "robot",
        "target",
        target_relation="leads_to",
    ),
    GoalTemplate("observe", "evidence_present", "observed", "robot", "target"),
    GoalTemplate("approach", "object_relation", "near", "robot", "target"),
    GoalTemplate("hold", "object_relation", "held_by", "target", "robot"),
    GoalTemplate("place", "object_relation", "at", "target", "destination"),
)


class GoalContractBuilder:
    """根据已注册模板构造不可变、可验证的成功条件。"""

    def __init__(
        self,
        known_entities: tuple[str, ...] = (),
        *,
        templates: GoalTemplateRegistry | None = None,
    ) -> None:
        self._known_entities = frozenset(known_entities)
        self.templates = templates or GoalTemplateRegistry(DEFAULT_GOAL_TEMPLATES)

    @property
    def goal_kinds(self) -> tuple[str, ...]:
        return self.templates.names

    def build(
        self,
        proposal: GoalProposal,
        *,
        robot_id: str,
        target_entity: SceneEntity | None = None,
    ) -> tuple[tuple[SuccessCriterion, ...], GoalBudgets]:
        template = self.templates.get(proposal.goal_kind)
        robot = f"robot:{robot_id}"
        values = {
            "robot": robot,
            "target": proposal.target.strip(),
            "destination": (proposal.destination or "").strip(),
        }
        if not values["target"]:
            raise ValueError("目标不能为空")
        if template.target_relation and target_entity is not None:
            relation_target = _relation_target(target_entity, template.target_relation)
            if relation_target is None:
                raise ValueError(
                    f"实体 {target_entity.entity_id} 缺少关系：{template.target_relation}"
                )
            values["target"] = relation_target

        subject_id = values[template.subject_source]
        object_id = values[template.object_source]
        if not subject_id or not object_id:
            raise ValueError("目标模板缺少必填实体")
        if object_id not in self._known_entities:
            raise ValueError(f"当前无法验证目标：{object_id}")
        if subject_id != robot and subject_id not in self._known_entities:
            raise ValueError(f"当前无法验证目标：{subject_id}")

        return (
            (
                SuccessCriterion(
                    criterion_id="goal_satisfied",
                    criterion_type=cast(_CRITERION_TYPE, template.criterion_type),
                    subject_id=subject_id,
                    predicate=cast(_PREDICATE, template.predicate),
                    object_id=object_id,
                    max_age_sec=30.0,
                ),
            ),
            GoalBudgets(),
        )


def _relation_target(entity: SceneEntity, predicate: str) -> str | None:
    matches = [
        relation.object_id
        for relation in entity.relations
        if relation.predicate == predicate and relation.object_id
    ]
    return matches[0] if len(matches) == 1 else None
