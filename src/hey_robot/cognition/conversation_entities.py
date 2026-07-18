"""Conversation 使用的帧级受信实体引用。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from hey_robot.protocol import RobotObservation, SceneEntity


class EntityResolutionError(ValueError):
    """用户目标无法映射到唯一的受信实体。"""


@dataclass(frozen=True)
class ResolvedEntity:
    """已验证、可供 Goal Template 使用的实体引用。"""

    entity: SceneEntity | None
    target_id: str


class EntityResolver:
    """仅按实体 ID 或配置别名解析目标，不解释自然语言或领域类型。"""

    def __init__(
        self,
        known_entities: tuple[str, ...],
        *,
        aliases: dict[str, str] | None = None,
        max_age_sec: float = 20.0,
    ) -> None:
        self._known_entities = frozenset(known_entities)
        self._aliases = {
            alias.strip(): entity_id
            for alias, entity_id in (aliases or {}).items()
            if alias.strip() and entity_id in self._known_entities
        }
        self._max_age_sec = max(1.0, max_age_sec)
        self._latest: dict[str, tuple[float, tuple[SceneEntity, ...]]] = {}

    def update(self, observation: RobotObservation) -> None:
        """接收新观测；空实体列表同样会使旧视觉引用失效。"""
        robot_id = observation.envelope.robot_id
        if robot_id:
            self._latest[robot_id] = (
                time.monotonic(),
                tuple(
                    entity
                    for entity in observation.entities
                    if entity.frame_id <= observation.frame_id
                ),
            )

    def context(self, robot_id: str | None) -> str:
        """向 Conversation 模型提供当前可直接引用的受信实体。"""
        if not robot_id:
            return "当前没有可用机器人的实体观测。"
        entities = self._fresh_entities(robot_id)
        if not entities:
            return "当前没有可用于目标解析的受信视觉实体。"
        return (
            "当前受信视觉实体如下。它们只能作为观察证据上下文使用；"
            "持续任务创建不要求预先解析为 entity_id。\n"
            + json.dumps(
                [
                    {
                        "entity_id": entity.entity_id,
                        "type": entity.entity_type,
                        "attributes": entity.attributes,
                        "relations": [
                            {
                                "predicate": relation.predicate,
                                "object_id": relation.object_id,
                            }
                            for relation in entity.relations
                        ],
                        "frame_id": entity.frame_id,
                    }
                    for entity in entities
                ],
                ensure_ascii=False,
            )
        )

    def resolve(self, reference: str, *, robot_id: str) -> ResolvedEntity:
        """验证一个精确实体 ID 或声明式别名。"""
        if reference in self._known_entities:
            return ResolvedEntity(None, reference)
        alias = self._aliases.get(reference)
        if alias is not None:
            return ResolvedEntity(None, alias)
        matches = [
            entity
            for entity in self._fresh_entities(robot_id)
            if entity.entity_id == reference
        ]
        if len(matches) == 1:
            return ResolvedEntity(matches[0], matches[0].entity_id)
        raise EntityResolutionError(
            "目标无法映射为唯一的受信实体。请先观察，或使用当前实体清单中的精确 ID。"
        )

    def _fresh_entities(self, robot_id: str) -> tuple[SceneEntity, ...]:
        observed_at, entities = self._latest.get(robot_id, (0.0, ()))
        return entities if time.monotonic() - observed_at <= self._max_age_sec else ()
