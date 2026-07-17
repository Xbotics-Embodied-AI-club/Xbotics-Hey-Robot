from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from hey_robot.protocol import SceneEntity, SceneRelation


@dataclass(frozen=True)
class SceneObject:
    name: str
    location: str | None = None
    confidence: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SceneUnderstanding:
    summary: str
    objects: list[SceneObject] = field(default_factory=list)
    entities: list[SceneEntity] = field(default_factory=list)
    task_relevance: str | None = None
    risks: list[str] = field(default_factory=list)
    next_observation_hint: str | None = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["objects"] = [item.to_dict() for item in self.objects]
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SceneUnderstanding:
        objects = [
            SceneObject(
                name=str(item.get("name") or item.get("label") or "unknown"),
                location=item.get("location") or item.get("position"),
                confidence=_float(item.get("confidence"), 0.0),
                attributes={
                    key: value
                    for key, value in item.items()
                    if key
                    not in {"name", "label", "location", "position", "confidence"}
                },
            )
            for item in payload.get("objects", []) or []
            if isinstance(item, dict)
        ]
        entities = [_scene_entity(item) for item in payload.get("entities", []) or []]
        return cls(
            summary=str(payload.get("summary") or payload.get("caption") or ""),
            objects=objects,
            entities=[item for item in entities if item is not None],
            task_relevance=payload.get("task_relevance"),
            risks=_string_list(payload.get("risks")),
            next_observation_hint=payload.get("next_observation_hint")
            or payload.get("next_hint"),
            confidence=_float(payload.get("confidence"), 0.0),
            metadata={
                key: value for key, value in payload.items() if key not in _KNOWN_KEYS
            },
        )


_KNOWN_KEYS = {
    "summary",
    "caption",
    "objects",
    "entities",
    "task_relevance",
    "risks",
    "next_observation_hint",
    "next_hint",
    "confidence",
}


def _scene_entity(value: Any) -> SceneEntity | None:
    if not isinstance(value, dict):
        return None
    entity_id = str(value.get("entity_id") or "").strip()
    entity_type = str(value.get("type") or value.get("entity_type") or "").strip()
    frame_id = value.get("frame_id")
    if not entity_id or not entity_type or not isinstance(frame_id, int):
        return None
    attributes = value.get("attributes")
    relations = value.get("relations")
    return SceneEntity(
        entity_id,
        entity_type,
        frame_id,
        dict(attributes) if isinstance(attributes, dict) else {},
        _scene_relations(relations),
    )


def _scene_relations(value: Any) -> list[SceneRelation]:
    if not isinstance(value, list):
        return []
    relations: list[SceneRelation] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        predicate = str(item.get("predicate") or "").strip()
        object_id = str(item.get("object_id") or "").strip()
        if predicate and object_id:
            relations.append(SceneRelation(predicate, object_id))
    return relations


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]
