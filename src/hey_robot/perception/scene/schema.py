from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


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
        return cls(
            summary=str(payload.get("summary") or payload.get("caption") or ""),
            objects=objects,
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
    "task_relevance",
    "risks",
    "next_observation_hint",
    "next_hint",
    "confidence",
}


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
