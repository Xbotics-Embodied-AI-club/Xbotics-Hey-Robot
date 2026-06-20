from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from hey_robot.perception.scene import SceneUnderstanding
from hey_robot.protocol import RobotObservation

SceneEvidenceStatus = Literal[
    "ok", "no_observation", "no_image", "stale", "caption_failed"
]


@dataclass(frozen=True)
class SceneEvidence:
    status: SceneEvidenceStatus
    frame_id: int | None = None
    image_count: int = 0
    summary: str = ""
    confidence: float | None = None
    objects: list[dict[str, Any]] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    next_observation_hint: str | None = None
    source: str = "scene_captioner"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "frame_id": self.frame_id,
            "image_count": self.image_count,
            "summary": self.summary,
            "confidence": self.confidence,
            "objects": list(self.objects),
            "risks": list(self.risks),
            "next_observation_hint": self.next_observation_hint,
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_understanding(
        cls,
        observation: RobotObservation,
        understanding: SceneUnderstanding,
        *,
        source: str = "scene_captioner",
        metadata: dict[str, Any] | None = None,
    ) -> SceneEvidence:
        status: SceneEvidenceStatus = "ok" if observation.images else "no_image"
        return cls(
            status=status,
            frame_id=observation.frame_id,
            image_count=len(observation.images),
            summary=understanding.summary,
            confidence=understanding.confidence,
            objects=[item.to_dict() for item in understanding.objects],
            risks=list(understanding.risks),
            next_observation_hint=understanding.next_observation_hint,
            source=source,
            metadata={**dict(understanding.metadata), **dict(metadata or {})},
        )
