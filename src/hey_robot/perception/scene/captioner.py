from __future__ import annotations

import json
from typing import Any, Protocol

from hey_robot.config import DeploymentConfig
from hey_robot.media import MediaResolver
from hey_robot.perception.scene.schema import SceneObject, SceneUnderstanding
from hey_robot.protocol import RobotObservation, RobotStatus
from hey_robot.providers import (
    ReasoningImage,
    ReasoningMessage,
    ReasoningProvider,
    build_provider,
)
from hey_robot.templates.loader import TemplateStore


class SceneCaptioner(Protocol):
    async def caption(
        self, observation: RobotObservation, status: RobotStatus | None = None
    ) -> SceneUnderstanding: ...


class DeterministicSceneCaptioner:
    async def caption(
        self, observation: RobotObservation, status: RobotStatus | None = None
    ) -> SceneUnderstanding:
        metrics = status.metrics if status is not None else {}
        camera = (
            metrics.get("camera")
            if isinstance(metrics.get("camera"), dict)
            else observation.raw.get("camera")
        )
        battery = (
            metrics.get("battery")
            if isinstance(metrics.get("battery"), dict)
            else observation.raw.get("battery")
        )
        objects = []
        if observation.images:
            objects.append(SceneObject("visual_frame", "front camera", 0.5))
        if isinstance(camera, dict) and camera.get("frame_available"):
            objects.append(
                SceneObject(
                    "camera", "front", 0.7, {"shape": camera.get("image_shape")}
                )
            )
        parts = [
            f"Observed frame {observation.frame_id} with {len(observation.images)} image(s)."
        ]
        if status is not None:
            parts.append(f"Robot state is {status.state}.")
        if isinstance(battery, dict) and battery.get("status"):
            parts.append(f"Battery is {battery.get('status')}.")
        return SceneUnderstanding(
            summary=" ".join(parts),
            objects=objects,
            task_relevance="Use this observation as the current scene state for the active robot task.",
            risks=[]
            if observation.images
            else ["No camera image is available for visual execution feedback."],
            next_observation_hint=None
            if observation.images
            else "Get a fresh observation with a valid camera frame.",
            confidence=0.45 if observation.images else 0.25,
            metadata={"provider": "deterministic"},
        )


class ReasoningSceneCaptioner:
    def __init__(
        self,
        provider: ReasoningProvider,
        *,
        image_resolver: MediaResolver | None = None,
        templates: TemplateStore | None = None,
    ) -> None:
        self.provider = provider
        self.image_resolver = image_resolver
        self.fallback = DeterministicSceneCaptioner()
        self.templates = templates or TemplateStore()

    async def caption(
        self, observation: RobotObservation, status: RobotStatus | None = None
    ) -> SceneUnderstanding:
        images = self._images(observation)
        if not images:
            return await self.fallback.caption(observation, status)
        response = await self.provider.chat(
            messages=[
                ReasoningMessage(
                    role="system",
                    content=self.templates.render("robot/scene_captioner/SYSTEM.md"),
                ),
                ReasoningMessage(
                    role="user",
                    content=self.templates.render(
                        "robot/scene_captioner/USER.md",
                        frame_id=observation.frame_id,
                        task=observation.task or "unknown",
                        robot_status=status.state if status else "unknown",
                    ),
                    images=images,
                ),
            ],
            tools=None,
        )
        if response.finish_reason == "error":
            fallback = await self.fallback.caption(observation, status)
            return SceneUnderstanding(
                summary=fallback.summary,
                objects=fallback.objects,
                task_relevance=fallback.task_relevance,
                risks=[*fallback.risks, response.content or "scene captioning failed"],
                next_observation_hint=fallback.next_observation_hint,
                confidence=fallback.confidence,
                metadata={"provider": "provider", "error": response.content},
            )
        return _parse_scene_understanding(response.content or "")

    def _images(self, observation: RobotObservation) -> list[ReasoningImage]:
        if self.image_resolver is None:
            return []
        return [
            ReasoningImage(data=image, name=f"scene_{index}")
            for index, image in enumerate(
                self.image_resolver.resolve_images(observation.images[:4])
            )
        ]


def build_scene_captioner(
    config: DeploymentConfig,
    agent_id: str,
    *,
    image_resolver: MediaResolver | None = None,
) -> SceneCaptioner:
    agent = config.agents.get(agent_id)
    cfg = {}
    if agent is not None:
        template_root = agent.settings.get("template_root")
        perception = agent.settings.get("perception")
        if isinstance(perception, dict):
            scene = perception.get("scene_captioner")
            if isinstance(scene, dict):
                cfg = scene
                template_root = cfg.get("template_root") or template_root
    else:
        template_root = None
    if not bool(cfg.get("enabled", False)):
        return DeterministicSceneCaptioner()
    provider = build_provider(
        config, agent_id, purpose=str(cfg.get("purpose") or "scene_captioner")
    )
    return ReasoningSceneCaptioner(
        provider,
        image_resolver=image_resolver,
        templates=TemplateStore(template_root),
    )


def _parse_scene_understanding(text: str) -> SceneUnderstanding:
    payload: Any = None
    raw = (text or "").strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            try:
                payload = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                payload = None
    if isinstance(payload, dict):
        parsed = SceneUnderstanding.from_dict(payload)
        if parsed.summary:
            return parsed
    return SceneUnderstanding(
        summary=raw or "scene caption unavailable",
        confidence=0.0,
        metadata={"raw": raw},
    )
