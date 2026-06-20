from __future__ import annotations

from hey_robot.media import MediaResolver
from hey_robot.protocol import RobotObservation
from hey_robot.providers.types import ReasoningImage


class ReasoningMediaResolver:
    def __init__(self, media_resolver: MediaResolver) -> None:
        self.media_resolver = media_resolver

    def observation_images(
        self, observation: RobotObservation | None, *, limit: int = 4
    ) -> list[ReasoningImage]:
        if observation is None:
            return []
        images = self.media_resolver.resolve_images(observation.images[:limit])
        return [
            ReasoningImage(data=image, name=f"observation_{index}")
            for index, image in enumerate(images)
        ]
