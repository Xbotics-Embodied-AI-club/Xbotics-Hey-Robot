from __future__ import annotations

from typing import cast

import numpy as np

from hey_robot.media import MediaResolver
from hey_robot.protocol import Envelope, ImageRef, RobotObservation
from hey_robot.providers.media import ReasoningMediaResolver


class ImageResolver:
    def __init__(self) -> None:
        self.seen: list[ImageRef] = []

    def resolve_images(self, refs: list[ImageRef]) -> list[np.ndarray]:
        self.seen = refs
        return [
            np.full((2, 3, 3), index, dtype=np.uint8) for index, _ref in enumerate(refs)
        ]


def test_reasoning_media_resolver_returns_empty_images_without_observation() -> None:
    resolver = ImageResolver()

    assert (
        ReasoningMediaResolver(cast(MediaResolver, resolver)).observation_images(None)
        == []
    )
    assert resolver.seen == []


def test_reasoning_media_resolver_limits_and_names_observation_images() -> None:
    resolver = ImageResolver()
    refs = [ImageRef(uri=f"file://frame-{index}.jpg") for index in range(5)]
    observation = RobotObservation(
        envelope=Envelope(robot_id="mock0"), frame_id=7, images=refs
    )

    images = ReasoningMediaResolver(cast(MediaResolver, resolver)).observation_images(
        observation, limit=2
    )

    assert resolver.seen == refs[:2]
    assert [image.name for image in images] == ["observation_0", "observation_1"]
    assert [image.media_type for image in images] == ["image/jpeg", "image/jpeg"]
    assert images[1].data.shape == (2, 3, 3)
    assert int(images[1].data[0, 0, 0]) == 1
