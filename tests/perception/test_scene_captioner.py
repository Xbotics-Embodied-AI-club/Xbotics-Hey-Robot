from __future__ import annotations

import numpy as np

from hey_robot.perception.scene import (
    DeterministicSceneCaptioner,
    ReasoningSceneCaptioner,
    SceneUnderstanding,
)
from hey_robot.protocol import Envelope, ImageRef, RobotObservation, RobotStatus
from hey_robot.providers import ReasoningMessage, ReasoningResponse


class FakeProvider:
    def __init__(self) -> None:
        self.messages: list[ReasoningMessage] = []

    async def chat(self, **kwargs):
        self.messages = list(kwargs.get("messages") or [])
        return ReasoningResponse(
            content=(
                '{"summary":"桌面上有一个杯子","objects":[{"name":"杯子","location":"桌面中央",'
                '"confidence":0.8}],"task_relevance":"目标可见","risks":[],"next_observation_hint":"靠近前保持目标居中",'
                '"confidence":0.7}'
            )
        )

    def get_default_model(self) -> str:
        return "fake-vlm"


class FakeResolver:
    def resolve_images(self, _images):
        return [np.zeros((8, 8, 3), dtype=np.uint8)]


async def test_deterministic_scene_captioner_without_image() -> None:
    observation = RobotObservation(envelope=Envelope(robot_id="xlerobot"), frame_id=3)
    status = RobotStatus(envelope=observation.envelope, state="idle")

    result = await DeterministicSceneCaptioner().caption(observation, status)

    assert "Observed frame 3" in result.summary
    assert result.risks


async def test_reasoning_scene_captioner_parses_structured_scene() -> None:
    provider = FakeProvider()
    observation = RobotObservation(
        envelope=Envelope(robot_id="xlerobot"),
        frame_id=4,
        images=[ImageRef(uri="media://image")],
    )

    result = await ReasoningSceneCaptioner(
        provider, image_resolver=FakeResolver()
    ).caption(observation)  # type: ignore[arg-type]

    assert result.summary == "桌面上有一个杯子"
    assert result.objects[0].name == "杯子"
    assert result.next_observation_hint == "靠近前保持目标居中"
    assert "robot scene captioner" in provider.messages[0].content
    assert "Robot observation frame: 4" in provider.messages[1].content


def test_scene_understanding_accepts_string_risks() -> None:
    result = SceneUnderstanding.from_dict(
        {"summary": "ok", "risks": "no visible hazard"}
    )

    assert result.risks == ["no visible hazard"]
