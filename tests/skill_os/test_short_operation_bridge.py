from __future__ import annotations

import pytest

from hey_robot.protocol import ActionProposal, Envelope, ShortOperationCommand, Topics
from hey_robot.protocol.messages import to_payload
from hey_robot.skill_os.controller import (
    SkillControllerService,
    _short_operation_intent,
)


def test_short_operation_maps_to_skill_intent() -> None:
    command = ShortOperationCommand(
        envelope=Envelope(robot_id="sim_robot"),
        operation_id="conversation_skill_1",
        proposal=ActionProposal(
            "skill",
            "move_base",
            "move forward",
            {"direction": "forward", "distance_cm": 20},
        ),
        timeout_sec=12.0,
    )

    intent = _short_operation_intent(command)

    assert intent.skill_id == "conversation_skill_1"
    assert intent.task_id == "conversation_skill_1"
    assert intent.name == "move_base"
    assert intent.intent_kind == "skill"
    assert intent.arguments == {"direction": "forward", "distance_cm": 20}
    assert intent.objective == "move forward"
    assert intent.timeout_sec == 12.0


@pytest.mark.asyncio
async def test_short_operation_handler_forwards_to_skill_intent() -> None:
    service = object.__new__(SkillControllerService)
    service.topics = Topics()
    forwarded = []

    async def on_skill_intent(topic, payload):
        forwarded.append((topic, payload))

    service._on_skill_intent = on_skill_intent
    command = ShortOperationCommand(
        envelope=Envelope(robot_id="sim_robot"),
        operation_id="conversation_skill_2",
        proposal=ActionProposal(
            "observation",
            "inspect_scene",
            "inspect current scene",
            {"question": "what is visible"},
        ),
    )

    await service._on_short_operation(
        service.topics.short_operation_command, to_payload(command)
    )

    assert forwarded[0][0] == service.topics.skill_intent
    assert forwarded[0][1]["skill_id"] == "conversation_skill_2"
    assert forwarded[0][1]["name"] == "inspect_scene"
