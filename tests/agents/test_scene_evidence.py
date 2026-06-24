from __future__ import annotations

from typing import Any

from hey_robot.agents.runtime.state import ToolCallRecord
from hey_robot.agents.scene_evidence import (
    is_scene_observation_evidence_record,
    reusable_scene_evidence_result,
)


def _parse_feedback(text: str) -> dict[str, Any] | None:
    prefix = "Execution feedback for skill "
    if not text.startswith(prefix):
        return None
    parsed: dict[str, Any] = {}
    for line in text.splitlines()[1:]:
        line = line.strip()
        if not line.startswith("- "):
            continue
        key, sep, value = line[2:].partition(":")
        if not sep:
            continue
        value = value.strip()
        parsed[key.strip()] = (
            True
            if value == "True"
            else False
            if value == "False"
            else None
            if value == "None"
            else value
        )
    return parsed


def test_scene_evidence_reuses_request_perception_for_inspect_scene() -> None:
    records = [
        ToolCallRecord(
            name="request_perception",
            arguments={"question": "what is ahead"},
            result="front camera sees a desk",
            success=True,
        )
    ]

    result = reusable_scene_evidence_result(
        records, "inspect_scene", parse_feedback=_parse_feedback
    )

    assert result is not None
    assert "front camera sees a desk" in result
    assert "Perception result reused" in result


def test_scene_evidence_reuses_successful_look_around_for_inspect_scene() -> None:
    records = [
        ToolCallRecord(
            name="request_capability",
            arguments={"capability": "look_around"},
            result=(
                "Execution feedback for skill skill_seen:\n"
                "- outcome: confirmed\n"
                "- subgoal_success: True\n"
                "- task_success: None\n"
                "- summary: looked around"
            ),
            success=True,
        )
    ]

    result = reusable_scene_evidence_result(
        records, "inspect_scene", parse_feedback=_parse_feedback
    )

    assert result is not None
    assert "looked around" in result


def test_scene_evidence_does_not_reuse_failed_perception_feedback() -> None:
    records = [
        ToolCallRecord(
            name="request_capability",
            arguments={"capability": "inspect_scene"},
            result=(
                "Execution feedback for skill skill_failed:\n"
                "- outcome: failed\n"
                "- subgoal_success: False\n"
                "- task_success: False\n"
                "- summary: camera unavailable"
            ),
            success=True,
        )
    ]

    assert (
        reusable_scene_evidence_result(
            records, "inspect_scene", parse_feedback=_parse_feedback
        )
        is None
    )


def test_scene_evidence_ignores_non_perception_requests_and_records() -> None:
    motion = ToolCallRecord(
        name="request_capability",
        arguments={"capability": "move_base"},
        result="moved forward",
        success=True,
    )

    assert is_scene_observation_evidence_record(motion) is False
    assert (
        reusable_scene_evidence_result(
            [motion], "move_base", parse_feedback=_parse_feedback
        )
        is None
    )
