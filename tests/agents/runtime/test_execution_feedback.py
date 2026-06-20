from __future__ import annotations

from hey_robot.agents.runtime.execution_feedback import (
    parse_execution_feedback_response,
)


def test_parse_execution_feedback_response_handles_embedded_json_and_invalid_confidence() -> (
    None
):
    result = parse_execution_feedback_response(
        'analysis: {"success": true, "full_task_complete": true, "missing_progress": "done", '
        '"next_step": "stop", "failure_reason": "none", "confidence": "bad"}'
    )

    assert result.subgoal_success is True
    assert result.task_success is True
    assert result.summary == "done"
    assert result.next_hint == "stop"
    assert result.failure_reason == "none"
    assert result.confidence is None


def test_parse_execution_feedback_response_handles_plain_text_and_empty_input() -> None:
    success = parse_execution_feedback_response("SUCCESS: task looks good")
    empty = parse_execution_feedback_response("")

    assert success.subgoal_success is True
    assert success.task_success is False
    assert success.summary == "SUCCESS: task looks good"

    assert empty.subgoal_success is False
    assert empty.task_success is False
    assert empty.summary == "execution feedback unavailable"
