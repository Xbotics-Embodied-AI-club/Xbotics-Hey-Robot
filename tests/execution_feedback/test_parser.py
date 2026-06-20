from __future__ import annotations

from hey_robot.agents.runtime.execution_feedback import (
    parse_execution_feedback_response,
)


def test_parse_execution_feedback_response_keeps_task_success_explicit():
    result = parse_execution_feedback_response(
        """
        {
          "success": true,
          "full_task_complete": false,
          "missing_progress": "the second object is still on the table",
          "next_step": "place the second object in the basket"
        }
        """
    )

    assert result.subgoal_success is True
    assert result.task_success is False
    assert result.summary == "the second object is still on the table"
    assert result.next_hint == "place the second object in the basket"


def test_parse_execution_feedback_response_does_not_promote_subgoal_to_task_success():
    result = parse_execution_feedback_response(
        '{"success": true, "summary": "picked up the cup"}'
    )

    assert result.subgoal_success is True
    assert result.task_success is False
    assert result.summary == "picked up the cup"
