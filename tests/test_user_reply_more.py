from hey_robot.user_reply import (
    looks_like_internal_user_reply,
    present_runtime_event_for_user,
    present_tool_result_for_user,
)


def test_request_skill_internal_results_get_user_fallbacks() -> None:
    inspect_failed = present_tool_result_for_user(
        tool="request_skill",
        args={"skill": "inspect_scene"},
        result='{"success": false}',
        success=False,
    )
    assert inspect_failed
    assert "success" not in inspect_failed

    completed = present_tool_result_for_user(
        tool="request_skill",
        args={"skill": "move_base"},
        result="skill completed",
        success=True,
    )
    failed = present_tool_result_for_user(
        tool="request_skill",
        args={"skill": "move_base"},
        result="skill completed",
        success=False,
    )

    assert completed
    assert failed
    assert completed != failed
    assert "skill completed" not in completed
    assert "skill completed" not in failed


def test_perception_payload_and_json_tool_results_are_user_facing() -> None:
    assert (
        present_tool_result_for_user(
            tool="request_perception",
            args={},
            result='{"evidence": {"summary": "desk ahead"}}',
            success=True,
        )
        == "desk ahead"
    )
    no_image = present_tool_result_for_user(
        tool="request_perception",
        args={},
        result='{"evidence": {"status": "no_image"}}',
        success=False,
    )
    assert no_image
    assert "no_image" not in no_image
    assert (
        present_tool_result_for_user(
            tool="get_robot_status",
            args={},
            result='{"summary": "battery normal"}',
            success=True,
        )
        == "battery normal"
    )


def test_runtime_event_presentation_filters_internal_lifecycle() -> None:
    assert present_runtime_event_for_user(kind="robot.status", payload={}) is None
    assert present_runtime_event_for_user(
        kind="skill.lifecycle",
        payload={"name": "human_follow", "ux": {"phase": "following"}},
    )
    assert (
        present_runtime_event_for_user(
            kind="skill.lifecycle",
            payload={"name": "move_base", "phase": "failed", "summary": "blocked"},
        )
        == "blocked"
    )
    assert (
        present_runtime_event_for_user(
            kind="skill.lifecycle",
            payload={"name": "move_base", "phase": "completed"},
        )
        is None
    )


def test_internal_reply_detector_covers_protocol_markers() -> None:
    for text in [
        "issued request_skill",
        "subgoal_success: True",
        "task continuation: continue",
        "inspect_scene completed",
    ]:
        assert looks_like_internal_user_reply(text)
