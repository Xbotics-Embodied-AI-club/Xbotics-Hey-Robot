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


def test_internal_reply_empty_and_variants() -> None:
    assert not looks_like_internal_user_reply("")
    assert not looks_like_internal_user_reply("   ")
    assert looks_like_internal_user_reply("execution feedback for skill move_base")
    assert looks_like_internal_user_reply(
        "execution feedback for skill inspect_scene: ..."
    )
    assert looks_like_internal_user_reply(
        '用户说"继续"，回顾一下之前的进展，然后决定下一步。'
    )


def test_request_skill_no_skill_name_empty_result() -> None:
    assert (
        present_tool_result_for_user(
            tool="request_skill",
            args={},
            result="",
            success=True,
        )
        is None
    )


def test_request_skill_inspect_scene_failure_empty_result() -> None:
    """Line 98: inspect_scene with failure, result empty/invalid → fallback message."""
    assert (
        present_tool_result_for_user(
            tool="request_skill",
            args={"skill": "inspect_scene"},
            result="",
            success=False,
        )
        == "我暂时没有拿到可用画面，不能可靠描述当前场景。"
    )


def test_request_skill_inspect_scene_non_json_clean() -> None:
    """Line 96: inspect_scene with clean plain-text result returns it."""
    assert (
        present_tool_result_for_user(
            tool="request_skill",
            args={"skill": "inspect_scene"},
            result="something went wrong",
            success=True,
        )
        == "something went wrong"
    )


def test_request_skill_inspect_scene_payload_message_fallback() -> None:
    assert (
        present_tool_result_for_user(
            tool="request_skill",
            args={"skill": "inspect_scene"},
            result='{"message": "partial view"}',
            success=True,
        )
        == "partial view"
    )


def test_request_skill_inspect_scene_payload_no_content() -> None:
    assert (
        present_tool_result_for_user(
            tool="request_skill",
            args={"skill": "inspect_scene"},
            result='{"other": "ignored"}',
            success=True,
        )
        == "我已经看了一下当前画面。"
    )


def test_perception_plain_text_result() -> None:
    assert (
        present_tool_result_for_user(
            tool="request_perception",
            args={},
            result="plain text summary",
            success=True,
        )
        == "plain text summary"
    )


def test_perception_json_with_result_field() -> None:
    assert (
        present_tool_result_for_user(
            tool="request_perception",
            args={},
            result='{"result": "found person"}',
            success=True,
        )
        == "found person"
    )


def test_perception_json_with_summary_field() -> None:
    assert (
        present_tool_result_for_user(
            tool="request_perception",
            args={},
            result='{"summary": "desk and chair"}',
            success=True,
        )
        == "desk and chair"
    )


def test_perception_json_summary_when_no_result() -> None:
    assert (
        present_tool_result_for_user(
            tool="request_perception",
            args={},
            result='{"summary": "summary only"}',
            success=True,
        )
        == "summary only"
    )


def test_runtime_event_skill_lifecycle_fallback_message() -> None:
    assert (
        present_runtime_event_for_user(
            kind="skill.lifecycle",
            payload={"name": "move_base", "phase": "failed"},
        )
        == "技能执行遇到问题，我已经进入恢复状态。"
    )


def test_runtime_event_human_follow_fallback_to_summary() -> None:
    result = present_runtime_event_for_user(
        kind="skill.lifecycle",
        payload={
            "name": "human_follow",
            "phase": "unknown_phase",
            "summary": "custom status text",
        },
    )
    assert result == "custom status text"


def test_clean_user_text_strips_none_null() -> None:
    from hey_robot.user_reply import _clean_user_text

    assert _clean_user_text("None") == ""
    assert _clean_user_text("null") == ""
    assert _clean_user_text("NULL") == ""


def test_clean_user_text_handles_carriage_return() -> None:
    from hey_robot.user_reply import _clean_user_text

    result = _clean_user_text("line1\r\nline2\rline3")
    assert "\r" not in result
    assert "line1" in result
    assert "line2" in result
    assert "line3" in result


def test_clean_user_text_capability_unavailable() -> None:
    from hey_robot.user_reply import _clean_user_text

    result = _clean_user_text("capability not available: arm")
    assert "机器人不支持" in result
    assert "没有继续执行" in result
