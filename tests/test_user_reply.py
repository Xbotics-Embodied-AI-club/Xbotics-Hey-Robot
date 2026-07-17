from hey_robot.user_reply import (
    _clean_user_text,
    looks_like_internal_user_reply,
    present_runtime_event_for_user,
)


def test_runtime_event_presentation_filters_internal_lifecycle() -> None:
    assert present_runtime_event_for_user(kind="robot.status", payload={}) is None
    assert (
        present_runtime_event_for_user(
            kind="skill.lifecycle",
            payload={"name": "human_follow", "ux": {"phase": "following"}},
        )
        == "我已经看到目标，正在跟随。"
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


def test_runtime_event_failure_has_safe_fallback() -> None:
    assert (
        present_runtime_event_for_user(
            kind="skill.lifecycle",
            payload={"name": "move_base", "phase": "failed"},
        )
        == "技能执行遇到问题，我已经进入恢复状态。"
    )


def test_internal_reply_detector_covers_protocol_markers() -> None:
    for text in (
        "issued request_skill",
        "subgoal_success: True",
        "task continuation: continue",
        "inspect_scene completed",
        '用户说"继续"，回顾一下之前的进展，然后决定下一步。',
    ):
        assert looks_like_internal_user_reply(text)
    assert not looks_like_internal_user_reply("")


def test_clean_user_text_normalizes_runtime_summaries() -> None:
    assert _clean_user_text("None") == ""
    assert _clean_user_text("line1\r\nline2\rline3") == "line1\nline2\nline3"
    assert "机器人不支持" in _clean_user_text("capability not available: arm")
    assert _clean_user_text("base turned right 15.0deg") == "已经向右转了约 15 度。"
