from __future__ import annotations

from hey_robot.agents.interaction import (
    classify_user_interaction,
    interpret_pending_confirmation_reply,
)


def _classify(text: str) -> str:
    return classify_user_interaction(text, robot_busy=True).kind


def test_busy_turn_classifier_handles_chinese_interrupts_and_corrections() -> None:
    assert _classify("先暂停，不要继续") == "interrupt"
    assert _classify("不是这个，往左一点") == "correction"


def test_interpret_pending_confirmation_reply_accepts_json_and_dict() -> None:
    assert (
        interpret_pending_confirmation_reply('{"action":"confirm"}').action == "confirm"
    )
    assert (
        interpret_pending_confirmation_reply({"action": "decline"}).action == "decline"
    )
    assert (
        interpret_pending_confirmation_reply('{"action":"new_task"}').action
        == "new_task"
    )


def test_interpret_pending_confirmation_reply_rejects_invalid_payloads() -> None:
    assert interpret_pending_confirmation_reply("").action == "ignore"
    assert interpret_pending_confirmation_reply("ok").action == "ignore"
    assert (
        interpret_pending_confirmation_reply('{"action":"something_else"}').action
        == "ignore"
    )


def test_interpret_pending_confirmation_reply_rejects_non_dict_json() -> None:
    assert interpret_pending_confirmation_reply("[]").action == "ignore"


def test_classify_user_interaction_returns_follow_up_and_new_task() -> None:
    assert (
        classify_user_interaction("tell me more", robot_busy=True).kind == "follow_up"
    )
    assert (
        classify_user_interaction("start a new task", robot_busy=False).kind
        == "new_task"
    )
