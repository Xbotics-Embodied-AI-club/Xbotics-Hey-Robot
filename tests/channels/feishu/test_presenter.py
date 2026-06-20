from __future__ import annotations

import json

from hey_robot.channels.feishu.presenter import format_outbound_reply
from hey_robot.protocol import AgentReply, Envelope


def test_presenter_formats_short_reply_as_text() -> None:
    msg_type, content = format_outbound_reply(
        AgentReply(
            envelope=Envelope(channel="feishu", chat_id="oc_chat_1"), text="robot ready"
        )
    )

    assert msg_type == "text"
    assert json.loads(content) == {"text": "robot ready"}


def test_presenter_formats_notification_as_card() -> None:
    msg_type, content = format_outbound_reply(
        AgentReply(
            envelope=Envelope(channel="feishu", chat_id="oc_chat_1"),
            text="watchdog stale",
            metadata={
                "notification": True,
                "severity": "warning",
                "notification_kind": "task_watchdog",
            },
        )
    )

    card = json.loads(content)
    assert msg_type == "interactive"
    assert card["header"]["template"] == "yellow"
    assert "task watchdog" in card["elements"][0]["content"]


def test_presenter_formats_recovery_notification_details() -> None:
    msg_type, content = format_outbound_reply(
        AgentReply(
            envelope=Envelope(channel="feishu", chat_id="oc_chat_1"),
            text="任务需要恢复。",
            metadata={
                "notification": True,
                "severity": "critical",
                "notification_kind": "task_update",
                "active_task": "pick up cup",
                "continuation_goal": "pick up cup",
                "recovery_strategy": "safe_abort",
                "recovery_next_step": "停止机器人，并请操作员恢复系统状态。",
            },
        )
    )

    card = json.loads(content)
    assert msg_type == "interactive"
    assert any("当前任务" in element["content"] for element in card["elements"])
    assert any("safe_abort" in element["content"] for element in card["elements"])
    assert any("下一步" in element["content"] for element in card["elements"])


def test_presenter_formats_markdown_table_as_card_table() -> None:
    msg_type, content = format_outbound_reply(
        AgentReply(
            envelope=Envelope(channel="feishu", chat_id="oc_chat_1"),
            text="status\n| item | value |\n| --- | --- |\n| arm | ok |",
        )
    )

    card = json.loads(content)
    assert msg_type == "interactive"
    assert any(element["tag"] == "table" for element in card["elements"])
