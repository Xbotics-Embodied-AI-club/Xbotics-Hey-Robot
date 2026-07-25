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
