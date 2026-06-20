from __future__ import annotations

from types import SimpleNamespace

from hey_robot.channels.feishu.inbound import (
    extract_post_content,
    extract_share_card_content,
    resolve_mentions,
)


def test_extract_post_content_handles_localized_payload() -> None:
    text, images = extract_post_content(
        {
            "post": {
                "zh_cn": {
                    "title": "status",
                    "content": [
                        [{"tag": "text", "text": "robot ready"}],
                        [{"tag": "img", "image_key": "img_1"}],
                    ],
                }
            }
        }
    )

    assert text == "status robot ready"
    assert images == ["img_1"]


def test_extract_share_card_content_handles_interactive_card() -> None:
    text = extract_share_card_content(
        {
            "header": {"title": {"tag": "plain_text", "content": "Task"}},
            "elements": [{"tag": "markdown", "content": "**done**"}],
        },
        "interactive",
    )

    assert "Task" in text
    assert "**done**" in text


def test_resolve_mentions_replaces_feishu_keys() -> None:
    text = resolve_mentions(
        "@_user_1 move forward",
        [
            SimpleNamespace(
                key="@_user_1", id=SimpleNamespace(open_id="ou_bot"), name="robot"
            )
        ],
    )

    assert text == "@robot (ou_bot) move forward"
