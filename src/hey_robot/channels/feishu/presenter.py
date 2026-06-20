from __future__ import annotations

import json
import re
from typing import Any

from hey_robot.notifications import (
    is_notification,
    notification_kind,
    notification_severity,
)
from hey_robot.protocol import AgentReply


def format_outbound_reply(reply: AgentReply) -> tuple[str, str]:
    text = (reply.text or "").strip()
    if not is_notification(reply):
        return format_outbound_message(text)
    severity = notification_severity(reply)
    kind = notification_kind(reply).replace("_", " ")
    strategy = str(reply.metadata.get("recovery_strategy") or "").strip()
    next_step = str(reply.metadata.get("recovery_next_step") or "").strip()
    active_task = str(reply.metadata.get("active_task") or "").strip()
    continuation_goal = str(reply.metadata.get("continuation_goal") or "").strip()
    elements = [
        {"tag": "markdown", "content": f"**{kind}**"},
        {"tag": "markdown", "content": text or kind},
    ]
    if active_task:
        elements.append({"tag": "markdown", "content": f"**当前任务：** {active_task}"})
    if strategy:
        elements.append({"tag": "markdown", "content": f"**恢复策略：** `{strategy}`"})
    if next_step:
        elements.append({"tag": "markdown", "content": f"**下一步：** {next_step}"})
    if continuation_goal and continuation_goal != active_task:
        elements.append(
            {"tag": "markdown", "content": f"**恢复后继续：** {continuation_goal}"}
        )
    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": notification_template(severity),
            "title": {
                "tag": "plain_text",
                "content": f"{severity.upper()} notification",
            },
        },
        "elements": elements,
    }
    return "interactive", json.dumps(card, ensure_ascii=False)


def format_outbound_message(text: str) -> tuple[str, str]:
    kind = detect_msg_format(text)
    if kind == "text":
        return "text", json.dumps({"text": text}, ensure_ascii=False)
    if kind == "post":
        return "post", markdown_to_post(text)
    card = json.dumps(
        {"config": {"wide_screen_mode": True}, "elements": build_card_elements(text)},
        ensure_ascii=False,
    )
    return "interactive", card


def detect_msg_format(content: str) -> str:
    stripped = content.strip()
    if extract_markdown_table(stripped) is not None:
        return "card"
    if (
        len(stripped) <= 500
        and "\n" not in stripped
        and "|" not in stripped
        and "```" not in stripped
    ):
        return "text"
    if len(stripped) <= 4000 and "```" not in stripped and stripped.count("\n") <= 20:
        return "post"
    return "card"


def notification_template(severity: str) -> str:
    return {
        "critical": "red",
        "warning": "yellow",
        "info": "blue",
    }.get(severity, "blue")


def markdown_to_post(content: str) -> str:
    rows: list[list[dict[str, Any]]] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append([{"tag": "text", "text": stripped}])
    payload = {"zh_cn": {"title": "", "content": rows}}
    return json.dumps(payload, ensure_ascii=False)


def build_card_elements(content: str) -> list[dict[str, Any]]:
    elements: list[dict[str, Any]] = []
    table_match = extract_markdown_table(content)
    if table_match is None:
        return [{"tag": "markdown", "content": content}]
    prefix, table_text, suffix = table_match
    if prefix.strip():
        elements.append({"tag": "markdown", "content": prefix.strip()})
    parsed_table = parse_md_table(table_text)
    if parsed_table is not None:
        elements.append(parsed_table)
    else:
        elements.append({"tag": "markdown", "content": table_text})
    if suffix.strip():
        elements.append({"tag": "markdown", "content": suffix.strip()})
    return elements


def extract_markdown_table(content: str) -> tuple[str, str, str] | None:
    pattern = re.compile(r"(^|\n)(\|.+\|\n\|[-:| ]+\|\n(?:\|.+\|\n?)*)", re.MULTILINE)
    match = pattern.search(content)
    if not match:
        return None
    start, end = match.span(2)
    return content[:start], content[start:end], content[end:]


def parse_md_table(table_text: str) -> dict[str, Any] | None:
    lines = [line.strip() for line in table_text.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    headers = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines[2:]]
    return {
        "tag": "table",
        "header": [
            {"text": {"tag": "plain_text", "content": header}} for header in headers
        ],
        "rows": [
            [{"text": {"tag": "plain_text", "content": cell}} for cell in row]
            for row in rows
        ],
    }
