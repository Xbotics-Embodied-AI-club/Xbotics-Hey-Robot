from __future__ import annotations

from hey_robot.protocol import AgentReply

_SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}


def is_notification(reply: AgentReply) -> bool:
    return bool(reply.metadata.get("notification"))


def notification_severity(reply: AgentReply) -> str:
    value = str(reply.metadata.get("severity") or "info").strip().lower()
    return value if value in _SEVERITY_ORDER else "info"


def notification_kind(reply: AgentReply) -> str:
    return (
        str(reply.metadata.get("notification_kind") or "notification").strip()
        or "notification"
    )


def format_notification_text(reply: AgentReply) -> str:
    if not is_notification(reply):
        return reply.text
    severity = notification_severity(reply).upper()
    kind = notification_kind(reply).replace("_", " ")
    text = f"[{severity}] {kind}: {reply.text}".strip()
    active_task = str(reply.metadata.get("active_task") or "").strip()
    strategy = str(reply.metadata.get("recovery_strategy") or "").strip()
    next_step = str(reply.metadata.get("recovery_next_step") or "").strip()
    continuation_goal = str(reply.metadata.get("continuation_goal") or "").strip()
    if active_task:
        text = f"{text} [任务: {active_task}]"
    if strategy:
        text = f"{text} [恢复策略: {strategy}]"
    if next_step:
        text = f"{text} 下一步: {next_step}"
    if continuation_goal and continuation_goal != active_task:
        text = f"{text} 恢复后继续: {continuation_goal}"
    return text


def should_deliver_notification(reply: AgentReply, allowed_levels: set[str]) -> bool:
    if not is_notification(reply):
        return True
    return notification_severity(reply) in allowed_levels
