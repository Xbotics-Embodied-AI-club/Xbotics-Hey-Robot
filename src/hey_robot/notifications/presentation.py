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
    text = f"[{severity}] {kind}: {present_notification_text(reply.text)}".strip()
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


def present_notification_text(text: str) -> str:
    raw = str(text or "").strip()
    lowered = raw.lower()
    if raw.startswith("任务监督告警："):
        raw = raw.split("：", 1)[1].strip()
        lowered = raw.lower()
    if "last execution feedback failed" in lowered:
        return "刚才的动作结果没有被系统可靠确认。我已暂停继续动作，请选择重新观察、重试或取消。"
    if "skill timed out" in lowered or "timed out after" in lowered:
        return "刚才的动作确认超时。我不会继续执行新的动作，建议先重新观察或重试。"
    if "camera blocked" in lowered or "camera degraded" in lowered:
        return "当前相机画面不可用，无法安全继续移动。请检查相机或让我重新观察。"
    return raw


def should_deliver_notification(reply: AgentReply, allowed_levels: set[str]) -> bool:
    if not is_notification(reply):
        return True
    return notification_severity(reply) in allowed_levels
