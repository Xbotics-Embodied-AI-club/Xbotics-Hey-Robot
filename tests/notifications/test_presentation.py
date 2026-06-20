from __future__ import annotations

from hey_robot.notifications import (
    format_notification_text,
    is_notification,
    notification_kind,
    notification_severity,
    should_deliver_notification,
)
from hey_robot.protocol import AgentReply, Envelope


def test_notification_presentation_helpers_cover_defaults_and_filtering() -> None:
    plain = AgentReply(envelope=Envelope(channel="web"), text="done")
    warning = AgentReply(
        envelope=Envelope(channel="web"),
        text="watchdog stale",
        metadata={
            "notification": True,
            "severity": "warning",
            "notification_kind": "task_watchdog",
            "active_task": "pick up cup",
            "continuation_goal": "pick up cup",
            "recovery_strategy": "reobserve",
            "recovery_next_step": "重新观察场景。",
        },
    )
    invalid = AgentReply(
        envelope=Envelope(channel="web"),
        text="unknown",
        metadata={"notification": True, "severity": "unknown"},
    )

    assert is_notification(plain) is False
    assert format_notification_text(plain) == "done"
    assert should_deliver_notification(plain, set()) is True

    assert is_notification(warning) is True
    assert notification_severity(warning) == "warning"
    assert notification_kind(warning) == "task_watchdog"
    assert format_notification_text(warning) == (
        "[WARNING] task watchdog: watchdog stale [任务: pick up cup] [恢复策略: reobserve] 下一步: 重新观察场景。"
    )
    assert should_deliver_notification(warning, {"warning", "critical"}) is True
    assert should_deliver_notification(warning, {"critical"}) is False

    assert notification_severity(invalid) == "info"
    assert notification_kind(invalid) == "notification"
