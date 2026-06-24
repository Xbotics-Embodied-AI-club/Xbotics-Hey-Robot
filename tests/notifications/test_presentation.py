from __future__ import annotations

from hey_robot.notifications import (
    format_notification_text,
    is_notification,
    notification_kind,
    notification_severity,
    should_deliver_notification,
)
from hey_robot.notifications.presentation import present_notification_text
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


def test_internal_execution_feedback_failure_not_leaked_to_user() -> None:
    text = present_notification_text("last execution feedback failed")

    assert "last execution feedback failed" not in text
    assert "没有被系统可靠确认" in text


def test_skill_timeout_presented_as_confirmation_timeout() -> None:
    text = present_notification_text("active skill timed out after 12.0s")

    assert "timed out" not in text
    assert "动作确认超时" in text


def test_task_watchdog_prefix_is_removed_from_notification_text() -> None:
    text = present_notification_text("任务监督告警：robot status stale for 34.6s")

    assert text == "robot status stale for 34.6s"
    assert "任务监督告警" not in text
