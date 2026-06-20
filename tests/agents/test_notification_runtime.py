from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from hey_robot.agents.notification_runtime import AgentNotificationRuntime
from hey_robot.agents.task_run import TaskRun
from hey_robot.tasks import TaskRecoveryDecision


def test_task_update_text_includes_active_task_and_continuation_goal_for_recovery() -> (
    None
):
    recovery = TaskRecoveryDecision(
        needed=True,
        strategy="reobserve",
        summary="target lost",
        operator_required=False,
        metadata={"next_step": "inspect again"},
    )
    task = cast(TaskRun, SimpleNamespace(root_task="pick up cup"))

    notification = AgentNotificationRuntime.task_update_text(
        feedback=None,
        recovery=recovery,
        task=task,
    )

    assert notification is not None
    text, status, severity, metadata = notification
    assert status == "recovering"
    assert severity == "warning"
    assert "pick up cup" in text
    assert "恢复后继续原任务" in text
    assert metadata["active_task"] == "pick up cup"
    assert metadata["continuation_goal"] == "pick up cup"
