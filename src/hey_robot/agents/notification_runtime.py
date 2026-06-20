from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hey_robot.tasks.recovery import TaskRecoveryDecision

from hey_robot.agents.execution_feedback import ExecutionFeedback
from hey_robot.agents.task_run import TaskRun
from hey_robot.notifications import (
    Notification,
    NotificationService,
    NotificationTarget,
)
from hey_robot.protocol import Envelope, SkillResult


class AgentNotificationRuntime:
    """Notification boundary for agent-originated proactive messages."""

    def __init__(
        self,
        *,
        agent_id: str,
        default_robot: str | None,
        service: NotificationService,
    ) -> None:
        self.agent_id = agent_id
        self.default_robot = default_robot
        self.service = service

    async def publish(
        self,
        text: str,
        *,
        base_envelope: Envelope | None,
        severity: str = "info",
        channel: str | None = None,
        chat_id: str | None = None,
        sender_id: str | None = None,
        message_id: str | None = None,
        reply_to_current: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        body = (text or "").strip()
        if not body:
            raise ValueError("notification text must not be empty")
        metadata = dict(metadata or {})
        explicit = any(value for value in (channel, chat_id, sender_id, message_id))
        episode_id = _episode_id_from(metadata, base_envelope)
        target = NotificationTarget(
            mode="explicit" if explicit else "episode",
            channel=channel,
            chat_id=chat_id,
            sender_id=sender_id,
            message_id=message_id,
            reply_to_current=reply_to_current,
        )
        published = await self.service.publish(
            Notification(
                kind=str(metadata.get("event") or "proactive_message"),
                body=body,
                severity=str(severity).strip().lower() or "info",  # type: ignore[arg-type]
                episode_id=episode_id if not explicit else None,
                robot_id=base_envelope.robot_id
                if base_envelope is not None
                else self.default_robot,
                agent_id=self.agent_id,
                trace_id=base_envelope.trace_id if base_envelope is not None else None,
                origin_envelope=base_envelope,
                target=target,
                metadata=metadata,
                dedupe_key=str(metadata.get("dedupe_key") or "") or None,
            )
        )
        if not published:
            raise ValueError(
                "proactive notification requires resolvable target channel and chat_id"
            )

    async def maybe_publish_task_update(
        self,
        result: SkillResult,
        *,
        feedback: ExecutionFeedback | None,
        recovery: TaskRecoveryDecision,
        task: TaskRun | None,
        waiting_for_skill: bool,
        base_envelope: Envelope | None,
    ) -> None:
        if waiting_for_skill:
            return
        notification = self.task_update_text(
            feedback=feedback, recovery=recovery, task=task
        )
        if notification is None:
            return
        text, status, severity, extra = notification
        await self.publish(
            text,
            base_envelope=base_envelope,
            severity=severity,
            metadata={
                "event": "task_update",
                "task_status": status,
                "episode_id": result.envelope.episode_id,
                "skill_id": result.skill_id,
                "dedupe_key": f"task_update:{status}:{result.envelope.episode_id}:{result.skill_id}",
                **extra,
            },
        )

    @staticmethod
    def task_update_text(
        *,
        feedback: ExecutionFeedback | None,
        recovery: TaskRecoveryDecision,
        task: TaskRun | None,
    ) -> tuple[str, str, str, dict[str, Any]] | None:
        active_task = str(getattr(task, "root_task", "") or "").strip()
        if recovery.needed:
            recovery_meta = dict(recovery.metadata or {})
            user_message = str(recovery_meta.get("user_message") or "").strip()
            next_step = str(recovery_meta.get("next_step") or "").strip()
            summary = str(recovery.summary or "").strip()
            if user_message:
                text = user_message
            elif active_task:
                text = f"任务“{active_task}”需要恢复：{summary or '请先处理当前阻塞。'}"
            else:
                text = f"任务需要恢复：{summary or '请先处理当前阻塞。'}"
            if next_step:
                text = f"{text}\n下一步：{next_step}"
            if active_task and recovery.strategy != "safe_abort":
                text = f"{text}\n恢复后继续原任务：{active_task}"
            return (
                text,
                "recovering",
                "critical" if recovery.operator_required else "warning",
                {
                    "active_task": active_task or None,
                    "continuation_goal": active_task or None,
                    "recovery_strategy": recovery.strategy,
                    "recovery_summary": recovery.summary,
                    "recovery_next_step": next_step or None,
                    "recovery_user_message": user_message or None,
                    "recovery_operator_required": recovery.operator_required,
                    "recovery_actions": list(recovery.actions),
                },
            )
        if feedback is None:
            return None
        if feedback.successful and bool(getattr(task, "task_success", False)):
            return (
                f"任务已完成：{feedback.summary}",
                "completed",
                "info",
                {"active_task": active_task or None},
            )
        if not feedback.successful:
            return (
                f"任务执行失败：{feedback.summary}",
                "failed",
                "warning",
                {
                    "active_task": active_task or None,
                    "continuation_goal": active_task or None,
                    "failure_reason": feedback.failure_reason,
                    "recovery_next_step": feedback.next_hint,
                    "recommended_action": feedback.recommended_action,
                },
            )
        return None


def _episode_id_from(
    metadata: dict[str, Any], base_envelope: Envelope | None
) -> str | None:
    raw_episode_id = metadata.get("episode_id")
    if isinstance(raw_episode_id, str) and raw_episode_id:
        return raw_episode_id
    return base_envelope.episode_id if base_envelope is not None else None
