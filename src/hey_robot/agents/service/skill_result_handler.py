from __future__ import annotations

from typing import Any

from hey_robot.agents.execution_feedback import ExecutionFeedback
from hey_robot.agents.runtime.grounding import is_perception_skill_name
from hey_robot.agents.service.recovery_notifier import SkillRecoveryNotifier
from hey_robot.logging import HeyRobotLogger
from hey_robot.protocol import SkillResult

logger = HeyRobotLogger(name="agent.skill_result")


class SkillResultHandler:
    """Handle skill result side effects outside the bus service boundary."""

    def __init__(self, service: Any) -> None:
        self.service = service
        self.recovery_notifier = SkillRecoveryNotifier(service)

    async def handle(self, result: SkillResult) -> None:
        svc = self.service
        logger.info(
            f"{svc.agent_id} 收到 skill_result skill={result.skill_id} status={result.status}"
        )
        if result.envelope.robot_id:
            svc.latest_skill_result[result.envelope.robot_id] = result
            if result.status in {"completed", "failed", "interrupted"}:
                svc.turn_sessions.release_robot(result.envelope.robot_id)
        svc.task_runtime.observe_robot_skill_result(result)
        feedback: ExecutionFeedback | None = None
        recovery = svc.task_runtime.decide_recovery(
            task=None,
            result=result,
            status=svc.latest_status.get(result.envelope.robot_id or ""),
        )
        waiting_for_skill = svc.core.is_waiting_for_skill(result.skill_id)
        task_run = None
        if result.status in {"completed", "failed", "interrupted"}:
            task_run = svc.task_runtime.observe_skill_result(result)
            feedback = await svc._evaluate_execution_feedback(result)
            task_run = (
                await svc._commit_execution_feedback(result, feedback) or task_run
            )
            recovery = svc.task_runtime.decide_recovery(
                task=task_run,
                result=result,
                status=svc.latest_status.get(result.envelope.robot_id or ""),
            )
            await self.recovery_notifier.publish_if_needed(result, recovery)
        svc.core.observe_skill_result(result.skill_id, result.status, result.error)
        if feedback is not None:
            svc.core.last_feedback_summary = feedback.summary
            svc.core.last_next_hint = feedback.next_hint
            if svc.core.skill_state.snapshot.skill_id == result.skill_id:
                svc.core.skill_state.mark_feedback_received(feedback.summary)
        agent_result_text = (
            svc.task_runtime.result_text_for_agent(result=result, feedback=feedback)
            if feedback is not None
            else None
        )
        if result.status == "completed" and is_perception_skill_name(result.name):
            scene_text = await svc.scene_runtime.skill_result_text_for_agent(result)
            if scene_text:
                agent_result_text = "\n\n".join(
                    part for part in (agent_result_text, scene_text) if part
                )
        svc.core.resolve_skill(
            result.skill_id,
            agent_result_text
            or (
                feedback.for_agent()
                if feedback is not None
                else result.summary or result.status
            ),
        )
        await svc.notification_runtime.maybe_publish_task_update(
            result,
            feedback=feedback,
            recovery=recovery,
            task=task_run,
            waiting_for_skill=waiting_for_skill,
            base_envelope=svc._current_turn_envelope(),
        )
        state = svc.turn_sessions.update_turn_status(
            result.envelope.trace_id, result.status
        )
        if state is None:
            logger.debug(
                f"{svc.agent_id} 无活跃 turn，skill_result trace={result.envelope.trace_id}"
            )
            return
        logger.debug(
            f"{svc.agent_id} 活跃 turn trace={result.envelope.trace_id} status={state.status}"
        )
        if (
            result.status in {"completed", "failed", "interrupted"}
            and feedback is not None
        ):
            await svc._publish_execution_feedback_event(result, feedback)
