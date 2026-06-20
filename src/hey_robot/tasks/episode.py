from __future__ import annotations

from typing import TYPE_CHECKING

from hey_robot.agents.task_run import TaskRun, TaskRunStore
from hey_robot.protocol import SkillResult

if TYPE_CHECKING:
    from hey_robot.agents.execution_feedback import ExecutionFeedback


class TaskEpisodeRuntime:
    """Small domain facade over TaskRunStore for task episode writes.

    The store remains the durable implementation. This facade gives the agent a
    product-level task episode vocabulary without leaking storage details.
    For reads, use TaskSessionQueryService which composes from all stores.
    """

    def __init__(self, store: TaskRunStore) -> None:
        self.store = store

    def observe_skill_result(self, result: SkillResult) -> TaskRun | None:
        episode_id = result.envelope.episode_id
        if not episode_id:
            return None
        state = self.store.append_attempt(
            episode_id,
            event="skill_result",
            summary=result.summary or result.status,
            skill_id=result.skill_id,
            metadata={
                "status": result.status,
                "error": result.error,
                "progress": result.progress,
                "steps_executed": result.steps_executed,
            },
        )
        if state is None:
            return None
        if result.status == "completed":
            for attempt in state.attempts:
                if attempt.skill_id == result.skill_id:
                    attempt.status = "completed"
                    attempt.metadata["completion_summary"] = (
                        result.summary or result.status
                    )
                    attempt.success = True
                    break
            state.status = "feedback_pending"
            state.failure_reason = None
        elif result.status in {"failed", "interrupted"}:
            for attempt in state.attempts:
                if attempt.skill_id == result.skill_id:
                    attempt.status = "failed"
                    attempt.metadata["failure_summary"] = (
                        result.error or result.summary or result.status
                    )
                    attempt.success = False
                    break
            state.status = "recovering"
            state.failure_reason = (
                result.error or result.summary or f"skill {result.status}"
            )
            state.recovery_count += 1
            self.store.events.append(
                episode_id=episode_id,
                task_id=state.task_id,
                kind="skill_failure",
                summary=result.error or result.summary or result.status,
                skill_id=result.skill_id,
                metadata={"status": result.status, "error": result.error},
            )
        return self.store.save(state)

    def observe_execution_feedback(
        self, episode_id: str, feedback: ExecutionFeedback
    ) -> TaskRun | None:
        state = self.store.append_attempt(
            episode_id,
            event="execution_feedback",
            summary=feedback.summary,
            skill_id=feedback.skill_id,
            metadata=feedback.to_dict(),
        )
        if state is None:
            return None
        for attempt in state.attempts:
            if attempt.skill_id == feedback.skill_id:
                attempt.status = (
                    "confirmed" if feedback.successful else "feedback_failed"
                )
                attempt.metadata["execution_feedback"] = feedback.to_dict()
                attempt.success = bool(feedback.successful)
                break
        state.last_step_success = feedback.subgoal_success
        if feedback.successful:
            state.status = "active"
            state.failure_reason = None
        else:
            state.status = "recovering"
            state.recovery_count += 1
            state.failure_reason = feedback.failure_reason or feedback.summary
            self.store.events.append(
                episode_id=episode_id,
                task_id=state.task_id,
                kind="feedback_failure",
                summary=feedback.summary,
                skill_id=feedback.skill_id,
                metadata={
                    "outcome": feedback.outcome,
                    "failure_reason": feedback.failure_reason,
                },
            )
        return self.store.save(state)
