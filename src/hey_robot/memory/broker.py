from __future__ import annotations

from typing import TYPE_CHECKING

from hey_robot.memory.scene import SceneMemoryStore

if TYPE_CHECKING:
    from hey_robot.agents.task_events import RobotTaskEventLog
    from hey_robot.agents.task_run import TaskRun
    from hey_robot.memory.runtime import MemoryRuntime


class MemoryBroker:
    """Unified memory broker that selects relevant context based on task state.

    Replaces the scattered memory composition across MemoryFacade,
    RobotMemoryContextBuilder, and MemoryRuntime with a single decision point
    that answers four runtime questions:
      - What task are we doing?
      - What did we just see?
      - What have we tried?
      - Why did we pause or fail?
    """

    def __init__(
        self,
        *,
        scene_memory: SceneMemoryStore,
        task_events: RobotTaskEventLog,
        ltm_runtime: MemoryRuntime | None = None,
    ) -> None:
        self.scene_memory = scene_memory
        self.task_events = task_events
        self.ltm = ltm_runtime

    def build(
        self,
        *,
        task: TaskRun | None,
        task_text: str = "",
        skill_catalog_context: str | None = None,
        current_context: str | None = None,
        perception_context: str | None = None,
        recovery_context: str | None = None,
        limit: int = 6,
    ) -> str | None:
        parts: list[str] = []
        if skill_catalog_context:
            parts.append(skill_catalog_context)

        status = task.status if task else "idle"

        if status in {"active", "recovering", "feedback_pending"}:
            state_block = self._task_state_block(task)
            if state_block:
                parts.append(state_block)

        if status == "active":
            scene_block = self._scene_block(task, limit=limit)
            if scene_block:
                parts.append(scene_block)
            ltm_block = self._ltm_block(task_text or (task.root_task if task else ""))
            if ltm_block:
                parts.append(ltm_block)

        elif status == "recovering":
            recovery_block = self._recovery_block(task, recovery_context)
            if recovery_block:
                parts.append(recovery_block)

        elif status == "feedback_pending":
            attempt_block = self._last_attempt_block(task)
            if attempt_block:
                parts.append(attempt_block)

        elif status in {"completed", "reported"}:
            summary_block = self._task_summary_block(task)
            if summary_block:
                parts.append(summary_block)

        else:
            ltm_block = self._ltm_block(task_text)
            if ltm_block:
                parts.append(ltm_block)

        if current_context:
            parts.append(current_context)
        if perception_context:
            parts.append(perception_context)

        return "\n\n".join(parts) if parts else None

    def append_scene(self, record, *, task_runs) -> None:
        self.scene_memory.append(record)
        if not record.episode_id:
            return
        task_runs.record_scene_memory(
            record.episode_id,
            summary=record.summary,
            frame_id=record.frame_id,
            metadata={
                "scene_record_id": record.record_id,
                "source": record.metadata.get("source"),
            },
        )

    def _task_state_block(self, task: TaskRun | None) -> str | None:
        if task is None:
            return None
        completed = sum(1 for a in task.attempts if a.success is True)
        failed = sum(1 for a in task.attempts if a.success is False)
        last_attempt = task.attempts[-1] if task.attempts else None
        last_action = ""
        if last_attempt is not None:
            skill_name = last_attempt.skill_id or "unknown"
            status_text = "completed" if last_attempt.success else "failed"
            last_action = f"{skill_name} -> {status_text}"
        recovery_info = ""
        if isinstance(task.recovery, dict):
            strategy = task.recovery.get("strategy", "")
            recovery_info = f"needed ({strategy})" if strategy else "needed"
        else:
            recovery_info = "not_needed"
        lines = [
            "Current task state:",
            f"- task: {task.root_task or 'unknown'}",
            f"- status: {task.status}",
            f"- attempts: {completed} completed, {failed} failed",
        ]
        if last_action:
            lines.append(f"- last_action: {last_action}")
        lines.append(f"- recovery: {recovery_info}")
        return "\n".join(lines)

    def _scene_block(self, task: TaskRun | None, *, limit: int) -> str | None:
        if task is None or not task.episode_id:
            return None
        return self.scene_memory.prompt_context(task.episode_id, limit=limit)

    def _ltm_block(self, task_text: str) -> str | None:
        if not task_text or self.ltm is None:
            return None
        return self.ltm.build_agent_context(task_text, limit=4)

    def _recovery_block(
        self, task: TaskRun | None, recovery_context: str | None
    ) -> str | None:
        parts: list[str] = []
        if recovery_context:
            parts.append(recovery_context)
        if task is not None and task.failure_reason:
            parts.append(f"Failure reason: {task.failure_reason}")
        return "\n".join(parts) if parts else None

    def _last_attempt_block(self, task: TaskRun | None) -> str | None:
        if task is None:
            return None
        for attempt in reversed(task.attempts):
            if attempt.skill_id:
                status_text = "completed" if attempt.success else "failed"
                feedback = attempt.metadata.get("execution_feedback", {})
                feedback_summary = (
                    feedback.get("summary", "") if isinstance(feedback, dict) else ""
                )
                lines = [
                    f"Last attempt: {attempt.skill_id} -> {status_text}",
                ]
                if feedback_summary:
                    lines.append(f"Feedback: {feedback_summary}")
                return "\n".join(lines)
        return None

    def _task_summary_block(self, task: TaskRun | None) -> str | None:
        if task is None:
            return None
        return (
            f"Task completed: {task.root_task or 'unknown'}\n"
            f"- success: {task.task_success}\n"
            f"- attempts: {len(task.attempts)}"
        )
