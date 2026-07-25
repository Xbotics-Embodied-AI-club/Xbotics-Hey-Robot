"""Event-driven bridge between sustained tasks and the SkillClient boundary."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Literal

from hey_robot.cognition.runtime.agent_task_store import (
    AgentTaskStep,
    AgentTaskStore,
    TaskStatus,
)
from hey_robot.cognition.tools.models import PhysicalToolCall
from hey_robot.protocol import Envelope, ToolOutcome
from hey_robot.skills.client import SkillClient
from hey_robot.skills.models import SkillCommand, SkillEvent


@dataclass(frozen=True)
class AppliedSkillEvent:
    step: AgentTaskStep
    task_status: TaskStatus
    should_resume: bool
    final_text: str | None = None


class TaskCoordinator:
    """Persist first, submit once, and apply SkillEvents idempotently."""

    def __init__(self, tasks: AgentTaskStore, skills: SkillClient) -> None:
        self._tasks = tasks
        self._skills = skills

    async def submit(
        self,
        *,
        task_id: str,
        proposal: PhysicalToolCall,
        envelope: Envelope,
        tool_call_id: str,
        deadline_at: float | None = None,
    ) -> AgentTaskStep:
        if not isinstance(proposal, PhysicalToolCall):
            raise TypeError(f"unsupported skill proposal: {type(proposal)!r}")
        task = self._tasks.task(task_id)
        if task is None or task.status != "active":
            raise ValueError("cannot submit a skill for a non-active task")
        if self._tasks.active_run_ids(task_id):
            raise RuntimeError("task already has an active skill run")
        run_id = f"run_{uuid.uuid4().hex}"
        step = self._tasks.add_pending_step(
            task_id,
            proposal,
            run_id=run_id,
            tool_call_id=tool_call_id,
        )
        command = SkillCommand(
            envelope=envelope.child(robot_id=task.robot_id),
            run_id=run_id,
            task_id=task_id,
            robot_id=task.robot_id,
            name=proposal.name,
            arguments=proposal.arguments,
            deadline_at=deadline_at,
        )
        try:
            await self._skills.submit(command)
        except Exception as exc:
            failed = self._tasks.apply_skill_event(
                run_id,
                outcome=ToolOutcome(
                    "failed",
                    "机器人操作提交失败。",
                    {
                        "failure_mode": "transport_submit_failed",
                        "error": str(exc) or type(exc).__name__,
                    },
                    operation_id=run_id,
                    retryable=True,
                ),
                status="failed",
                event_sequence=1,
            )
            if failed is None:
                raise RuntimeError(
                    f"failed to persist rejected skill submission {run_id}"
                ) from exc
            return failed
        return step

    async def reconcile_active_run_results(
        self,
    ) -> tuple[tuple[SkillEvent, AppliedSkillEvent], ...]:
        """Reconcile active runs while retaining task-level completion semantics."""
        reconciled: list[tuple[SkillEvent, AppliedSkillEvent]] = []
        for step in self._tasks.active_skill_steps():
            if step.run_id is None:
                continue
            event = await self._skills.status(step.run_id)
            if event is None:
                continue
            applied = self.apply_result(event)
            if applied is not None:
                reconciled.append((event, applied))
        return tuple(reconciled)

    def apply(self, event: SkillEvent) -> AgentTaskStep | None:
        applied = self.apply_result(event)
        return applied.step if applied is not None else None

    def apply_result(self, event: SkillEvent) -> AppliedSkillEvent | None:
        step = self._apply_step(event)
        if step is None:
            return None
        task = self._tasks.task(step.task_id)
        if task is None:
            return None
        terminal = event.phase in {"completed", "failed", "cancelled"}
        final_text: str | None = None
        if (
            terminal
            and event.result is not None
            and event.result.success
            and event.result.data.get("termination_reason") == "environment_done"
        ):
            final_text = event.result.summary or "Environment reported task completion."
            self._tasks.complete_from_environment(step.task_id, recap=final_text)
            task = self._tasks.task(step.task_id) or task
        return AppliedSkillEvent(
            step,
            task.status,
            terminal and task.status == "active",
            final_text,
        )

    def _apply_step(self, event: SkillEvent) -> AgentTaskStep | None:
        if event.phase == "accepted":
            return self._tasks.apply_skill_event(
                event.run_id,
                outcome=ToolOutcome(
                    "accepted", event.summary, operation_id=event.run_id
                ),
                status="pending",
                event_sequence=event.sequence,
            )
        if event.phase in {"running", "progress"}:
            return self._tasks.apply_skill_event(
                event.run_id,
                outcome=ToolOutcome(
                    "waiting", event.summary, operation_id=event.run_id
                ),
                status="running",
                event_sequence=event.sequence,
            )
        if event.result is None:
            return None
        status: Literal["completed", "failed", "cancelled"]
        if event.phase == "completed":
            status = "completed"
        elif event.phase == "failed":
            status = "failed"
        elif event.phase == "cancelled":
            status = "cancelled"
        else:
            return None
        outcome = event.result.to_tool_outcome(operation_id=event.run_id)
        return self._tasks.apply_skill_event(
            event.run_id,
            outcome=outcome,
            status=status,
            event_sequence=event.sequence,
        )
