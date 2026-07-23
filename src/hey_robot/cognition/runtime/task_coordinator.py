"""Event-driven bridge between sustained tasks and the SkillClient boundary."""

from __future__ import annotations

import uuid

from hey_robot.cognition.runtime.agent_task_store import AgentTaskStep, AgentTaskStore
from hey_robot.cognition.tools.skill_tools import (
    SkillCallProposal,
    skill_call_from_legacy,
)
from hey_robot.protocol import Envelope, ToolOutcome
from hey_robot.skills.client import SkillClient
from hey_robot.skills.models import SkillCommand, SkillEvent


class TaskCoordinator:
    """Persist first, submit once, and apply SkillEvents idempotently."""

    def __init__(self, tasks: AgentTaskStore, skills: SkillClient) -> None:
        self._tasks = tasks
        self._skills = skills

    async def submit(
        self,
        *,
        task_id: str,
        proposal: SkillCallProposal,
        envelope: Envelope,
        tool_call_id: str,
        deadline_at: float | None = None,
    ) -> AgentTaskStep:
        if not isinstance(proposal, SkillCallProposal) and (
            proposal.__class__.__name__ == "ActionProposal"
        ):
            proposal = skill_call_from_legacy(proposal)
        if not isinstance(proposal, SkillCallProposal):
            raise TypeError(f"unsupported skill proposal: {type(proposal)!r}")
        task = self._tasks.task(task_id)
        if task is None or task.status != "active":
            raise ValueError("cannot submit a skill for a non-active task")
        run_id = f"run_{uuid.uuid4().hex}"
        step = self._tasks.start_skill_step(
            task_id,
            run_id=run_id,
            tool_call_id=tool_call_id,
            tool_name=proposal.name,
            arguments=proposal.arguments,
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
        await self._skills.submit(command)
        return step

    def apply(self, event: SkillEvent) -> AgentTaskStep | None:
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
        status = event.phase
        outcome = ToolOutcome(
            "completed" if event.result.success else "failed",
            event.result.summary,
            {**event.result.data, "evidence_ids": list(event.result.evidence_ids)},
            operation_id=event.run_id,
            retryable=event.result.failure_mode in {"timeout", "unavailable"},
        )
        return self._tasks.apply_skill_event(
            event.run_id,
            outcome=outcome,
            status=status,
            event_sequence=event.sequence,
        )
