"""Event-driven bridge between sustained tasks and the SkillClient boundary."""

from __future__ import annotations

import uuid
from typing import Literal

from hey_robot.cognition.runtime.agent_task_store import AgentTaskStep, AgentTaskStore
from hey_robot.cognition.tools.skill_tools import SkillCallProposal
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

    async def reconcile_active_runs(self) -> tuple[AgentTaskStep, ...]:
        """将 transport 已知 event 回填到持久化的 non-terminal step。

        transport 不知道该 run 时可以返回 ``None``，此时 step 保持 pending/running。
        持有 durable receipt 的 Worker 会把失去执行 ownership 的 non-terminal run 收敛为
        ``failed/execution_lost``，Agent process 不会重放 physical robot command。
        """
        return tuple(step for _event, step in await self.reconcile_active_run_events())

    async def reconcile_active_run_events(
        self,
    ) -> tuple[tuple[SkillEvent, AgentTaskStep], ...]:
        """返回已回填的 event/step，供 Agent startup resume 使用。"""
        reconciled: list[tuple[SkillEvent, AgentTaskStep]] = []
        for step in self._tasks.active_skill_steps():
            if step.run_id is None:
                continue
            event = await self._skills.status(step.run_id)
            if event is None:
                continue
            applied = self.apply(event)
            if applied is not None:
                reconciled.append((event, applied))
        return tuple(reconciled)

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
        status: Literal["completed", "failed", "cancelled"]
        if event.phase == "completed":
            status = "completed"
        elif event.phase == "failed":
            status = "failed"
        elif event.phase == "cancelled":
            status = "cancelled"
        else:
            return None
        outcome = ToolOutcome(
            "completed" if event.result.success else "failed",
            event.result.summary,
            {
                **event.result.data,
                "evidence_ids": list(event.result.evidence_ids),
                "artifacts": [
                    {
                        "uri": artifact.uri,
                        "artifact_type": artifact.artifact_type,
                        "role": artifact.role,
                    }
                    for artifact in event.result.artifacts
                ],
            },
            operation_id=event.run_id,
            retryable=event.result.failure_mode in {"timeout", "unavailable"},
        )
        step = self._tasks.apply_skill_event(
            event.run_id,
            outcome=outcome,
            status=status,
            event_sequence=event.sequence,
        )
        if (
            step is not None
            and event.result.success
            and event.result.data.get("termination_reason") == "environment_done"
        ):
            self._tasks.complete_from_environment(
                step.task_id,
                recap=event.result.summary or "Environment reported task completion.",
            )
        return step
