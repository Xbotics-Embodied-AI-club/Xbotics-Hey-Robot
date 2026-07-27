"""Execution boundary for validated Agent tool calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from hey_robot.cognition.runtime.agent_task_store import (
    AgentTask,
    AgentTaskStep,
    AgentTaskStore,
    TaskStatus,
)
from hey_robot.cognition.runtime.task_coordinator import TaskCoordinator
from hey_robot.cognition.tools.models import (
    AgentResponseCall,
    HarnessToolCall,
    PhysicalToolCall,
    PreparedToolCall,
)
from hey_robot.config import DeploymentConfig
from hey_robot.protocol import AgentControl, Envelope, ToolOutcome
from hey_robot.skills.client import SkillClient


@dataclass(frozen=True)
class ToolExecution:
    directive: Literal["continue", "wait", "respond", "finish"]
    outcome: ToolOutcome
    proposal: PreparedToolCall
    step: AgentTaskStep | None = None
    task: AgentTask | None = None
    final_text: str | None = None


class AgentToolExecutor:
    """Execute one prepared call without owning the model loop."""

    def __init__(
        self,
        config: DeploymentConfig,
        tasks: AgentTaskStore,
        coordinator: TaskCoordinator,
        skill_client: SkillClient,
    ) -> None:
        self._config = config
        self._tasks = tasks
        self._coordinator = coordinator
        self._skill_client = skill_client

    async def execute(
        self,
        *,
        session_key: str,
        envelope: Envelope,
        interaction_id: str = "",
        objective: str,
        proposal: PreparedToolCall,
        tool_call_id: str,
    ) -> ToolExecution:
        if isinstance(proposal, PhysicalToolCall):
            return await self._execute_skill(
                session_key,
                envelope,
                interaction_id,
                objective,
                proposal,
                tool_call_id,
            )
        if isinstance(proposal, AgentResponseCall):
            return await self._respond(
                session_key=session_key,
                envelope=envelope,
                interaction_id=interaction_id,
                objective=objective,
                proposal=proposal,
            )
        if isinstance(proposal, HarnessToolCall):
            try:
                outcome = await proposal.execute()
            except Exception as exc:
                outcome = ToolOutcome(
                    "failed",
                    f"普通工具 {proposal.name} 执行失败。",
                    {"failure_mode": "harness_tool_failed", "error": str(exc)},
                    retryable=True,
                )
            return ToolExecution(
                "continue",
                outcome,
                proposal,
            )
        raise TypeError(f"unsupported prepared tool call: {type(proposal)!r}")

    async def _respond(
        self,
        *,
        session_key: str,
        envelope: Envelope,
        interaction_id: str,
        objective: str,
        proposal: AgentResponseCall,
    ) -> ToolExecution:
        task = self._tasks.active_task(session_key)
        if proposal.task_state == "none":
            if task is not None:
                return ToolExecution(
                    "continue",
                    ToolOutcome(
                        "failed",
                        (
                            "当前存在进行中的持续任务；请明确选择 wait、complete "
                            "或 cancel，不能使用 none。"
                        ),
                        {"failure_mode": "active_task_state_required"},
                    ),
                    proposal,
                    task=task,
                )
            return ToolExecution(
                "respond",
                ToolOutcome("completed", proposal.message),
                proposal,
                task=task,
                final_text=proposal.message,
            )
        if proposal.task_state == "wait":
            if task is None:
                import time

                task = self._tasks.create_task(
                    session_key=session_key,
                    envelope=envelope,
                    interaction_id=interaction_id,
                    objective=objective,
                    ui_summary=objective,
                    deadline_at=time.time()
                    + self._config.agent_runtime.hard_max_wall_time_sec,
                )
            return ToolExecution(
                "respond",
                ToolOutcome("completed", proposal.message),
                proposal,
                task=task,
                final_text=proposal.message,
            )
        if proposal.task_state == "complete":
            if task is None:
                return ToolExecution(
                    "finish",
                    ToolOutcome(
                        "failed",
                        "无法完成任务：当前没有进行中的持续任务。",
                        {"failure_mode": "task_not_active"},
                    ),
                    proposal,
                    final_text="无法完成任务：当前没有进行中的持续任务。",
                )
            if self._tasks.active_run_ids(task.task_id):
                return ToolExecution(
                    "respond",
                    ToolOutcome(
                        "failed",
                        "当前机器人操作仍在执行，暂时不能完成任务。",
                        {"failure_mode": "skill_still_running"},
                    ),
                    proposal,
                    task=task,
                    final_text="当前机器人操作仍在执行，暂时不能完成任务。",
                )
            if not self._tasks.has_successful_step(task.task_id):
                return ToolExecution(
                    "respond",
                    ToolOutcome(
                        "failed",
                        "当前没有可信的机器人执行结果，不能把任务标记为完成。",
                        {"failure_mode": "task_evidence_missing"},
                    ),
                    proposal,
                    task=task,
                    final_text="当前没有可信的机器人执行结果，不能把任务标记为完成。",
                )
            self._tasks.close_task(task.task_id, recap=proposal.message)
            return ToolExecution(
                "finish",
                ToolOutcome("completed", proposal.message),
                proposal,
                task=self._tasks.task(task.task_id),
                final_text=proposal.message,
            )
        if proposal.task_state == "cancel":
            if task is None:
                return ToolExecution(
                    "finish",
                    ToolOutcome(
                        "failed",
                        "无法取消任务：当前没有进行中的持续任务。",
                        {"failure_mode": "task_not_active"},
                    ),
                    proposal,
                    final_text="无法取消任务：当前没有进行中的持续任务。",
                )
            await self._apply_control(
                session_key,
                "cancel",
                proposal.message,
                agent_id=envelope.agent_id,
            )
            return ToolExecution(
                "finish",
                ToolOutcome("completed", proposal.message),
                proposal,
                task=self._tasks.task(task.task_id),
                final_text=proposal.message,
            )
        raise ValueError(f"unsupported task state: {proposal.task_state!r}")

    async def control(self, command: AgentControl) -> str:
        return await self._apply_control(
            command.session_key,
            command.action,
            command.reason,
            agent_id=command.envelope.agent_id,
        )

    async def _execute_skill(
        self,
        session_key: str,
        envelope: Envelope,
        interaction_id: str,
        objective: str,
        proposal: PhysicalToolCall,
        tool_call_id: str,
    ) -> ToolExecution:
        task = self._tasks.active_task(session_key)
        max_steps = max(1, int(self._config.agent_runtime.hard_max_skills))
        if task is not None and task.step_count >= max_steps:
            text = "任务已暂停：达到最大机器人步骤预算，需要你确认后再继续。"
            self._tasks.control_task(task.task_id, "blocked", text)
            return ToolExecution(
                "finish",
                ToolOutcome("failed", text),
                proposal,
                task=task,
                final_text=text,
            )
        if task is None:
            import time

            task = self._tasks.create_task(
                session_key=session_key,
                envelope=envelope,
                interaction_id=interaction_id,
                objective=objective,
                ui_summary=objective,
                deadline_at=time.time()
                + self._config.agent_runtime.hard_max_wall_time_sec,
            )
        step = await self._coordinator.submit(
            task_id=task.task_id,
            proposal=proposal,
            envelope=envelope,
            tool_call_id=tool_call_id,
            deadline_at=task.deadline_at,
        )
        outcome = step.outcome
        task = self._tasks.task(task.task_id) or task
        if outcome.status in {"accepted", "waiting"}:
            return ToolExecution("wait", outcome, proposal, step=step, task=task)
        if outcome.status == "failed" and not outcome.retryable:
            self._tasks.control_task(
                task.task_id, "blocked", outcome.user_summary or "操作没有完成。"
            )
            return ToolExecution(
                "finish",
                outcome,
                proposal,
                step=step,
                task=task,
                final_text=outcome.user_summary or "这次操作没有完成。",
            )
        return ToolExecution("continue", outcome, proposal, step=step, task=task)

    async def _apply_control(
        self,
        session_key: str,
        action: Literal["pause", "resume", "cancel", "block", "emergency_stop"],
        requested_reason: str,
        *,
        agent_id: str | None,
    ) -> str:
        task = self._tasks.current_task(session_key)
        if action == "resume":
            if task is None:
                return "当前没有进行中的持续任务。"
            if task.status != "paused":
                return "当前任务不在暂停状态。"
            if self._tasks.active_run_ids(task.task_id):
                return "当前机器人操作仍在停止中；收到终态后才能恢复任务。"
            self._tasks.resume_task(task.task_id)
            return requested_reason or "已恢复当前任务。"
        if task is None and action != "emergency_stop":
            return "当前没有进行中的持续任务。"
        reason = (
            requested_reason
            or {
                "pause": "任务已暂停。",
                "cancel": "任务已取消。",
                "block": "任务已阻塞，需要人工确认。",
                "emergency_stop": "已请求紧急停止。",
            }[action]
        )
        if action == "emergency_stop":
            robot_id = (
                task.robot_id
                if task is not None
                else self._config.default_robot_id(agent_id)
            )
            if robot_id is None:
                return "无法执行紧急停止：当前没有配置机器人。"
            await self._skill_client.emergency_stop(robot_id, reason=reason)
            if task is not None:
                self._tasks.control_task(task.task_id, "cancelled", reason)
            return reason
        if task is None:
            return "当前没有进行中的持续任务。"
        for run_id in self._tasks.active_run_ids(task.task_id):
            await self._skill_client.cancel(run_id, reason=reason)
        if action == "pause":
            self._tasks.pause_task(task.task_id, reason)
        else:
            status = cast(TaskStatus, "blocked" if action == "block" else "cancelled")
            self._tasks.control_task(task.task_id, status, reason)
        return reason
