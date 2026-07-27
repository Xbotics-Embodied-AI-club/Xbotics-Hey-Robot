"""Stateful per-session Agent lifecycle built on the stateless AgentRunner."""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal

from hey_robot.cognition.runtime.agent_context import AgentContextBuilder
from hey_robot.cognition.runtime.agent_runner import AgentRunner, AgentTurnRequest
from hey_robot.cognition.runtime.agent_task_store import AgentTask, AgentTaskStore
from hey_robot.cognition.runtime.conversation_store import ConversationStore
from hey_robot.cognition.tools.executor import AgentToolExecutor
from hey_robot.cognition.tools.registry import ToolRegistry
from hey_robot.model import ModelMessage, TextDeltaCallback
from hey_robot.protocol import Envelope

MAX_MODEL_TURNS_PER_WAKEUP = 8


@dataclass(frozen=True)
class AgentCommand:
    session_key: str
    interaction_id: str
    envelope: Envelope
    text: str


@dataclass(frozen=True)
class ResumeTrigger:
    task_id: str
    source: Literal["skill_terminal", "startup_recovery", "user_resume"]
    step_id: str | None = None


@dataclass(frozen=True)
class AgentRunResult:
    status: Literal[
        "responded", "waiting", "completed", "blocked", "cancelled", "failed"
    ]
    text: str
    operation_id: str | None = None


class Agent:
    """One interactive Agent instance bound to one durable session."""

    def __init__(
        self,
        *,
        session_key: str,
        runner: AgentRunner,
        tools: ToolRegistry,
        executor: AgentToolExecutor,
        context: AgentContextBuilder,
        tasks: AgentTaskStore,
        conversations: ConversationStore,
    ) -> None:
        self.session_key = session_key
        self._runner = runner
        self._tools = tools
        self._executor = executor
        self._context = context
        self._tasks = tasks
        self._conversations = conversations
        self._state_lock = asyncio.Lock()
        self._run_task: asyncio.Task[AgentRunResult] | None = None

    async def prompt(
        self,
        command: AgentCommand,
        *,
        on_text_delta: TextDeltaCallback | None = None,
    ) -> AgentRunResult:
        return await self._launch(command, on_text_delta=on_text_delta)

    async def steer(
        self,
        command: AgentCommand,
        *,
        on_text_delta: TextDeltaCallback | None = None,
    ) -> AgentRunResult:
        task = self._tasks.current_task(self.session_key)
        if task is not None:
            if task.status == "active":
                self._tasks.update_route(
                    task.task_id,
                    envelope=command.envelope,
                    interaction_id=command.interaction_id,
                )
            active_runs = self._tasks.active_run_ids(task.task_id)
            if active_runs:
                # Conversation is the only source of user intent. The returned
                # text is transient UI feedback and is not written back into it.
                self._conversations.append(command.session_key, "user", command.text)
                text = "已记录你的修改，将在当前机器人操作结束后从安全点继续。"
                return AgentRunResult("waiting", text, active_runs[0])
        await self.abort_inference()
        return await self._launch(command, on_text_delta=on_text_delta)

    async def resume(self, trigger: ResumeTrigger) -> AgentRunResult:
        task = self._tasks.task(trigger.task_id)
        if task is None:
            return AgentRunResult("failed", "无法恢复：任务不存在。")
        if task.status == "paused" and trigger.source == "user_resume":
            task = self._tasks.resume_task(task.task_id)
        if task.status != "active":
            status: Literal["completed", "cancelled"] = (
                "completed" if task.status == "completed" else "cancelled"
            )
            return AgentRunResult(status, task.final_recap or task.last_error or "")
        sequence = self._tasks.resume_after_sequence(task.task_id)
        steps = self._tasks.recent_steps(task.task_id, limit=200)
        step = next(
            (
                item
                for item in reversed(steps)
                if (trigger.step_id and item.step_id == trigger.step_id)
                or (not trigger.step_id and item.sequence == sequence)
            ),
            steps[-1] if steps else None,
        )
        if step is None:
            return AgentRunResult("failed", "无法恢复：没有可用的步骤结果。")
        envelope = self._tasks.task_envelope(task.task_id)
        if envelope is None:
            return AgentRunResult("failed", "无法恢复：任务路由不完整。")
        interaction_id = f"resume_{task.task_id}_{step.sequence}"
        messages = self._context.for_resume(task, step)
        return await self._launch_drive(
            messages,
            envelope,
            interaction_id,
            task.objective,
        )

    async def abort_inference(self) -> None:
        async with self._state_lock:
            current = self._run_task
            if current is not None and not current.done():
                current.cancel()
        if current is not None and not current.done():
            with suppress(asyncio.CancelledError):
                await current

    async def _launch(
        self,
        command: AgentCommand,
        *,
        on_text_delta: TextDeltaCallback | None,
    ) -> AgentRunResult:
        previous: asyncio.Task[AgentRunResult] | None
        async with self._state_lock:
            previous = self._run_task
        if previous is not None and not previous.done():
            with suppress(asyncio.CancelledError):
                await previous
        active_task = self._tasks.active_task(command.session_key)
        if active_task is not None:
            self._tasks.update_route(
                active_task.task_id,
                envelope=command.envelope,
                interaction_id=command.interaction_id,
            )
        messages = self._context.for_command(command.session_key, command.text)
        self._conversations.append(command.session_key, "user", command.text)
        return await self._launch_drive(
            messages,
            command.envelope,
            command.interaction_id,
            command.text,
            on_text_delta=on_text_delta,
        )

    async def _launch_drive(
        self,
        messages: list[ModelMessage],
        envelope: Envelope,
        interaction_id: str,
        objective: str,
        *,
        on_text_delta: TextDeltaCallback | None = None,
    ) -> AgentRunResult:
        async with self._state_lock:
            existing = self._run_task
            if existing is not None and not existing.done():
                task = existing
            else:
                task = asyncio.create_task(
                    self._drive(
                        messages,
                        envelope,
                        interaction_id,
                        objective,
                        on_text_delta=on_text_delta,
                    )
                )
                self._run_task = task
        try:
            result = await task
        finally:
            async with self._state_lock:
                if self._run_task is task:
                    self._run_task = None
        if result.text and result.status != "waiting":
            self._conversations.append(self.session_key, "assistant", result.text)
        return result

    async def _drive(
        self,
        messages: list[ModelMessage],
        envelope: Envelope,
        interaction_id: str,
        objective: str,
        *,
        on_text_delta: TextDeltaCallback | None,
    ) -> AgentRunResult:
        del on_text_delta
        model_turns = 0
        while True:
            active_task = self._tasks.active_task(self.session_key)
            if (
                active_task is not None
                and active_task.deadline_at is not None
                and time.time() >= active_task.deadline_at
            ):
                text = "任务已暂停：达到最长持续时间，需要你确认后再继续。"
                self._tasks.control_task(active_task.task_id, "blocked", text)
                return AgentRunResult("blocked", text)
            if model_turns >= MAX_MODEL_TURNS_PER_WAKEUP:
                if active_task is None:
                    return AgentRunResult(
                        "failed", "模型没有按结构化回复协议完成这次请求。"
                    )
                text = "任务已暂停：单次唤醒达到模型轮次上限，请确认后继续。"
                self._tasks.pause_task(active_task.task_id, text)
                return AgentRunResult("blocked", text)
            deadline = _next_decision_deadline(active_task)

            decision = await self._runner.run(
                AgentTurnRequest(
                    tuple(messages), self._tools.names, deadline, interaction_id
                ),
            )
            model_turns += 1
            if decision.status == "failed":
                detail = (
                    decision.failure.message if decision.failure else "模型决策失败"
                )
                return AgentRunResult("failed", f"这次请求没有完成：{detail}")
            if decision.status == "returned":
                text = decision.final_text or ""
                messages.extend(
                    (
                        ModelMessage(role="assistant", content=text),
                        ModelMessage(
                            role="user",
                            content=(
                                "普通文本不是有效的 Agent 响应。请使用当前 function "
                                "schema 暴露的结构化响应能力。"
                            ),
                        ),
                    )
                )
                continue
            proposal = decision.proposal
            if proposal is None or not decision.tool_calls:
                return AgentRunResult("failed", "工具没有产生有效调用。")
            call = decision.tool_calls[0]
            execution = await self._executor.execute(
                session_key=self.session_key,
                envelope=envelope,
                interaction_id=interaction_id,
                objective=objective,
                proposal=proposal,  # type: ignore[arg-type]
                tool_call_id=call.tool_call_id,
            )
            messages.extend(
                self._context.tool_result_messages(
                    tool_call_id=call.tool_call_id,
                    tool_name=call.name,
                    arguments=call.arguments,
                    proposal=execution.proposal,
                    outcome=execution.outcome,
                    step=execution.step,
                    task=execution.task,
                )
            )
            if execution.directive == "wait":
                return AgentRunResult(
                    "waiting",
                    "已提交机器人操作，正在等待执行结果。",
                    execution.outcome.operation_id,
                )
            if execution.directive == "respond":
                return AgentRunResult(
                    "responded",
                    execution.final_text
                    or execution.outcome.user_summary
                    or "任务仍在进行中。",
                    execution.task.task_id if execution.task is not None else None,
                )
            if execution.directive == "finish":
                task = execution.task
                result_status: Literal[
                    "completed", "blocked", "cancelled", "failed"
                ] = "completed"
                if task is not None:
                    current = self._tasks.task(task.task_id)
                    if current is not None:
                        if current.status == "blocked":
                            result_status = "blocked"
                        elif current.status == "cancelled":
                            result_status = "cancelled"
                        elif current.status == "failed":
                            result_status = "failed"
                elif execution.outcome.status == "failed":
                    result_status = "failed"
                return AgentRunResult(
                    result_status,
                    execution.final_text
                    or execution.outcome.user_summary
                    or "请求已经处理。",
                    execution.outcome.operation_id,
                )


def _next_decision_deadline(task: AgentTask | None) -> float:
    timeout_sec = 120.0
    if task is not None and task.deadline_at is not None:
        timeout_sec = min(timeout_sec, max(0.001, task.deadline_at - time.time()))
    return time.monotonic() + timeout_sec
