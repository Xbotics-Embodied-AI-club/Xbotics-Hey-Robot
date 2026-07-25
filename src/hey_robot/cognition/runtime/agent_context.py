"""Canonical projection from durable session state to model messages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from hey_robot.cognition.runtime.agent_task_store import (
    AgentTask,
    AgentTaskStep,
    AgentTaskStore,
)
from hey_robot.cognition.runtime.conversation_store import ConversationStore
from hey_robot.cognition.tools.models import PhysicalToolCall, PreparedToolCall
from hey_robot.model import ModelMessage, ModelToolCall
from hey_robot.protocol import ToolOutcome


class TemplateView(Protocol):
    def render(self, name: str, **context: object) -> str: ...


@dataclass(frozen=True)
class AgentSessionView:
    session_key: str
    transcript: tuple[ModelMessage, ...]
    active_task: AgentTask | None
    recent_steps: tuple[AgentTaskStep, ...]


class AgentContextBuilder:
    """Build one model context for prompts, inline outcomes, and resumes."""

    def __init__(
        self,
        templates: TemplateView,
        conversations: ConversationStore,
        tasks: AgentTaskStore,
    ) -> None:
        self._templates = templates
        self._conversations = conversations
        self._tasks = tasks

    def view(self, session_key: str) -> AgentSessionView:
        task = self._tasks.active_task(session_key)
        steps = self._tasks.recent_steps(task.task_id, limit=6) if task else ()
        return AgentSessionView(
            session_key,
            tuple(self._conversations.recent(session_key)),
            task,
            steps,
        )

    def for_command(self, session_key: str, text: str) -> list[ModelMessage]:
        view = self.view(session_key)
        return [
            ModelMessage(role="system", content=self._policy(view)),
            *view.transcript,
            ModelMessage(role="user", content=text),
        ]

    def for_resume(self, task: AgentTask, step: AgentTaskStep) -> list[ModelMessage]:
        view = self.view(task.session_key)
        tool_call_id = step.tool_call_id or f"resume_{step.step_id}"
        return [
            ModelMessage(role="system", content=self._policy(view)),
            *view.transcript,
            ModelMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ModelToolCall(
                        tool_call_id,
                        step.proposal.name,
                        dict(step.proposal.arguments),
                    )
                ],
            ),
            ModelMessage(
                role="tool",
                content=self.outcome_context(
                    step.proposal, step.outcome, step=step, task=task
                ),
                tool_call_id=tool_call_id,
            ),
        ]

    def tool_result_messages(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, object],
        proposal: PreparedToolCall,
        outcome: ToolOutcome,
        step: AgentTaskStep | None,
        task: AgentTask | None,
    ) -> tuple[ModelMessage, ModelMessage]:
        if isinstance(proposal, PhysicalToolCall):
            content = self.outcome_context(proposal, outcome, step=step, task=task)
        else:
            content = (
                f"tool_result status={outcome.status}; "
                f"summary={outcome.user_summary or ''}; "
                f"data={json.dumps(outcome.data, ensure_ascii=False)}"
            )
        return (
            ModelMessage(
                role="assistant",
                content="",
                tool_calls=[ModelToolCall(tool_call_id, tool_name, arguments)],
            ),
            ModelMessage(role="tool", content=content, tool_call_id=tool_call_id),
        )

    def outcome_context(
        self,
        proposal: PhysicalToolCall,
        outcome: ToolOutcome,
        *,
        step: AgentTaskStep | None = None,
        task: AgentTask | None = None,
    ) -> str:
        summary = outcome.user_summary or "no user-visible summary"
        evidence = ""
        if step is not None and step.evidence_ids:
            evidence = "; evidence_ids=" + ",".join(step.evidence_ids)
        context = (
            f"tool_result status={outcome.status}; skill={proposal.name}; "
            f"summary={summary}{evidence}"
        )
        if task is not None:
            context += f"\nactive_task id={task.task_id}"
        return context

    def _policy(self, view: AgentSessionView) -> str:
        return self._templates.render(
            "agent/SYSTEM.md",
            agent_soul=self._templates.render("agent/SOUL.md"),
            task_context=self._tasks.projection(view.session_key),
        )
