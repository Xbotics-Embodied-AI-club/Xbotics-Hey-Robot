"""Canonical projection from durable session state to model messages."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from hey_robot.cognition.runtime.agent_task_store import (
    AgentTask,
    AgentTaskStep,
    AgentTaskStore,
)
from hey_robot.cognition.runtime.conversation_store import ConversationStore
from hey_robot.cognition.tools.models import PhysicalToolCall, PreparedToolCall
from hey_robot.model import ModelImage, ModelMessage, ModelToolCall
from hey_robot.protocol import ImageRef, ToolOutcome

LOGGER = logging.getLogger(__name__)

# camera1 = agentview left (main view), camera3 = eye-in-hand (wrist).
_CAMERA_PRIORITY = {"camera1": 0, "camera3": 1, "camera2": 2}


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
        image_resolver: Any | None = None,
    ) -> None:
        self._templates = templates
        self._conversations = conversations
        self._tasks = tasks
        self._image_resolver = image_resolver

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
        images: list[ModelImage] = []
        if isinstance(proposal, PhysicalToolCall):
            content = self.outcome_context(proposal, outcome, step=step, task=task)
            # Attach the latest observation images to perception
            # tool results so the planner can SEE the scene and localize object
            # pixels (a text-only observation alone leaves the planner blind
            # for pixel back-projection).
            if self._image_resolver is not None:
                images = self._resolve_outcome_images(outcome)
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
            ModelMessage(
                role="tool",
                content=content,
                tool_call_id=tool_call_id,
                images=images,
            ),
        )

    def _resolve_outcome_images(self, outcome: ToolOutcome) -> list[ModelImage]:
        """Load observation image refs from a tool outcome as model images."""
        observations = outcome.data.get("observations")
        if not isinstance(observations, list) or not observations:
            return []
        # Prefer the main agentview (camera1) and wrist (camera3); skip the
        # redundant second agentview to bound the image payload.
        ordered = sorted(
            (item for item in observations if isinstance(item, dict)),
            key=lambda item: _CAMERA_PRIORITY.get(str(item.get("camera") or ""), 9),
        )
        refs = []
        for item in ordered[:2]:
            uri = str(item.get("uri") or "")
            if not uri:
                continue
            refs.append(
                ImageRef(
                    uri=uri,
                    camera=item.get("camera"),
                    content_type=item.get("content_type"),
                    width=item.get("width"),
                    height=item.get("height"),
                )
            )
        if not refs:
            return []
        resolver = self._image_resolver
        if resolver is None:
            return []
        try:
            arrays = resolver.resolve_images(refs)
        except Exception:
            return []
        images: list[ModelImage] = []
        for index, array in enumerate(arrays[:3]):
            if array is None:
                continue
            try:
                import numpy as np

                data = np.asarray(array, dtype=np.uint8)
                if data.ndim != 3 or data.shape[2] != 3:
                    continue
                # low detail: 256x256 images already carry enough pixels for
                # object localization; high detail multiplies VLM tokens and
                # stalls the reasoning model on long contexts.
                images.append(
                    ModelImage(
                        data=data,
                        name=f"tool_{refs[index].camera or index}",
                        detail="low",
                    )
                )
            except Exception:
                LOGGER.debug("Skipping an unconvertible tool image", exc_info=True)
        return images

    def outcome_context(
        self,
        proposal: PhysicalToolCall,
        outcome: ToolOutcome,
        *,
        step: AgentTaskStep | None = None,
        task: AgentTask | None = None,
    ) -> str:
        summary = outcome.user_summary or "no user-visible summary"
        result_state = outcome.data.get("decision_state")
        if not isinstance(result_state, dict):
            result_state = {}
        evidence = ""
        if step is not None and step.evidence_ids:
            evidence = "; evidence_ids=" + ",".join(step.evidence_ids)
        context = (
            f"tool_result status={outcome.status}; skill={proposal.name}; "
            f"summary={summary}{evidence}"
        )
        if result_state:
            context += "; result_state=" + json.dumps(
                result_state, ensure_ascii=False, sort_keys=True
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
