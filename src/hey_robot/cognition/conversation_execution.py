"""Controlled bridge from conversation tools to Skill OS or Supervisor."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from hey_robot.cognition.runtime.conversation_store import ConversationStore
from hey_robot.protocol import (
    ActionProposal,
    Envelope,
    GoalBudgets,
    GoalCommand,
    ShortOperationCommand,
    SkillResult,
    SuccessCriterion,
    ToolOutcome,
    Topics,
)
from hey_robot.protocol.messages import to_payload
from hey_robot.skill_os.base import SkillCatalog


class RobotExecutionAdapter:
    """Maps proposals deterministically; the model never constructs SkillIntent."""

    def __init__(
        self,
        bus: Any,
        topics: Topics,
        catalog: SkillCatalog,
        store: ConversationStore,
        known_entities: tuple[str, ...] = (),
        *,
        timeout_sec: float = 45.0,
    ) -> None:
        self._bus = bus
        self._topics = topics
        self._catalog = catalog
        self._store = store
        self._known_entities = frozenset(known_entities)
        self._timeout_sec = timeout_sec
        self._waiters: dict[str, asyncio.Future[SkillResult]] = {}

    async def execute(
        self, proposal: ActionProposal, envelope: Envelope, session_key: str
    ) -> ToolOutcome:
        spec = self._catalog.get(proposal.skill_name)
        if spec.category in {"navigation", "interaction", "manipulation"}:
            return await self._create_long_goal(proposal, envelope, session_key)
        return await self._execute_short_operation(proposal, envelope)

    async def _create_long_goal(
        self, proposal: ActionProposal, envelope: Envelope, session_key: str
    ) -> ToolOutcome:
        target = proposal.arguments.get("target")
        if not isinstance(target, str) or not target.strip():
            return ToolOutcome(
                "failed", "我需要知道要前往或操作的具体目标。", retryable=True
            )
        if not envelope.robot_id:
            return ToolOutcome("failed", "当前没有可用的机器人。")
        if target.strip() not in self._known_entities:
            return ToolOutcome(
                "failed",
                "我还不知道这个目标的位置，请告诉我更具体的位置或先让我观察。",
                retryable=True,
            )
        goal_id = str(uuid.uuid4())
        command = GoalCommand(
            envelope=envelope,
            command_id=f"conversation_goal_{uuid.uuid4().hex}",
            action="create",
            goal_id=goal_id,
            objective=proposal.objective,
            success_criteria=(
                SuccessCriterion(
                    criterion_id="target_reached",
                    criterion_type="object_relation",
                    subject_id=f"robot:{envelope.robot_id}",
                    predicate="at",
                    object_id=target.strip(),
                    max_age_sec=300.0,
                ),
            ),
            budgets=GoalBudgets(),
        )
        self._store.link_goal(goal_id, session_key, envelope)
        await self._bus.publish(self._topics.goal_command, to_payload(command))
        return ToolOutcome(
            "accepted",
            "任务已开始，完成或需要确认时我会告诉你。",
            {"skill": proposal.skill_name, "target": target.strip()},
            goal_id=goal_id,
        )

    async def _execute_short_operation(
        self, proposal: ActionProposal, envelope: Envelope
    ) -> ToolOutcome:
        skill_id = f"conversation_skill_{uuid.uuid4().hex}"
        future: asyncio.Future[SkillResult] = asyncio.get_running_loop().create_future()
        self._waiters[skill_id] = future
        try:
            await self._bus.publish(
                self._topics.short_operation_command,
                to_payload(
                    ShortOperationCommand(
                        envelope=envelope,
                        operation_id=skill_id,
                        proposal=proposal,
                        timeout_sec=self._timeout_sec,
                    )
                ),
            )
            result = await asyncio.wait_for(future, timeout=self._timeout_sec)
        except TimeoutError:
            return ToolOutcome(
                "waiting",
                "操作仍在执行，等待机器人返回结果。",
                operation_id=skill_id,
                retryable=True,
            )
        finally:
            self._waiters.pop(skill_id, None)
        if result.status == "completed" and result.success is True:
            if proposal.intent_kind == "observation":
                summary = _trusted_observation_summary(result.summary)
                if summary is None:
                    return ToolOutcome(
                        "failed",
                        "我获取到了相机图像，但当前没有可信的场景识别结果，不能确定看到了什么。",
                        operation_id=skill_id,
                        retryable=True,
                    )
                return ToolOutcome(
                    "completed",
                    summary,
                    {"observation_summary": summary},
                    operation_id=skill_id,
                )
            return ToolOutcome(
                "completed",
                result.summary,
                dict(result.metadata),
                operation_id=skill_id,
            )
        return ToolOutcome(
            "failed",
            result.summary or result.error or "操作未完成。",
            dict(result.metadata),
            operation_id=skill_id,
            retryable=result.status == "unknown",
        )

    def accept_result(self, result: SkillResult) -> None:
        waiter = self._waiters.get(result.skill_id)
        if waiter is not None and not waiter.done():
            waiter.set_result(result)


def _trusted_observation_summary(value: str | None) -> str | None:
    """Accept only semantic fields emitted by Runtime, never execution metadata."""
    text = (value or "").strip()
    for part in text.split(";"):
        item = part.strip()
        if item.startswith("scene="):
            scene = item.removeprefix("scene=").strip()
            if scene:
                return scene
    return None
