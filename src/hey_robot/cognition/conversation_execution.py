"""将对话工具提案受控地桥接到 Skill OS 或 Supervisor。"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import replace
from typing import Any

from hey_robot.cognition.conversation_entities import (
    EntityResolutionError,
    EntityResolver,
)
from hey_robot.cognition.conversation_goal import (
    GoalContractBuilder,
    GoalControlProposal,
    GoalProposal,
)
from hey_robot.cognition.runtime.conversation_store import ConversationStore
from hey_robot.protocol import (
    ActionProposal,
    Envelope,
    GoalCommand,
    ShortOperationCommand,
    SkillResult,
    ToolOutcome,
    Topics,
)
from hey_robot.protocol.messages import to_payload
from hey_robot.skill_os.base import SkillCatalog


class RobotExecutionAdapter:
    """确定性映射对话提案；模型永远不直接构造 SkillIntent。"""

    def __init__(
        self,
        bus: Any,
        topics: Topics,
        catalog: SkillCatalog,
        store: ConversationStore,
        known_entities: tuple[str, ...] = (),
        goal_contract_builder: GoalContractBuilder | None = None,
        entity_resolver: EntityResolver | None = None,
        *,
        timeout_sec: float = 45.0,
    ) -> None:
        self._bus = bus
        self._topics = topics
        self._catalog = catalog
        self._store = store
        self._known_entities = frozenset(known_entities)
        self._goal_contract_builder = goal_contract_builder or GoalContractBuilder(
            known_entities
        )
        self._entity_resolver = entity_resolver or EntityResolver(known_entities)
        self._timeout_sec = timeout_sec
        self._waiters: dict[str, asyncio.Future[SkillResult]] = {}

    async def execute(
        self,
        proposal: ActionProposal | GoalProposal | GoalControlProposal,
        envelope: Envelope,
        session_key: str,
    ) -> ToolOutcome:
        if isinstance(proposal, GoalProposal):
            return await self._create_long_goal(proposal, envelope, session_key)
        if isinstance(proposal, GoalControlProposal):
            return await self._control_goal(proposal, envelope, session_key)
        return await self._execute_short_operation(proposal, envelope)

    async def _create_long_goal(
        self, proposal: GoalProposal, envelope: Envelope, session_key: str
    ) -> ToolOutcome:
        active = self._store.active_goal(session_key)
        if active is not None:
            return ToolOutcome(
                "failed",
                "An active task already exists; cancel or finish it before creating another.",
                {"goal_id": active["goal_id"], "status": active["status"]},
            )
        if not envelope.robot_id:
            return ToolOutcome("failed", "当前没有可用的机器人。")
        try:
            resolved_target = self._entity_resolver.resolve(
                proposal.target, robot_id=envelope.robot_id
            )
            proposal = replace(proposal, target=resolved_target.target_id)
            success_criteria, budgets = self._goal_contract_builder.build(
                proposal,
                robot_id=envelope.robot_id,
                target_entity=resolved_target.entity,
            )
        except EntityResolutionError as exc:
            return ToolOutcome("failed", str(exc), retryable=True)
        except ValueError as exc:
            return ToolOutcome(
                "failed",
                f"无法创建可验证任务：{exc}。请说明已知地点，或先让我观察。",
                retryable=True,
            )
        goal_id = str(uuid.uuid4())
        command = GoalCommand(
            envelope=envelope,
            command_id=f"conversation_goal_{uuid.uuid4().hex}",
            action="create",
            goal_id=goal_id,
            objective=proposal.objective,
            success_criteria=success_criteria,
            budgets=budgets,
        )
        self._store.link_goal(
            goal_id, session_key, envelope, objective=proposal.objective
        )
        await self._bus.publish(self._topics.goal_command, to_payload(command))
        return ToolOutcome(
            "accepted",
            "任务已开始，完成或需要确认时我会告诉你。",
            {"goal_kind": proposal.goal_kind, "target": proposal.target},
            goal_id=goal_id,
        )

    async def _control_goal(
        self,
        proposal: GoalControlProposal,
        envelope: Envelope,
        session_key: str,
    ) -> ToolOutcome:
        active = self._store.active_goal(session_key)
        if active is None and proposal.action != "emergency_stop":
            return ToolOutcome("failed", "当前没有可控制的进行中任务。")
        goal_id = None if active is None else active["goal_id"]
        command = GoalCommand(
            envelope=envelope,
            command_id=f"conversation_goal_control_{uuid.uuid4().hex}",
            action=proposal.action,
            goal_id=goal_id,
            condition_id=proposal.condition_id,
        )
        await self._bus.publish(self._topics.goal_command, to_payload(command))
        if proposal.action == "cancel":
            return ToolOutcome("accepted", "已请求取消当前任务。", goal_id=goal_id)
        if proposal.action == "emergency_stop":
            return ToolOutcome("accepted", "已请求紧急停止。", goal_id=goal_id)
        return ToolOutcome("accepted", "已确认任务可以继续。", goal_id=goal_id)

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
    """仅接收 Runtime 产生的语义字段，绝不接收执行元数据。"""
    text = (value or "").strip()
    for part in text.split(";"):
        item = part.strip()
        if item.startswith("scene="):
            scene = item.removeprefix("scene=").strip()
            if scene:
                return scene
    return None
