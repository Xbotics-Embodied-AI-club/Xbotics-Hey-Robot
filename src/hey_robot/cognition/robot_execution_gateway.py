"""将对话工具提案受控地桥接到 Skill OS 或 Supervisor。"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from hey_robot.cognition.runtime.conversation_store import ConversationStore
from hey_robot.protocol import (
    ActionProposal,
    Envelope,
    ShortOperationCommand,
    SkillResult,
    ToolOutcome,
    Topics,
)
from hey_robot.protocol.messages import to_payload
from hey_robot.skill_os.base import SkillCatalog


class RobotExecutionGateway:
    """确定性映射对话提案；模型永远不直接构造 SkillIntent。"""

    def __init__(
        self,
        bus: Any,
        topics: Topics,
        catalog: SkillCatalog,
        store: ConversationStore,
        *,
        timeout_sec: float = 45.0,
    ) -> None:
        self._bus = bus
        self._topics = topics
        self._catalog = catalog
        self._store = store
        self._timeout_sec = timeout_sec
        self._waiters: dict[str, asyncio.Future[SkillResult]] = {}

    async def execute(
        self, proposal: ActionProposal, envelope: Envelope, session_key: str
    ) -> ToolOutcome:
        del session_key
        return await self._execute_short_operation(proposal, envelope)

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
                "failed",
                "这次操作没有完成：机器人没有在限定时间内返回最终结果。",
                operation_id=skill_id,
                retryable=False,
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
                    {
                        "observation_summary": summary,
                        "frame_id": result.frame_id,
                        "evidence_ids": [fact.evidence_id for fact in result.evidence],
                    },
                    operation_id=skill_id,
                )
            return ToolOutcome(
                "completed",
                result.summary,
                {
                    **dict(result.metadata),
                    "frame_id": result.frame_id,
                    "evidence_ids": [fact.evidence_id for fact in result.evidence],
                },
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
