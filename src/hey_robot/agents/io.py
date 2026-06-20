from __future__ import annotations

from typing import Any, Protocol

from hey_robot.agents.perception_query import SceneEvidence
from hey_robot.protocol import AgentReply, SkillIntent


class AgentIO(Protocol):
    async def submit_skill(self, skill: SkillIntent) -> None: ...

    async def publish_reply(self, reply: AgentReply) -> None: ...

    async def publish_notification(
        self,
        text: str,
        *,
        channel: str | None = None,
        chat_id: str | None = None,
        sender_id: str | None = None,
        message_id: str | None = None,
        reply_to_current: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    async def publish_task_result(self, *, success: bool, summary: str) -> None: ...

    async def query_scene_evidence(
        self,
        *,
        robot_id: str | None,
        question: str,
        baseline_frame_id: int | None = None,
        freshness: str = "fresh",
        timeout_sec: float = 2.0,
    ) -> SceneEvidence: ...
