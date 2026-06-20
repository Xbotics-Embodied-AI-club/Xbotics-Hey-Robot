from __future__ import annotations

from typing import Any

from hey_robot.agents.progress import RobotAgentProgress
from hey_robot.protocol import SkillResult


class SkillRecoveryNotifier:
    """Applies recovery decisions and publishes recovery progress events."""

    def __init__(self, service: Any) -> None:
        self.service = service

    async def publish_if_needed(self, result: SkillResult, recovery: Any) -> None:
        if not result.envelope.episode_id or not recovery.needed:
            return
        svc = self.service
        svc.task_runtime.apply_recovery(
            episode_id=result.envelope.episode_id,
            decision=recovery,
        )
        await svc._publish_agent_progress(
            RobotAgentProgress(
                phase="recovery_planned",
                summary=recovery.summary,
                episode_id=result.envelope.episode_id,
                agent_id=result.envelope.agent_id,
                robot_id=result.envelope.robot_id,
                skill_id=result.skill_id,
                trace_id=result.envelope.trace_id,
                metadata=recovery.to_dict(),
            )
        )
