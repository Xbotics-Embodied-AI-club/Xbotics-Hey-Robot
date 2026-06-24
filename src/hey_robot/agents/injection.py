from __future__ import annotations

from dataclasses import dataclass

from hey_robot.agents.interaction import UserInteractionIntent
from hey_robot.agents.task_run import TaskRun
from hey_robot.agents.types import RobotSnapshot
from hey_robot.protocol import UserTurn


@dataclass(frozen=True)
class InjectedTurnPlan:
    text: str
    metadata: dict[str, object]


class RobotTurnInjector:
    """Merge mid-turn user input into the active robot task context."""

    def inject(
        self,
        *,
        turn: UserTurn,
        intent: UserInteractionIntent | None,
        task: TaskRun | None,
        snapshot: RobotSnapshot,
    ) -> InjectedTurnPlan:
        if intent is None or intent.kind not in {
            "correction",
            "interrupt",
            "follow_up",
            "retry",
        }:
            return InjectedTurnPlan(text=turn.text, metadata={})
        root_task = task.root_task if task is not None else turn.text
        active = task.attempts[-1] if task is not None and task.attempts else None
        previous_skill = active.text if active is not None else None
        frame_id = snapshot.status.frame_id if snapshot.status is not None else None
        recovery_resume = turn.metadata.get("_recovery_resume")
        lines = [f"Continue the active robot task: {root_task}"]
        if previous_skill and previous_skill != root_task:
            lines.append(f"Previous skill or action: {previous_skill}")
        if isinstance(recovery_resume, dict):
            strategy = str(recovery_resume.get("strategy") or "reobserve")
            summary = str(
                recovery_resume.get("summary")
                or "Resolve the last recovery step first."
            )
            recommended_tools = [
                str(item).strip()
                for item in recovery_resume.get("recommended_tools", [])
                if str(item).strip()
            ]
            continuation_guidance = str(
                recovery_resume.get("continuation_guidance")
                or "Resolve the recovery condition before continuing the original task."
            )
            lines.append(
                "Recovery was active in the previous turn and has just been resumed."
            )
            lines.append(f"Recovery strategy: {strategy}")
            lines.append(f"Recovery summary: {summary}")
            if recommended_tools:
                lines.append(
                    f"Recommended recovery tools first: {', '.join(recommended_tools)}"
                )
            lines.append(f"Recovery continuation guidance: {continuation_guidance}")
            lines.append(
                "First resolve the recovery step, then continue the original task."
            )
        lines.append(f"User {intent.kind}: {turn.text}")
        if frame_id is not None:
            lines.append(f"Current robot frame: {frame_id}")
        lines.append(
            "Produce the next safe robot skill objective that follows the user's latest instruction."
        )
        return InjectedTurnPlan(
            text="\n".join(lines),
            metadata={
                "injected": True,
                "interaction_intent": intent.to_dict(),
                "root_task": root_task,
                "previous_skill": previous_skill,
                "frame_id": frame_id,
            },
        )
