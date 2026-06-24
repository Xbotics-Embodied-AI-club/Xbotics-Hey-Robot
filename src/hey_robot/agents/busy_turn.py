from __future__ import annotations

from collections.abc import Awaitable, Callable

from hey_robot.agents.interaction import (
    UserInteractionIntent,
    classify_user_interaction,
)
from hey_robot.agents.progress import RobotAgentProgress
from hey_robot.agents.session import AgentTurnSessions
from hey_robot.agents.skill_gateway import SkillGateway
from hey_robot.agents.task_run import TaskRun
from hey_robot.agents.task_runtime import RobotStateCache, TaskRunManager
from hey_robot.protocol import AgentReply, SkillEvent, SkillIntent, UserTurn

ReplyPublisher = Callable[[AgentReply], Awaitable[None]]
ProgressPublisher = Callable[[RobotAgentProgress], Awaitable[None]]
SkillEventPublisher = Callable[[SkillEvent], Awaitable[None]]
SkillIntentPublisher = Callable[[SkillIntent], Awaitable[None]]


class BusyTurnHandler:
    """Handles user turns that arrive while the robot is busy.

    This keeps the service boundary focused on turn handling. Busy-turn policy lives
    here: read-only status answers, queued corrections, follow-ups, and
    interrupt skill intents.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        default_robot: str | None,
        robot_cache: RobotStateCache,
        task_runtime: TaskRunManager,
        turn_sessions: AgentTurnSessions,
        publish_reply: ReplyPublisher,
        publish_progress: ProgressPublisher,
        publish_skill_event: SkillEventPublisher,
        publish_skill_intent: SkillIntentPublisher,
    ) -> None:
        self.agent_id = agent_id
        self.default_robot = default_robot
        self.robot_cache = robot_cache
        self.task_runtime = task_runtime
        self.turn_sessions = turn_sessions
        self.publish_reply = publish_reply
        self.publish_progress = publish_progress
        self.publish_skill_event = publish_skill_event
        self.publish_skill_intent = publish_skill_intent

    async def handle(self, turn: UserTurn, *, active_skill_id: str) -> bool:
        task = (
            self.task_runtime.task_runs.load_active(turn.envelope.episode_id)
            if turn.envelope.episode_id
            else None
        )
        intent = self._refine_busy_intent(
            classify_user_interaction(turn.text, robot_busy=True),
            task=task,
            active_skill_id=active_skill_id,
        )
        if intent.kind == "read_only":
            await self.publish_progress(
                RobotAgentProgress(
                    phase="readonly_during_skill",
                    summary=f"user read-only interaction answered during active skill {active_skill_id}",
                    episode_id=turn.envelope.episode_id,
                    agent_id=self.agent_id,
                    robot_id=turn.envelope.robot_id,
                    skill_id=active_skill_id,
                    trace_id=turn.envelope.trace_id,
                    metadata={"intent": intent.to_dict(), "text": turn.text},
                )
            )
            await self._reply_readonly(
                turn, active_skill_id=active_skill_id, intent=intent
            )
            return True

        if turn.envelope.episode_id:
            self.task_runtime.enqueue_pending_turn(
                turn,
                reason=intent.kind,
                intent=intent.to_dict(),
                active_skill_id=active_skill_id,
            )
        await self.publish_progress(
            RobotAgentProgress(
                phase=f"queued_{intent.kind}",
                summary=f"user {intent.kind} queued for active skill {active_skill_id}",
                episode_id=turn.envelope.episode_id,
                agent_id=self.agent_id,
                robot_id=turn.envelope.robot_id,
                skill_id=active_skill_id,
                trace_id=turn.envelope.trace_id,
                metadata={"intent": intent.to_dict(), "text": turn.text},
            )
        )

        if intent.kind in {"interrupt", "emergency_stop"}:
            await self.publish_skill_event(
                SkillEvent(
                    envelope=turn.envelope,
                    skill_id=active_skill_id,
                    phase="interrupted",
                    text=turn.text,
                    error="interrupted by user",
                    summary="user requested interruption",
                    metadata={"source": "agent.busy_turn", "intent": intent.to_dict()},
                )
            )
            await self.publish_skill_intent(
                SkillGateway.build_interrupt_intent(
                    envelope=turn.envelope,
                    active_skill_id=active_skill_id,
                    objective=turn.text,
                    metadata={
                        **dict(turn.metadata),
                        "mode": intent.kind,
                        "source": "agent.busy_turn",
                        "intent": intent.to_dict(),
                    },
                )
            )
            self.turn_sessions.release_robot(turn.envelope.robot_id)
            await self.publish_reply(
                AgentReply(
                    envelope=turn.envelope,
                    text="Interrupt received. I will stop the active skill and reconsider the task.",
                    metadata={
                        "busy": True,
                        "queued": True,
                        "intent": intent.to_dict(),
                        "skill_id": active_skill_id,
                    },
                )
            )
            return True

        if intent.kind in {"correction", "follow_up", "retry", "reset"}:
            await self.publish_reply(
                AgentReply(
                    envelope=turn.envelope,
                    text="Update received. I will merge it into the active task at the next safe boundary.",
                    metadata={
                        "busy": True,
                        "queued": True,
                        "intent": intent.to_dict(),
                        "skill_id": active_skill_id,
                    },
                )
            )
            return True

        await self.publish_reply(
            AgentReply(
                envelope=turn.envelope,
                text=f"Robot {turn.envelope.robot_id} is busy with skill {active_skill_id}",
                metadata={"busy": True, "skill_id": active_skill_id},
            )
        )
        return True

    @staticmethod
    def _refine_busy_intent(
        intent: UserInteractionIntent,
        *,
        task: TaskRun | None,
        active_skill_id: str,
    ) -> UserInteractionIntent:
        del active_skill_id
        if intent.kind in {"interrupt", "emergency_stop", "read_only"}:
            return intent
        if task is None:
            return (
                intent
                if intent.kind == "follow_up"
                else type(intent)(
                    kind="follow_up",
                    urgency="safe_boundary",
                    target="task",
                )
            )
        if task.status in {"paused", "recovering", "awaiting_confirmation"}:
            if intent.kind == "correction":
                return type(intent)(
                    kind="follow_up",
                    urgency="safe_boundary",
                    target="task",
                )
            return intent
        if task.status == "executing" and task.active_attempt_id:
            return intent
        if intent.kind == "correction":
            return type(intent)(
                kind="follow_up",
                urgency="safe_boundary",
                target="task",
            )
        return intent

    async def _reply_readonly(
        self, turn: UserTurn, *, active_skill_id: str, intent
    ) -> None:
        snapshot = self.robot_cache.snapshot(
            turn.envelope.robot_id, default_robot=self.default_robot
        )
        lines = [f"Robot is executing skill {active_skill_id}."]
        if snapshot.status is not None:
            lines.append(f"Status: {snapshot.status.state}.")
            battery = snapshot.status.metrics.get("battery")
            if isinstance(battery, dict):
                battery_parts = []
                if battery.get("percentage") is not None:
                    battery_parts.append(f"{battery['percentage']}%")
                if battery.get("voltage") is not None:
                    battery_parts.append(f"{battery['voltage']}V")
                if battery.get("status") is not None:
                    battery_parts.append(str(battery["status"]))
                if battery_parts:
                    lines.append(f"Battery: {', '.join(battery_parts)}.")
            arm = snapshot.status.metrics.get("arm")
            if isinstance(arm, dict):
                arm_summary = ", ".join(
                    f"{key}={value}" for key, value in list(arm.items())[:4]
                )
                if arm_summary:
                    lines.append(f"Arm: {arm_summary}.")
        if snapshot.observation is not None:
            lines.append(
                f"Observation frame {snapshot.observation.frame_id}, images={len(snapshot.observation.images)}."
            )
        if snapshot.skill_result is not None:
            lines.append(
                f"Last skill: {snapshot.skill_result.skill_id} {snapshot.skill_result.status}."
            )
        await self.publish_reply(
            AgentReply(
                envelope=turn.envelope,
                text=" ".join(lines),
                metadata={
                    "busy": True,
                    "readonly": True,
                    "intent": intent.to_dict(),
                    "skill_id": active_skill_id,
                },
            )
        )
