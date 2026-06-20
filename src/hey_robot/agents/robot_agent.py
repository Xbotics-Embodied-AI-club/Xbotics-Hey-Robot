from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hey_robot.agents.container import RobotAgentRuntimeContainer
from hey_robot.agents.execution_feedback import ExecutionFeedback
from hey_robot.agents.interaction import PendingConfirmationDecision
from hey_robot.agents.loop import RobotTurnTraceEntry
from hey_robot.agents.perception_query import SceneEvidence
from hey_robot.agents.progress import RobotAgentProgress
from hey_robot.agents.runtime.grounding import is_perception_skill_name
from hey_robot.agents.service.skill_result_handler import SkillResultHandler
from hey_robot.agents.types import AgentCoreResult, AgentTurnInput, RobotSnapshot
from hey_robot.bus.factory import create_bus_client
from hey_robot.capability.catalog.models import CapabilityManifest
from hey_robot.config import DeploymentConfig
from hey_robot.events import EventKind, RuntimeEvent
from hey_robot.events.bus import BusEventPublisher
from hey_robot.logging import HeyRobotLogger
from hey_robot.protocol import (
    AgentReply,
    RobotObservation,
    RobotStatus,
    SkillEvent,
    SkillIntent,
    SkillResult,
    Topics,
    UserTurn,
)
from hey_robot.protocol.messages import from_payload, to_payload

logger = HeyRobotLogger(name="agent")


@dataclass(frozen=True)
class PreparedTurn:
    turn: UserTurn
    snapshot: RobotSnapshot
    baseline_frame_id: int | None


class RobotAgentService:
    """Agent-side service boundary for robot task turns.

    The service owns the deployable contract: consume routed user turns, inspect
    robot state, submit skill intents, and send replies through the gateway.
    Rich LLM planning can be layered behind this boundary without changing
    channels, episodes, policies, or robot drivers.
    """

    def __init__(
        self,
        config: DeploymentConfig,
        *,
        agent_id: str,
        episode_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.agent_id = agent_id
        self.topics = Topics()
        self.bus = create_bus_client(config.deployment.bus)
        self.events = BusEventPublisher(self.bus, self.topics)
        self._bind_runtime(
            RobotAgentRuntimeContainer.build(
                config=config,
                agent_id=agent_id,
                episode_dir=episode_dir,
                io=self,
                publish_reply=self.publish_reply,
                publish_progress=self._publish_agent_progress,
                publish_skill_event=self._publish_skill_event,
                publish_skill_intent=self._publish_skill_intent,
            )
        )

    def _bind_runtime(self, runtime: RobotAgentRuntimeContainer) -> None:
        self.runtime_container = runtime
        self.episodes = runtime.episodes
        self.robot_states = runtime.robot_states
        self.notification_runtime = runtime.notification_runtime
        self.robot_cache = runtime.robot_cache
        self.latest_status = self.robot_cache.status
        self.latest_observation = self.robot_cache.observation
        self.latest_skill_result = self.robot_cache.skill_result
        self.latest_skill_event_frame = self.robot_cache.skill_event_frame
        self.turn_sessions = runtime.turn_sessions
        self._episode_locks: dict[str, asyncio.Lock] = {}
        self._robot_locks: dict[str, asyncio.Lock] = {}
        self.task_runtime = runtime.task_runtime
        self.media_resolver = runtime.media_resolver
        self.scene_captioner = runtime.scene_captioner
        self.scene_runtime = runtime.scene_runtime
        self.busy_turns = runtime.busy_turns
        self.turn_timeout_sec = runtime.turn_timeout_sec
        self.skill_lease_timeout_sec = runtime.skill_lease_timeout_sec
        self.core = runtime.core
        self.loop = runtime.loop
        self.capabilities = runtime.capabilities
        self.skill_result_handler = SkillResultHandler(self)

    def capability_manifest(self) -> CapabilityManifest:
        robot_type = self.core._configured_robot_type()
        return self.capabilities.build(robot_type=robot_type)

    async def start(self) -> None:
        logger.info(
            f"启动 agent service：agent={self.agent_id} deployment="
            f"[{self.config.deployment.id}] bus={self.config.deployment.bus.url}"
        )
        await self.bus.connect()
        logger.info(f"{self.agent_id} 已连接 bus")
        await self.core.start()
        self.task_runtime.mark_recovery_required_for_nonterminal()
        await self.bus.subscribe([self.topics.robot_status], self._on_status)
        await self.bus.subscribe([self.topics.robot_observation], self._on_observation)
        await self.bus.subscribe([self.topics.skill_result], self._on_skill_result)
        await self.bus.subscribe([self.topics.skill_event], self._on_skill_event)
        await self.bus.subscribe([self.topics.user_turn], self._on_user_turn)
        logger.info(
            f"{self.agent_id} 就绪，已订阅 "
            f"{self.topics.user_turn}, {self.topics.robot_observation}, "
            f"{self.topics.robot_status}, {self.topics.skill_result}, {self.topics.skill_event}"
        )

        await asyncio.Event().wait()

    async def stop(self) -> None:
        await self.scene_runtime.stop()
        await self.core.close()
        await self.bus.close()

    async def _on_status(self, _topic: str, payload: dict) -> None:
        status = from_payload(RobotStatus, payload)
        if status.envelope.robot_id:
            self.latest_status[status.envelope.robot_id] = status
        self.task_runtime.observe_status(status)

    async def _on_observation(self, _topic: str, payload: dict) -> None:
        observation = from_payload(RobotObservation, payload)
        logger.debug(
            f"{self.agent_id} 收到 observation robot={observation.envelope.robot_id} "
            f"frame={observation.frame_id} images={len(observation.images)}"
        )
        if observation.envelope.robot_id:
            self.latest_observation[observation.envelope.robot_id] = observation
            self.task_runtime.observe_observation(observation)
            status = self.latest_status.get(observation.envelope.robot_id)
            if status is None or status.frame_id != observation.frame_id:
                self.latest_status[observation.envelope.robot_id] = RobotStatus(
                    envelope=observation.envelope,
                    frame_id=observation.frame_id,
                    state="observed",
                    task=observation.task,
                    metrics={"image_count": len(observation.images)},
                )
                status = self.latest_status[observation.envelope.robot_id]
            self.scene_runtime.schedule_memory(
                observation,
                status,
                progress_callback=self._publish_agent_progress,
            )

    async def _on_skill_event(self, _topic: str, payload: dict) -> None:
        event = from_payload(SkillEvent, payload)
        logger.debug(
            f"{self.agent_id} 收到 skill_event skill={event.skill_id} "
            f"phase={event.phase} robot={event.envelope.robot_id}"
        )
        if event.frame_id is not None and event.phase in {
            "completed",
            "failed",
            "interrupted",
        }:
            self.latest_skill_event_frame[event.skill_id] = event.frame_id
        self.task_runtime.observe_skill_event(event)

    async def _on_skill_result(self, _topic: str, payload: dict) -> None:
        result = from_payload(SkillResult, payload)
        await self.skill_result_handler.handle(result)

    async def _on_user_turn(self, _topic: str, payload: dict) -> None:
        turn = from_payload(UserTurn, payload)
        if turn.envelope.agent_id != self.agent_id:
            return
        logger.info(
            f"{self.agent_id} 收到 turn trace={turn.envelope.trace_id} "
            f"robot={turn.envelope.robot_id} text_len={len(turn.text)}"
        )
        if self.turn_sessions.is_duplicate_or_remember(turn):
            return
        if await self._precheck_busy_turn(turn):
            return
        await self._run_turn_pipeline(turn)

    async def _precheck_busy_turn(self, turn: UserTurn) -> bool:
        lease = self.turn_sessions.active_robot_lease(
            turn.envelope.robot_id,
            timeout_sec=self.skill_lease_timeout_sec,
        )
        if lease is None:
            return False
        await self._handle_busy_turn(turn, active_skill_id=lease[0])
        return True

    async def _run_turn_pipeline(self, turn: UserTurn) -> None:
        episode_key = turn.envelope.episode_id or turn.envelope.trace_id
        robot_key = (
            turn.envelope.robot_id
            or self.config.default_robot_id(self.agent_id)
            or "none"
        )
        try:
            async with (
                self._lock(self._episode_locks, episode_key),
                self._lock(self._robot_locks, robot_key),
            ):
                await asyncio.wait_for(
                    self._handle_user_turn_locked(turn), timeout=self.turn_timeout_sec
                )
        except TimeoutError:
            logger.error(
                f"{self.agent_id} turn 超时 trace={turn.envelope.trace_id} "
                f"timeout_sec={self.turn_timeout_sec}"
            )
            await self._publish_turn_failure_reply(
                turn,
                text="处理这个问题超时了，请稍后重试。",
                reason="turn_timeout",
            )
        except Exception as exc:
            logger.error(
                f"{self.agent_id} turn 异常 trace={turn.envelope.trace_id} "
                f"error={type(exc).__name__}: {exc}"
            )
            await self._publish_turn_failure_reply(
                turn,
                text="处理这个问题时发生了错误，请稍后重试。",
                reason="turn_error",
                error=type(exc).__name__,
            )
        await self._drain_pending_turns(turn.envelope.episode_id)

    async def _publish_turn_failure_reply(
        self,
        turn: UserTurn,
        *,
        text: str,
        reason: str,
        error: str | None = None,
    ) -> None:
        await self.publish_reply(
            AgentReply(
                envelope=turn.envelope,
                text=text,
                metadata={"source": reason, "error": error},
            )
        )
        await self.events.publish(
            RuntimeEvent.make(
                EventKind.AGENT_TURN_END,
                source="agent",
                severity="error",
                trace_id=turn.envelope.trace_id,
                episode_id=turn.envelope.episode_id,
                agent_id=self.agent_id,
                robot_id=turn.envelope.robot_id,
                channel=turn.envelope.channel,
                payload={"status": reason, "error": error},
            )
        )

    async def _drain_pending_turns(self, episode_id: str | None) -> None:
        if episode_id:
            await self._continue_pending_turn_for_episode(episode_id)

    async def _handle_user_turn_locked(
        self, turn: UserTurn, *, replayed: bool = False
    ) -> None:
        prepared = await self._build_turn(turn, replayed=replayed)
        if prepared is None:
            return
        try:
            result, trace = await self._execute_turn(prepared)
        finally:
            self.turn_sessions.release_robot(prepared.turn.envelope.robot_id or "none")
        await self._commit_turn_result(
            prepared=prepared,
            result=result,
            trace=trace,
        )

    async def _build_turn(
        self,
        turn: UserTurn,
        *,
        replayed: bool = False,
    ) -> PreparedTurn | None:
        snapshot = self._snapshot(turn.envelope.robot_id)
        resolved_turn = await self._resolve_pending_confirmation_turn(
            turn, snapshot=snapshot
        )
        if resolved_turn is None:
            return None
        turn = resolved_turn
        logger.debug(
            f"{self.agent_id} 处理 user turn trace={turn.envelope.trace_id} "
            f"robot={turn.envelope.robot_id} episode={turn.envelope.episode_id}"
        )
        if not replayed:
            self.task_runtime.clear_for_new_turn(turn.envelope.episode_id)
        self.turn_sessions.lease_robot(
            turn.envelope.robot_id or "none", "__agent_turn__"
        )
        await self.events.publish(
            RuntimeEvent.make(
                EventKind.AGENT_TURN_START,
                source="agent",
                trace_id=turn.envelope.trace_id,
                episode_id=turn.envelope.episode_id,
                agent_id=self.agent_id,
                robot_id=turn.envelope.robot_id,
                channel=turn.envelope.channel,
                payload={"text_len": len(turn.text)},
            )
        )
        self.task_runtime.mark_task_started(turn, agent_id=self.agent_id)
        agent_spec = self.config.agents.get(self.agent_id)
        max_age_sec = float(
            agent_spec.settings.get("active_perception_max_age_sec", 15.0)
            if agent_spec is not None
            else 15.0
        )
        scene_freshness = self.scene_runtime.assess_turn_freshness(
            robot_id=turn.envelope.robot_id,
            default_robot=self.config.default_robot_id(self.agent_id),
            text=turn.text,
            max_age_sec=max_age_sec,
        )
        turn.metadata["_scene_freshness"] = scene_freshness.to_dict()
        baseline_frame_id = (
            snapshot.observation.frame_id if snapshot.observation is not None else None
        )
        return PreparedTurn(
            turn=turn, snapshot=snapshot, baseline_frame_id=baseline_frame_id
        )

    async def _execute_turn(self, prepared: PreparedTurn):
        return await self.loop.run_turn(
            turn=prepared.turn,
            snapshot=prepared.snapshot,
            history=self._episode_history(prepared.turn.envelope.episode_id),
            recovery_context=self.task_runtime.recovery_context(
                prepared.turn.envelope.episode_id
            ),
            progress_callback=self._publish_agent_progress,
        )

    async def _commit_turn_result(
        self,
        *,
        prepared: PreparedTurn,
        result,
        trace,
    ) -> None:
        self.turn_sessions.record_turn_result(
            turn=prepared.turn,
            result=result,
            baseline_frame_id=prepared.baseline_frame_id,
        )
        if result.reply_text:
            await self.publish_reply(
                AgentReply(
                    envelope=prepared.turn.envelope,
                    text=result.reply_text,
                    metadata={
                        "source": "final_response",
                        "tool": result.tool,
                        "stop_reason": result.metadata.get("stop_reason"),
                        **dict(result.metadata.get("reply_metadata") or {}),
                    },
                )
            )
        await self._publish_turn_end(turn=prepared.turn, result=result, trace=trace)

    async def _publish_turn_start(self, turn: UserTurn) -> None:
        await self.events.publish(
            RuntimeEvent.make(
                EventKind.AGENT_TURN_START,
                source="agent",
                trace_id=turn.envelope.trace_id,
                episode_id=turn.envelope.episode_id,
                agent_id=self.agent_id,
                robot_id=turn.envelope.robot_id,
                channel=turn.envelope.channel,
                payload={"text_len": len(turn.text)},
            )
        )

    async def _publish_turn_end(self, *, turn: UserTurn, result, trace) -> None:
        await self.events.publish(
            RuntimeEvent.make(
                EventKind.AGENT_TURN_END,
                source="agent",
                trace_id=turn.envelope.trace_id,
                episode_id=turn.envelope.episode_id,
                agent_id=self.agent_id,
                robot_id=turn.envelope.robot_id,
                channel=turn.envelope.channel,
                payload={
                    "status": result.tool,
                    "stop_reason": result.metadata.get("stop_reason"),
                    "turn_trace": [entry.__dict__ for entry in trace],
                    **result.metadata,
                },
            )
        )

    async def _resolve_pending_confirmation_turn(
        self, turn: UserTurn, *, snapshot: RobotSnapshot
    ) -> UserTurn | None:
        proposal = self.task_runtime.pending_confirmation(turn.envelope.episode_id)
        if not proposal:
            return turn
        decision = self._classify_pending_confirmation_turn(turn=turn)
        if decision.action == "ignore":
            return turn
        if decision.action == "new_task":
            self.task_runtime.clear_pending_confirmation(turn.envelope.episode_id)
            return turn
        self.task_runtime.clear_pending_confirmation(turn.envelope.episode_id)
        if decision.action == "decline":
            await self.publish_reply(
                AgentReply(
                    envelope=turn.envelope,
                    text="好的，这个动作我先不执行。",
                    metadata={
                        "source": "final_response",
                        "proposal_id": proposal.get("proposal_id"),
                        "pending_confirmation_declined": True,
                    },
                )
            )
            return None
        await self._execute_confirmed_proposal(
            turn, proposal=proposal, snapshot=snapshot
        )
        return None

    async def _execute_confirmed_proposal(
        self,
        turn: UserTurn,
        *,
        proposal: dict[str, Any],
        snapshot: RobotSnapshot,
    ) -> None:
        capability = str(proposal.get("capability") or "").strip()
        objective = str(proposal.get("objective") or "").strip()
        if not capability or not objective:
            return
        raw_slots = proposal.get("slots")
        slots: dict[str, Any] = dict(raw_slots) if isinstance(raw_slots, dict) else {}
        interrupt = bool(proposal.get("interrupt"))
        confirmed_turn = UserTurn(
            envelope=turn.envelope,
            text=objective,
            media=turn.media,
            intent=turn.intent,
            metadata={
                **dict(turn.metadata),
                "_pending_confirmation": True,
                "_confirmed_proposal": dict(proposal),
            },
        )
        await self._publish_turn_start(confirmed_turn)
        self.task_runtime.mark_task_started(confirmed_turn, agent_id=self.agent_id)
        self.core.bind_turn_context(
            AgentTurnInput(turn=confirmed_turn, snapshot=snapshot)
        )
        self.core._refresh_tool_context()
        args: dict[str, Any] = {
            "capability": capability,
            "objective": objective,
            "slots": slots,
            "interrupt": interrupt,
            "wait_policy": "wait_acceptance",
        }
        robot_key = confirmed_turn.envelope.robot_id or "none"
        self.turn_sessions.lease_robot(robot_key, "__agent_turn__")
        self.core._turn_submitted_skill_id = None
        try:
            result_text = await self.core.request_capability(
                capability,
                objective,
                slots=slots,
                interrupt=interrupt,
                wait_policy="wait_acceptance",
                confirmed=True,
            )
            skill_id = getattr(self.core, "_turn_submitted_skill_id", None)
            reply_text = _confirmed_capability_reply(capability)
            result = AgentCoreResult(
                reply_text=reply_text,
                skill_submitted=True,
                task_finished=False,
                tool="request_capability",
                metadata={
                    "tool": "request_capability",
                    "args": args,
                    "result": result_text,
                    "skill_id": skill_id,
                    "stop_reason": "confirmed_capability",
                    "confirmed_proposal": True,
                    "proposal_id": proposal.get("proposal_id"),
                },
            )
        except Exception as exc:
            result = AgentCoreResult(
                reply_text=_confirmed_capability_failure_reply(capability),
                skill_submitted=False,
                task_finished=True,
                tool="request_capability",
                metadata={
                    "tool": "request_capability",
                    "args": args,
                    "result": str(exc),
                    "skill_id": None,
                    "stop_reason": "confirmed_capability_failed",
                    "confirmed_proposal": True,
                    "proposal_id": proposal.get("proposal_id"),
                },
            )
        finally:
            self.turn_sessions.release_robot(robot_key)
        baseline_frame_id = (
            snapshot.observation.frame_id if snapshot.observation is not None else None
        )
        self.task_runtime.observe_turn_result(
            turn=confirmed_turn,
            result_tool=result.tool,
            reply_text=result.reply_text,
            task_finished=result.task_finished,
            skill_id=result.metadata.get("skill_id"),
            last_observation_frame=baseline_frame_id,
        )
        await self._commit_turn_result(
            prepared=PreparedTurn(
                turn=confirmed_turn,
                snapshot=snapshot,
                baseline_frame_id=baseline_frame_id,
            ),
            result=result,
            trace=[
                RobotTurnTraceEntry(state="confirm", event="accepted"),
                RobotTurnTraceEntry(state="run", event="ok"),
                RobotTurnTraceEntry(state="save", event="ok"),
            ],
        )

    def _classify_pending_confirmation_turn(
        self,
        *,
        turn: UserTurn,
    ) -> PendingConfirmationDecision:
        text = " ".join((turn.text or "").strip().lower().split())
        if not text:
            return PendingConfirmationDecision(action="ignore")
        confirm_markers = {
            "yes",
            "ok",
            "okay",
            "sure",
            "do it",
            "go ahead",
            "confirm",
            "start",
            "begin",
            "\u597d",
            "\u597d\u7684",
            "\u53ef\u4ee5",
            "\u884c",
            "\u786e\u8ba4",
            "\u662f\u7684",
            "\u6267\u884c",
            "\u5f00\u59cb",
            "\u5f00\u59cb\u5427",
            "\u542f\u52a8",
            "\u542f\u52a8\u5427",
            "\u73b0\u5728\u542f\u52a8",
        }
        decline_markers = {
            "no",
            "nope",
            "don't",
            "do not",
            "cancel",
            "stop",
            "\u4e0d\u7528",
            "\u4e0d\u8981",
            "\u4e0d\u884c",
            "\u53d6\u6d88",
            "\u5148\u522b",
            "\u522b\u505a",
        }
        if any(marker in text for marker in decline_markers):
            return PendingConfirmationDecision(action="decline")
        if any(marker in text for marker in confirm_markers):
            return PendingConfirmationDecision(action="confirm")
        return PendingConfirmationDecision(action="new_task")

    async def submit_skill(self, skill: SkillIntent) -> None:
        logger.info(
            f"{self.agent_id} 下发 skill={skill.skill_id} name={skill.name!r} "
            f"robot={skill.envelope.robot_id} objective_len={len(skill.objective)}"
        )
        await self._interrupt_superseded_skills(skill)
        await self._publish_skill_progress_reply(skill)
        await self.bus.publish(self.topics.skill_intent, to_payload(skill))
        if skill.envelope.episode_id:
            self.task_runtime.task_runs.bind_skill(
                skill.envelope.episode_id,
                skill.skill_id,
                skill.objective,
            )
        await self.events.publish(
            RuntimeEvent.make(
                EventKind.AGENT_SKILL_SUBMITTED,
                source="agent",
                trace_id=skill.envelope.trace_id,
                episode_id=skill.envelope.episode_id,
                agent_id=self.agent_id,
                robot_id=skill.envelope.robot_id,
                payload={"skill_id": skill.skill_id, "name": skill.name},
            )
        )

    async def _interrupt_superseded_skills(self, skill: SkillIntent) -> None:
        episode_id = skill.envelope.episode_id
        if not episode_id:
            return
        task = self.task_runtime.task_runs.load_active(episode_id)
        if task is None:
            return
        superseded = task.metadata.get("_superseded_skill_ids")
        if not isinstance(superseded, list) or not superseded:
            return
        for superseded_skill_id in superseded:
            await self.bus.publish(
                self.topics.skill_intent,
                to_payload(
                    SkillIntent(
                        envelope=skill.envelope,
                        skill_id=str(superseded_skill_id),
                        name="interrupt",
                        objective="task superseded; interrupting active skill",
                        interrupt=True,
                        timeout_sec=3.0,
                        feedback_mode="none",
                    )
                ),
            )
        task.metadata.pop("_superseded_skill_ids", None)
        self.task_runtime.task_runs.save(task)

    def _latest_status_for(self, robot_id: str | None) -> RobotStatus | None:
        resolved = robot_id or self.config.default_robot_id(self.agent_id) or ""
        return self.latest_status.get(resolved)

    async def _publish_skill_progress_reply(self, skill: SkillIntent) -> None:
        if not _should_publish_skill_progress_reply(skill):
            return
        text = _skill_progress_text(skill)
        if not text:
            return
        await self.publish_reply(
            AgentReply(
                envelope=skill.envelope,
                text=text,
                final=False,
                metadata={
                    "source": "skill_progress",
                    "tool": "request_capability",
                    "skill_id": skill.skill_id,
                    "skill_name": skill.name,
                },
            )
        )

    async def _publish_skill_event(self, event: SkillEvent) -> None:
        await self.bus.publish(self.topics.skill_event, to_payload(event))

    async def _publish_skill_intent(self, skill: SkillIntent) -> None:
        await self.bus.publish(self.topics.skill_intent, to_payload(skill))

    async def _publish_agent_progress(self, progress: RobotAgentProgress) -> None:
        await self.bus.publish(
            self.topics.runtime_event,
            RuntimeEvent.make(
                "agent.progress",
                source="agent",
                trace_id=progress.trace_id,
                episode_id=progress.episode_id,
                agent_id=progress.agent_id,
                robot_id=progress.robot_id,
                payload=progress.to_dict(),
            ).to_dict(),
        )

    async def publish_reply(self, reply: AgentReply) -> None:
        logger.info(
            f"{self.agent_id} 发布 reply trace={reply.envelope.trace_id} text_len={len(reply.text)}"
        )
        await self.bus.publish(self.topics.agent_reply, to_payload(reply))

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
    ) -> None:
        await self.notification_runtime.publish(
            text,
            base_envelope=self._current_turn_envelope(),
            channel=channel,
            chat_id=chat_id,
            sender_id=sender_id,
            message_id=message_id,
            reply_to_current=reply_to_current,
            metadata=metadata,
        )

    def _current_turn_envelope(self):
        base_turn = getattr(self.core, "_current_turn", None)
        return base_turn.envelope if base_turn is not None else None

    async def publish_task_result(self, *, success: bool, summary: str) -> None:
        turn = getattr(self.core, "_current_turn", None)
        episode_id = turn.envelope.episode_id if turn is not None else None
        if episode_id:
            self.task_runtime.task_runs.mark_task_reported(
                episode_id,
                success=bool(success),
                summary=summary,
            )
        await self.publish_reply(
            AgentReply(
                envelope=getattr(self.core, "_current_turn").envelope,
                text=summary,
                metadata={"task_success": bool(success)},
            )
        )

    async def query_scene_evidence(
        self,
        *,
        robot_id: str | None,
        question: str,
        baseline_frame_id: int | None = None,
        freshness: str = "fresh",
        timeout_sec: float = 2.0,
    ) -> SceneEvidence:
        return await self.scene_runtime.query_scene_evidence(
            robot_id=robot_id,
            default_robot=self.config.default_robot_id(self.agent_id),
            question=question,
            baseline_frame_id=baseline_frame_id,
            freshness=freshness,
            timeout_sec=timeout_sec,
        )

    def _snapshot(self, robot_id: str | None) -> RobotSnapshot:
        return self.robot_cache.snapshot(
            robot_id, default_robot=self.config.default_robot_id(self.agent_id)
        )

    def _episode_history(self, episode_id: str | None, *, limit: int = 12):
        if not episode_id:
            return []
        return self.episodes.history(episode_id, limit=limit)

    def _lock(self, locks: dict[str, asyncio.Lock], key: str) -> asyncio.Lock:
        lock = locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            locks[key] = lock
        return lock

    async def _evaluate_execution_feedback(
        self, result: SkillResult
    ) -> ExecutionFeedback:
        snapshot = self._snapshot(result.envelope.robot_id)
        target = self.task_runtime.feedback_target_for(result)
        return await self.core.feedback_evaluator.evaluate(
            task=target.task,
            skill_objective=target.skill_objective,
            result=result,
            snapshot=snapshot,
            mode=target.mode,  # type: ignore[arg-type]
        )

    async def _commit_execution_feedback(
        self,
        result: SkillResult,
        feedback: ExecutionFeedback,
    ):
        return self.task_runtime.commit_execution_feedback(
            result=result, feedback=feedback
        )

    async def _publish_execution_feedback_event(
        self, result: SkillResult, feedback: ExecutionFeedback
    ) -> None:
        await self._publish_skill_event(
            SkillEvent(
                envelope=result.envelope,
                skill_id=result.skill_id,
                phase="confirmed" if feedback.successful else "feedback_failed",
                summary=feedback.summary,
                metadata=feedback.to_dict(),
            )
        )

    async def _handle_busy_turn(self, turn: UserTurn, *, active_skill_id: str) -> bool:
        return await self.busy_turns.handle(turn, active_skill_id=active_skill_id)

    async def _continue_pending_turn_for_episode(self, episode_id: str) -> None:
        if not episode_id:
            return
        pending = self.task_runtime.pop_pending_turn(episode_id)
        if pending is None:
            logger.debug(f"{self.agent_id} 无待处理 turn，episode={episode_id}")
            self.task_runtime.clear_checkpoint_if_terminal(episode_id)
            return
        logger.info(
            f"{self.agent_id} 继续处理待执行 turn trace={pending.envelope.trace_id} episode={episode_id}"
        )
        robot_key = (
            pending.envelope.robot_id
            or self.config.default_robot_id(self.agent_id)
            or "none"
        )
        episode_key = pending.envelope.episode_id or pending.envelope.trace_id
        async with (
            self._lock(self._episode_locks, episode_key),
            self._lock(self._robot_locks, robot_key),
        ):
            await self._handle_user_turn_locked(pending, replayed=True)


def _confirmed_capability_reply(capability: str) -> str:
    if capability == "human_follow":
        return "好的，已启动跟随模式。"
    return "好的，已开始执行。"


def _confirmed_capability_failure_reply(capability: str) -> str:
    if capability == "human_follow":
        return "跟随模式没有成功启动，我会保持原地。"
    return "这个动作没有成功启动。"


def _should_publish_skill_progress_reply(skill: SkillIntent) -> bool:
    return not _looks_like_observation_skill(skill)


def _skill_progress_text(skill: SkillIntent) -> str:
    if skill.name in {"move_base", "turn_base"}:
        return "先调整一下底盘位置。"
    if _looks_like_observation_skill(skill):
        return "先观察一下当前场景。"
    if skill.interrupt or skill.name == "interrupt":
        return "先停止当前动作。"
    objective = _compact_text(skill.objective or skill.name or "当前任务", max_chars=36)
    return f"开始处理：{objective}。"


def _looks_like_observation_skill(skill: SkillIntent) -> bool:
    if is_perception_skill_name(skill.name):
        return True
    text = f"{skill.name} {skill.objective}".lower()
    observation_markers = (
        "front view",
        "current scene",
        "look",
        "see",
        "observe",
        "inspect",
        "camera",
        "看",
        "观察",
        "前方",
        "相机",
        "场景",
    )
    return any(marker in text for marker in observation_markers)


def _compact_text(text: str, *, max_chars: int) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"
