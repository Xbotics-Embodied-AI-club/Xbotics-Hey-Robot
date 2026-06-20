from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hey_robot.tasks.recovery import TaskRecoveryDecision, TaskRecoveryPlanner

from hey_robot.agents.checkpoint import RobotAgentCheckpointStore
from hey_robot.agents.execution_feedback import ExecutionFeedback
from hey_robot.agents.injection import RobotTurnInjector
from hey_robot.agents.runtime.response_policy import looks_like_internal_agent_protocol
from hey_robot.agents.task_run import TaskRun, TaskRunStore
from hey_robot.agents.types import RobotSnapshot
from hey_robot.episode import EpisodeRecord, RobotEpisodeStateStore
from hey_robot.logging import HeyRobotLogger
from hey_robot.memory import MemoryBroker, SceneMemoryRecord, SceneMemoryStore

logger = HeyRobotLogger(name="agent.task_runtime")
from hey_robot.protocol import (
    RobotObservation,
    RobotStatus,
    SkillEvent,
    SkillResult,
    UserTurn,
)


@dataclass(frozen=True)
class BuiltTaskTurn:
    turn: UserTurn
    task: TaskRun | None
    pending_turns: list[UserTurn]
    memory_context: str | None
    recovery_context: str | None
    metadata: dict[str, Any]
    block_actuation: bool = False


@dataclass(frozen=True)
class FeedbackEvaluationTarget:
    task: str
    skill_objective: str
    mode: str
    task_run: TaskRun | None


@dataclass(frozen=True)
class RecoveryResumeGuidance:
    strategy: str
    summary: str
    recommended_tools: tuple[str, ...]
    continuation_guidance: str


class RobotStateCache:
    """In-memory cache for latest robot runtime signals.

    Durable task state belongs to TaskRunManager. This cache is only the current
    sensor/status view needed to build snapshots and answer read-only turns.
    """

    def __init__(self) -> None:
        self.status: dict[str, RobotStatus] = {}
        self.observation: dict[str, RobotObservation] = {}
        self.skill_result: dict[str, SkillResult] = {}
        self.skill_event_frame: dict[str, int] = {}

    def snapshot(
        self, robot_id: str | None, *, default_robot: str | None = None
    ) -> RobotSnapshot:
        resolved = robot_id or default_robot or ""
        return RobotSnapshot(
            robot_id=resolved,
            status=self.status.get(resolved),
            observation=self.observation.get(resolved),
            skill_result=self.skill_result.get(resolved),
        )


class RecoveryManager:
    """Single recovery decision boundary for skill, feedback, and watchdog failures.

    Tracks per-episode recovery attempts and escalates when the same failure
    mode repeats without resolution — preventing endless autonomous retry loops.
    """

    _MAX_AUTO_RETRIES = 2

    def __init__(self, planner: TaskRecoveryPlanner | None = None) -> None:
        if planner is None:
            from hey_robot.tasks.recovery import TaskRecoveryPlanner

            planner = TaskRecoveryPlanner()
        self.planner = planner
        self._attempts: dict[str, int] = {}
        self._last_failure: dict[str, str] = {}

    def decide_for_skill_result(
        self,
        *,
        task: TaskRun | None,
        result: SkillResult,
        status: RobotStatus | None,
        health_reports: tuple[dict[str, Any], ...] = (),
    ) -> TaskRecoveryDecision:
        decision = self.planner.decide(
            task=task,
            result=result,
            status=status,
            health_reports=health_reports,
        )
        if not decision.needed or not task:
            return decision
        episode_id = task.episode_id
        if not episode_id:
            return decision
        failure_key = self._failure_key(result)
        if failure_key == self._last_failure.get(episode_id):
            self._attempts[episode_id] = self._attempts.get(episode_id, 0) + 1
        else:
            self._attempts[episode_id] = 1
            self._last_failure[episode_id] = failure_key
        attempts = self._attempts[episode_id]
        if attempts > self._MAX_AUTO_RETRIES:
            return self._escalate(decision, attempts, task)
        return decision

    def clear(self, episode_id: str | None) -> None:
        if episode_id:
            self._attempts.pop(episode_id, None)
            self._last_failure.pop(episode_id, None)

    @staticmethod
    def apply(
        store: TaskRunStore,
        *,
        episode_id: str | None,
        decision: TaskRecoveryDecision,
    ) -> TaskRun | None:
        if not episode_id or not decision.needed:
            return None
        return store.set_recovery(
            episode_id,
            strategy=decision.strategy,
            summary=decision.summary,
            metadata=decision.to_dict(),
        )

    @staticmethod
    def _failure_key(result: SkillResult) -> str:
        failure_mode = str(
            result.failure_mode or result.metadata.get("failure_mode") or ""
        ).strip()
        status = str(result.status or "").strip()
        return f"{status}:{failure_mode}" if failure_mode else status

    @staticmethod
    def _escalate(
        decision: TaskRecoveryDecision, attempts: int, task: TaskRun
    ) -> TaskRecoveryDecision:
        from hey_robot.tasks.recovery import TaskRecoveryDecision

        if attempts >= 5:
            return TaskRecoveryDecision(
                needed=True,
                strategy="safe_abort",
                summary=f"Recovery failed {attempts} times for task '{task.root_task}'. "
                "Stopping autonomous execution and waiting for operator intervention.",
                severity="operator_required",
                actions=("hold_task", "stop_motion", "ask_operator"),
                operator_required=True,
                retryable=False,
                metadata={
                    **decision.metadata,
                    "escalated_from": decision.strategy,
                    "recovery_attempts": attempts,
                },
            )
        return TaskRecoveryDecision(
            needed=True,
            strategy="ask_operator",
            summary=f"Recovery attempted {attempts} times for task '{task.root_task}'. "
            "Pausing for operator guidance before retrying.",
            severity="operator_required",
            actions=("hold_task", "ask_operator"),
            operator_required=True,
            retryable=True,
            metadata={
                **decision.metadata,
                "escalated_from": decision.strategy,
                "recovery_attempts": attempts,
            },
        )


class TaskRunManager:
    """The task-runtime source of truth for agent turns.

    This class is the bridge between the agent brain and robot execution. It owns
    task state, pending-turn checkpoints, scene-memory writes, and recovery
    decisions so RobotAgentService can stay focused on bus orchestration.
    """

    def __init__(
        self,
        *,
        episode_root: str | Path,
        runtime_dir: str | Path,
        events_max_items: int,
        robot_states: RobotEpisodeStateStore,
    ) -> None:
        self.checkpoints = RobotAgentCheckpointStore(episode_root)
        self.task_runs = TaskRunStore(episode_root)
        from hey_robot.tasks.episode import TaskEpisodeRuntime

        self.task_episodes = TaskEpisodeRuntime(self.task_runs)
        self.scene_memory = SceneMemoryStore(
            Path(runtime_dir) / "scene_memory", max_items=events_max_items
        )
        self.memory = MemoryBroker(
            scene_memory=self.scene_memory, task_events=self.task_runs.events
        )
        self.recovery = RecoveryManager()
        self.robot_states = robot_states

    def mark_recovery_required_for_nonterminal(self) -> None:
        self.robot_states.mark_recovery_required_for_nonterminal()

    def observe_status(self, status: RobotStatus) -> None:
        if status.envelope.episode_id:
            self.robot_states.update_status(status.envelope.episode_id, status)
        elif status.envelope.robot_id:
            self.robot_states.update_status_for_robot(status.envelope.robot_id, status)

    def observe_observation(self, observation: RobotObservation) -> None:
        if observation.envelope.episode_id:
            self.robot_states.update_observation(
                observation.envelope.episode_id, observation
            )

    def observe_skill_event(self, event: SkillEvent) -> None:
        self.robot_states.apply_skill_event(event)

    def observe_robot_skill_result(self, result: SkillResult) -> None:
        self.robot_states.apply_skill_result(result)

    def mark_task_started(self, turn: UserTurn, *, agent_id: str) -> None:
        if not turn.envelope.episode_id:
            return
        task_text = str(
            (turn.metadata.get("_confirmed_proposal") or {}).get("objective")
            or turn.text
        )
        self.robot_states.mark_task_started(
            turn.envelope.episode_id,
            task=task_text,
            agent_id=agent_id,
            robot_id=turn.envelope.robot_id,
        )

    def resume_episode_task(
        self, episode_id: str | None, *, operator: str | None = None
    ) -> TaskRun | None:
        if not episode_id:
            return None
        task = self.task_runs.resume(episode_id, operator=operator)
        if task is None:
            return None
        self.robot_states.clear_recovery(episode_id, task=task.root_task)
        self.recovery.clear(episode_id)
        return task

    def mark_restore(self, turn: UserTurn) -> None:
        if turn.envelope.episode_id:
            self.checkpoints.mark_phase(turn.envelope.episode_id, phase="restore")

    def build_turn(
        self,
        *,
        turn: UserTurn,
        snapshot: RobotSnapshot,
        history: list[EpisodeRecord],
        recovery_context: str | None,
        context_builder: Any,
        injector: RobotTurnInjector,
    ) -> BuiltTaskTurn:
        episode_id = turn.envelope.episode_id
        pending = self.checkpoints.pending_turns(episode_id)
        intent = _intent_from_turn_metadata(turn)
        existing_task = self.task_runs.load_active(episode_id) if episode_id else None
        effective_recovery_context = recovery_context
        recovery_resume_guidance: RecoveryResumeGuidance | None = None
        task_text = _task_text_from_turn(turn)
        if existing_task is not None and _should_reuse_existing_task(
            turn=turn, intent=intent
        ):
            task_text = existing_task.root_task or task_text
        task_run = None
        if episode_id:
            task_run = self.task_runs.ensure_active(
                episode_id=episode_id,
                task=task_text,
                agent_id=turn.envelope.agent_id,
                robot_id=turn.envelope.robot_id,
            )
            if (
                task_run is not None
                and existing_task is not None
                and existing_task.status in {"recovering", "paused"}
                and intent is not None
                and intent.kind == "follow_up"
            ):
                recovery_resume_guidance = self._resume_guidance_for_task(existing_task)
                resumed = self.resume_episode_task(episode_id)
                if resumed is not None:
                    task_run = resumed
                    effective_recovery_context = None
        turn_metadata = dict(turn.metadata)
        if recovery_resume_guidance is not None:
            turn_metadata["_recovery_resume"] = {
                "strategy": recovery_resume_guidance.strategy,
                "summary": recovery_resume_guidance.summary,
                "recommended_tools": list(recovery_resume_guidance.recommended_tools),
                "continuation_guidance": recovery_resume_guidance.continuation_guidance,
            }
            turn = UserTurn(
                envelope=turn.envelope,
                text=turn.text,
                media=turn.media,
                intent=turn.intent,
                metadata=turn_metadata,
            )
        injected = injector.inject(
            turn=turn,
            intent=intent,
            task=task_run,
            snapshot=snapshot,
        )
        if injected.metadata:
            turn = UserTurn(
                envelope=turn.envelope,
                text=injected.text,
                media=turn.media,
                intent=turn.intent,
                metadata={**turn.metadata, "_injection": injected.metadata},
            )
        built = context_builder.build(
            turn=turn,
            snapshot=snapshot,
            history=history,
            recovery_context=effective_recovery_context,
            pending_turns=pending,
            task=task_run,
        )
        metadata = dict(built.metadata)
        scene_context = self.memory.build(task=task_run, skill_catalog_context=None)
        if scene_context:
            metadata["scene_context"] = scene_context
        if episode_id:
            self.checkpoints.mark_phase(episode_id, phase="build")
        return BuiltTaskTurn(
            turn=turn,
            task=task_run,
            pending_turns=pending,
            memory_context=_join_contexts(built.memory_context(), scene_context),
            recovery_context=built.recovery_context,
            block_actuation=_compute_block_actuation(task_run),
            metadata=metadata,
        )

    def observe_turn_result(
        self,
        *,
        turn: UserTurn,
        result_tool: str,
        reply_text: str | None,
        task_finished: bool,
        skill_id: str | None,
        last_observation_frame: int | None,
    ) -> None:
        episode_id = turn.envelope.episode_id
        if not episode_id:
            return
        if result_tool in {"request_capability", "final_response"} and skill_id:
            active_task = self.task_runs.load_active(episode_id)
            already_bound = bool(
                active_task
                and any(
                    attempt.skill_id == str(skill_id)
                    for attempt in active_task.attempts
                )
            )
            if not already_bound:
                self.task_runs.bind_skill(
                    episode_id,
                    skill_id=str(skill_id),
                    objective=turn.text,
                )
        self.checkpoints.mark_phase(
            episode_id,
            phase="task_finished" if task_finished else "responded",
            skill_id=skill_id,
        )
        del last_observation_frame
        latest_feedback_task_success = self._latest_feedback_task_success(
            episode_id, skill_id
        )
        reply_is_internal = bool(
            reply_text and looks_like_internal_agent_protocol(reply_text)
        )
        if (
            result_tool == "final_response"
            and reply_text
            and skill_id
            and not reply_is_internal
        ):
            self.task_runs.mark_skill_completed(
                episode_id,
                skill_id=str(skill_id),
                summary=reply_text,
                success=True,
            )
        if reply_text and task_finished and result_tool != "task_safety":
            if latest_feedback_task_success is False:
                if result_tool == "final_response" and not reply_is_internal:
                    self.task_runs.mark_task_reported(
                        episode_id,
                        success=True,
                        summary=reply_text,
                    )
                else:
                    self.checkpoints.mark_phase(
                        episode_id,
                        phase="continuing",
                        skill_id=skill_id,
                    )
            else:
                task_success = (
                    bool(latest_feedback_task_success)
                    if latest_feedback_task_success is not None
                    else not self._has_unresolved_recovery(episode_id)
                )
                logger.info(
                    f"标记任务完成：episode={episode_id} "
                    f"task_success={task_success} "
                    f"latest_feedback_task_success={latest_feedback_task_success} "
                    f"result_tool={result_tool}"
                )
                self.task_runs.mark_task_reported(
                    episode_id,
                    success=task_success,
                    summary=reply_text,
                )
        if result_tool == "task_safety":
            self.task_runs.mark_task_reported(
                episode_id,
                success=False,
                summary=reply_text or "task blocked by safety policy",
            )

    def save_turn(self, *, episode_id: str | None, task_finished: bool) -> None:
        if episode_id and task_finished:
            self.checkpoints.clear_if_terminal(episode_id)
            self.recovery.clear(episode_id)

    def flush_episode_state(
        self,
        *,
        episode_id: str | None,
        status: RobotStatus | None,
        observation: RobotObservation | None,
    ) -> None:
        if not episode_id:
            return
        if status is not None:
            self.robot_states.update_status(episode_id, status)
        if observation is not None:
            self.robot_states.update_observation(episode_id, observation)

    def clear_for_new_turn(self, episode_id: str | None) -> None:
        if episode_id:
            self.checkpoints.reset_for_external_turn(episode_id)

    def enqueue_pending_turn(
        self,
        turn: UserTurn,
        *,
        reason: str,
        intent: dict[str, Any] | None,
        active_skill_id: str | None,
    ) -> None:
        self.checkpoints.enqueue_pending_turn(
            turn,
            reason=reason,
            intent=intent,
            active_skill_id=active_skill_id,
        )
        episode_id = turn.envelope.episode_id
        if not episode_id:
            return
        self.task_runs.append_attempt(
            episode_id,
            event="pending_turn_queued",
            summary=turn.text,
            skill_id=active_skill_id,
            metadata={
                "reason": reason,
                "intent": dict(intent or {}),
                "trace_id": turn.envelope.trace_id,
                "message_id": turn.envelope.message_id,
            },
        )

    def pop_pending_turn(self, episode_id: str) -> UserTurn | None:
        turn = self.checkpoints.pop_pending_turn(episode_id)
        if turn is None:
            return None
        self.task_runs.append_attempt(
            episode_id,
            event="pending_turn_replayed",
            summary=turn.text,
            skill_id=str((turn.metadata.get("_active_skill_id") or "") or None),
            metadata={
                "reason": turn.metadata.get("_pending_reason"),
                "intent": dict(turn.metadata.get("_interaction_intent") or {}),
                "trace_id": turn.envelope.trace_id,
                "message_id": turn.envelope.message_id,
            },
        )
        return turn

    def clear_checkpoint_if_terminal(self, episode_id: str) -> None:
        self.checkpoints.clear_if_terminal(episode_id)

    def pending_confirmation(self, episode_id: str | None) -> dict[str, Any] | None:
        if not episode_id:
            return None
        task = self.task_runs.load_active(episode_id)
        if task is None or not isinstance(task.pending_confirmation, dict):
            return None
        return dict(task.pending_confirmation)

    def store_pending_confirmation(
        self, episode_id: str | None, proposal: dict[str, Any]
    ) -> None:
        if not episode_id:
            return
        self.task_runs.set_pending_confirmation(
            episode_id,
            capability=str(proposal.get("capability") or ""),
            objective=str(proposal.get("objective") or ""),
            prompt=str(proposal.get("prompt") or ""),
            slots=proposal.get("slots")
            if isinstance(proposal.get("slots"), dict)
            else None,
            interrupt=bool(proposal.get("interrupt")),
            proposal_id=str(proposal.get("proposal_id") or "") or None,
            agent_id=str(proposal.get("agent_id") or "") or None,
            robot_id=str(proposal.get("robot_id") or "") or None,
        )

    def clear_pending_confirmation(self, episode_id: str | None) -> None:
        if not episode_id:
            return
        self.task_runs.clear_pending_confirmation(episode_id)

    def record_scene_memory(self, record: SceneMemoryRecord) -> None:
        self.memory.append_scene(record, task_runs=self.task_runs)

    def observe_skill_result(self, result: SkillResult) -> TaskRun | None:
        task = self.task_episodes.observe_skill_result(result)
        episode_id = result.envelope.episode_id
        if not episode_id:
            return task
        metadata = dict(result.metadata or {})
        self.task_runs.bind_skill_trace_metadata(
            episode_id,
            skill_id=result.skill_id,
            status=result.status,
            success=result.success,
            summary=result.summary,
            metadata=metadata,
        )
        return task

    def observe_execution_feedback(
        self,
        *,
        episode_id: str | None,
        feedback: ExecutionFeedback,
    ) -> TaskRun | None:
        if not episode_id:
            return None
        return self.task_episodes.observe_execution_feedback(episode_id, feedback)

    def commit_execution_feedback(
        self,
        *,
        result: SkillResult,
        feedback: ExecutionFeedback,
    ) -> TaskRun | None:
        episode_id = result.envelope.episode_id
        if not episode_id:
            return None
        self.robot_states.mark_execution_feedback(
            episode_id,
            skill_id=result.skill_id,
            subgoal_success=bool(feedback.subgoal_success),
            task_success=bool(feedback.task_success),
            summary=feedback.summary,
            next_hint=feedback.next_hint,
        )
        self.checkpoints.mark_execution_feedback(
            episode_id,
            skill_id=result.skill_id,
            success=feedback.successful,
            summary=feedback.summary,
            metadata=feedback.to_dict(),
        )
        return self.observe_execution_feedback(episode_id=episode_id, feedback=feedback)

    def feedback_target_for(self, result: SkillResult) -> FeedbackEvaluationTarget:
        task_run = (
            self.task_runs.load_active(result.envelope.episode_id)
            if result.envelope.episode_id
            else None
        )
        objective = result.summary or result.name or result.skill_id
        if task_run is not None:
            attempt = next(
                (
                    item
                    for item in reversed(task_run.attempts)
                    if item.skill_id == result.skill_id
                ),
                None,
            )
            if attempt is not None:
                objective = attempt.objective or attempt.text or objective
        contract = result.metadata.get("contract")
        mode = str(
            contract.get("feedback_mode") if isinstance(contract, dict) else "status"
        )
        return FeedbackEvaluationTarget(
            task=task_run.root_task if task_run is not None else objective,
            skill_objective=objective,
            mode=mode,
            task_run=task_run,
        )

    def decide_recovery(
        self,
        *,
        task: TaskRun | None,
        result: SkillResult,
        status: RobotStatus | None,
        health_reports: tuple[dict[str, Any], ...] = (),
    ) -> TaskRecoveryDecision:
        return self.recovery.decide_for_skill_result(
            task=task,
            result=result,
            status=status,
            health_reports=health_reports,
        )

    def apply_recovery(
        self,
        *,
        episode_id: str | None,
        decision: TaskRecoveryDecision,
    ) -> TaskRun | None:
        return self.recovery.apply(
            self.task_runs, episode_id=episode_id, decision=decision
        )

    def recovery_context(self, episode_id: str | None) -> str | None:
        if not episode_id:
            return None
        state = self.robot_states.load(episode_id)
        return state.recovery_context() if state is not None else None

    def result_text_for_agent(
        self,
        *,
        result: SkillResult,
        feedback: ExecutionFeedback | None,
    ) -> str | None:
        if feedback is not None:
            base = feedback.for_agent()
            continuation = self._continuation_text_after_recovery(
                result=result, feedback=feedback
            )
            if continuation is None:
                continuation = self._continuation_text_after_successful_step(
                    result=result, feedback=feedback
                )
            trace = self._skill_trace_context(result.envelope.episode_id)
            return _join_contexts(base, continuation, trace)
        return None

    def _resume_guidance_for_task(self, task: TaskRun) -> RecoveryResumeGuidance | None:
        recovery = task.recovery if isinstance(task.recovery, dict) else None
        if not recovery:
            return None
        strategy = (
            str(recovery.get("strategy") or "reobserve").strip().lower() or "reobserve"
        )
        summary = str(
            recovery.get("summary")
            or task.failure_reason
            or "Resolve the last recovery step first."
        )
        return RecoveryResumeGuidance(
            strategy=strategy,
            summary=summary,
            recommended_tools=_recommended_tools_for_resume(strategy),
            continuation_guidance=_continuation_guidance_for_resume(strategy),
        )

    def _continuation_text_after_recovery(
        self,
        *,
        result: SkillResult,
        feedback: ExecutionFeedback,
    ) -> str | None:
        if not feedback.successful:
            return None
        match = self._task_and_attempt_for_skill(
            result.envelope.episode_id, result.skill_id
        )
        if match is None:
            return None
        task, attempt = match
        recovery_resume = attempt.metadata.get("recovery_resume")
        if not isinstance(recovery_resume, dict):
            return None
        strategy = str(recovery_resume.get("strategy") or "reobserve")
        summary = str(recovery_resume.get("summary") or "").strip()
        guidance = str(
            recovery_resume.get("continuation_guidance")
            or "Use the successful recovery step to continue the original task."
        ).strip()
        scene_lines = self._scene_evidence_lines(task)
        runtime_lines = self._runtime_evidence_lines(
            result.envelope.episode_id, feedback=feedback
        )
        lines = [
            "Recovery continuation:",
            "- recovery_step_completed: true",
            f"- original_task: {task.root_task or 'unknown'}",
            f"- recovery_strategy: {strategy}",
        ]
        if summary:
            lines.append(f"- resolved_recovery_step: {summary}")
        lines.append(f"- continuation_guidance: {guidance}")
        lines.extend(scene_lines)
        lines.extend(runtime_lines)
        lines.append(
            "- next_instruction: Use the latest feedback, status, and observation to continue the original task now."
        )
        return "\n".join(lines)

    def _continuation_text_after_successful_step(
        self,
        *,
        result: SkillResult,
        feedback: ExecutionFeedback,
    ) -> str | None:
        if not feedback.successful or bool(feedback.task_success):
            return None
        match = self._task_and_attempt_for_skill(
            result.envelope.episode_id, result.skill_id
        )
        if match is None:
            return None
        task, attempt = match
        if isinstance(attempt.metadata.get("recovery_resume"), dict):
            return None
        completed_step = str(
            attempt.objective or attempt.text or result.summary or result.skill_id
        ).strip()
        if not completed_step:
            completed_step = result.skill_id
        scene_lines = self._scene_evidence_lines(task)
        runtime_lines = self._runtime_evidence_lines(
            result.envelope.episode_id, feedback=feedback
        )
        recent_steps = self._recent_completed_steps(
            task, current_skill_id=result.skill_id
        )
        lines = [
            "Task continuation:",
            "- step_completed: true",
            f"- original_task: {task.root_task or 'unknown'}",
            f"- completed_step: {completed_step}",
            f"- latest_step_summary: {feedback.summary}",
        ]
        if feedback.next_hint:
            lines.append(f"- continuation_focus: {feedback.next_hint}")
        lines.append(
            "- remaining_goal: Continue advancing the original task until you can report completion."
        )
        if recent_steps:
            lines.append(f"- recent_completed_steps: {' | '.join(recent_steps)}")
        lines.extend(scene_lines)
        lines.extend(runtime_lines)
        lines.append(
            "- next_instruction: Use the latest feedback and scene evidence to decide the next useful tool call now."
        )
        return "\n".join(lines)

    def _task_and_attempt_for_skill(
        self,
        episode_id: str | None,
        skill_id: str,
    ) -> tuple[TaskRun, Any] | None:
        if not episode_id:
            return None
        for task in self.task_runs.list_for_episode(episode_id):
            for attempt in reversed(task.attempts):
                if attempt.skill_id == skill_id:
                    return task, attempt
        return None

    def _latest_feedback_task_success(
        self,
        episode_id: str | None,
        skill_id: str | None,
    ) -> bool | None:
        if not episode_id:
            return None
        task = self.task_runs.load_active(episode_id)
        if task is None:
            return None
        attempts = list(reversed(task.attempts))
        if skill_id:
            attempts = [item for item in attempts if item.skill_id == skill_id]
        for attempt in attempts:
            feedback = attempt.metadata.get("execution_feedback")
            if isinstance(feedback, dict) and "task_success" in feedback:
                return bool(feedback.get("task_success"))
        return None

    def _scene_evidence_lines(self, task: TaskRun) -> list[str]:
        lines: list[str] = []
        understanding: dict[str, Any] | None = None
        latest = task.metadata.get("latest_scene_event")
        if isinstance(latest, dict):
            frame_id = latest.get("frame_id")
            summary = str(latest.get("summary") or "").strip()
            if frame_id is not None:
                lines.append(f"- latest_scene_frame: {frame_id}")
            if summary:
                lines.append(f"- latest_scene_summary: {summary}")
            metadata = latest.get("metadata")
            if isinstance(metadata, dict):
                raw_understanding = metadata.get("understanding")
                if isinstance(raw_understanding, dict):
                    understanding = raw_understanding
        recent_records = self.scene_memory.recent(task.episode_id, limit=1)
        if recent_records:
            record = recent_records[0]
            if record.frame_id is not None and not any(
                line.startswith("- latest_scene_frame:") for line in lines
            ):
                lines.append(f"- latest_scene_frame: {record.frame_id}")
            if record.summary and not any(
                line.startswith("- latest_scene_summary:") for line in lines
            ):
                lines.append(f"- latest_scene_summary: {record.summary}")
            if understanding is None:
                raw_understanding = record.metadata.get("understanding")
                if isinstance(raw_understanding, dict):
                    understanding = raw_understanding
        if understanding is not None:
            hint = str(
                understanding.get("next_observation_hint")
                or understanding.get("next_hint")
                or ""
            ).strip()
            risks = understanding.get("risks")
            if isinstance(risks, list):
                risk_items = [str(item).strip() for item in risks if str(item).strip()]
                if risk_items:
                    lines.append(f"- latest_scene_risks: {', '.join(risk_items)}")
            if hint:
                lines.append(f"- latest_scene_hint: {hint}")
        return lines

    def _runtime_evidence_lines(
        self,
        episode_id: str | None,
        *,
        feedback: ExecutionFeedback,
    ) -> list[str]:
        lines: list[str] = []
        if feedback.next_hint:
            lines.append(f"- feedback_next_hint: {feedback.next_hint}")
        if not episode_id:
            return lines
        state = self.robot_states.load(episode_id)
        if state is None:
            return lines
        if state.last_observation_frame is not None:
            lines.append(
                f"- runtime_last_observation_frame: {state.last_observation_frame}"
            )
        status = state.last_status if isinstance(state.last_status, dict) else {}
        state_name = str(status.get("state") or "").strip()
        if state_name:
            lines.append(f"- runtime_robot_state: {state_name}")
        execution_feedback = state.last_execution_feedback
        if isinstance(execution_feedback, dict):
            summary = str(execution_feedback.get("summary") or "").strip()
            next_hint = str(execution_feedback.get("next_hint") or "").strip()
            if summary:
                lines.append(f"- runtime_feedback_summary: {summary}")
            if next_hint and not any(
                line == f"- feedback_next_hint: {next_hint}" for line in lines
            ):
                lines.append(f"- runtime_feedback_next_hint: {next_hint}")
        return lines

    def _skill_trace_context(self, episode_id: str | None) -> str | None:
        if not episode_id:
            return None
        task = self.task_runs.load_active(episode_id)
        if task is None:
            tasks = self.task_runs.list_for_episode(episode_id)
            task = tasks[0] if tasks else None
        if task is None or not task.skill_trace:
            return None
        lines = ["Skill trace:"]
        for item in task.skill_trace[-6:]:
            skill_name = str(item.get("skill") or "").strip() or "unknown"
            backend = str(item.get("backend") or "").strip() or "unknown"
            implementation = (
                str(item.get("implementation_name") or "").strip() or "unknown"
            )
            status = str(item.get("status") or "").strip() or "unknown"
            skill_id = str(item.get("skill_id") or "").strip()
            objective = str(item.get("objective") or "").strip()
            summary = str(item.get("summary") or "").strip()
            parts = [
                skill_name,
                f"backend={backend}",
                f"implementation={implementation}",
                f"status={status}",
            ]
            if skill_id:
                parts.append(f"skill_id={skill_id}")
            if objective:
                parts.append(f"objective={objective}")
            if summary:
                parts.append(f"summary={summary}")
            lines.append(f"- {'; '.join(parts)}")
        return "\n".join(lines)

    def _has_unresolved_recovery(self, episode_id: str) -> bool:
        state = self.robot_states.load(episode_id)
        if state is not None and state.recovery_required:
            return True
        task = self.task_runs.load_active(episode_id)
        if task is not None and task.attempts:
            last = task.attempts[-1]
            if last.success is not True:
                return True
        return False

    @staticmethod
    def _recent_completed_steps(task: TaskRun, *, current_skill_id: str) -> list[str]:
        steps: list[str] = []
        for attempt in reversed(task.attempts):
            if attempt.skill_id == current_skill_id:
                continue
            if attempt.success is not True:
                continue
            text = str(attempt.objective or attempt.text or "").strip()
            if not text:
                continue
            steps.append(text)
            if len(steps) >= 2:
                break
        steps.reverse()
        return steps


def _join_contexts(*parts: str | None) -> str | None:
    values = [part for part in parts if part]
    return "\n\n".join(values) if values else None


def _compute_block_actuation(task: TaskRun | None) -> bool:
    """Return True if the task's recovery state blocks new actuation skills."""
    if task is None:
        return False
    if task.status != "recovering":
        return False
    recovery = task.recovery if isinstance(task.recovery, dict) else None
    if not recovery:
        return False
    strategy = str(recovery.get("strategy") or "").strip().lower()
    return strategy not in {"degraded_continue", "none", ""}


def _task_text_from_turn(turn: UserTurn) -> str:
    return str(
        (turn.metadata.get("_confirmed_proposal") or {}).get("objective") or turn.text
    )


def _should_reuse_existing_task(*, turn: UserTurn, intent: Any) -> bool:  # noqa: ARG001
    if intent is None:
        return False
    return intent.kind in {"follow_up", "correction", "interrupt"}


def _intent_from_turn_metadata(turn: UserTurn):
    from hey_robot.agents.interaction import UserInteractionIntent

    raw = turn.metadata.get("_interaction_intent")
    if not isinstance(raw, dict):
        return None
    kind = raw.get("kind")
    if not kind:
        return None
    return UserInteractionIntent(
        kind=str(kind),
        urgency=str(raw.get("urgency") or "normal"),
        target=str(raw.get("target") or "task"),
    )


def _recommended_tools_for_resume(strategy: str) -> tuple[str, ...]:
    mapping = {
        "clarify": ("get_task_context", "wait"),
        "reobserve": ("get_task_context", "request_perception"),
        "reposition": ("get_task_context", "request_perception"),
        "retry_with_adjustment": ("get_task_context", "request_perception"),
        "degraded_continue": ("get_task_context", "get_robot_status"),
        "safe_abort": ("get_task_context", "wait"),
        "ask_operator": ("get_task_context", "wait"),
    }
    return mapping.get(strategy, ("get_task_context", "wait"))


def _attempt_metadata_from_turn(turn: UserTurn) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    recovery_resume = turn.metadata.get("_recovery_resume")
    if isinstance(recovery_resume, dict):
        metadata["recovery_resume"] = dict(recovery_resume)
    auto_recovery_continue = turn.metadata.get("_auto_recovery_continue")
    if isinstance(auto_recovery_continue, dict):
        metadata["auto_recovery_continue"] = dict(auto_recovery_continue)
    return metadata


def _continuation_guidance_for_resume(strategy: str) -> str:
    mapping = {
        "clarify": (
            "Explain the blockage, ask one focused clarification, then wait for the reply before another robot action."
        ),
        "reobserve": (
            "Collect fresh perception first, then use the new evidence to continue the original task."
        ),
        "reposition": (
            "Do not retry immediately; improve viewpoint first, inspect again, then continue the original task."
        ),
        "retry_with_adjustment": (
            "Retry the same skill with adjusted parameters (e.g. different approach angle, grip force, or distance). "
            "Inspect the scene first to decide the adjustment."
        ),
        "degraded_continue": (
            "Continue the original task while working around the degraded resource. "
            "Avoid relying on the degraded capability and prefer alternative approaches."
        ),
        "safe_abort": "Do not continue autonomous actuation; stop and wait for operator intervention.",
        "ask_operator": "Route the task back to the user or operator before continuing.",
    }
    return mapping.get(
        strategy, "Resolve the recovery condition before continuing the original task."
    )
