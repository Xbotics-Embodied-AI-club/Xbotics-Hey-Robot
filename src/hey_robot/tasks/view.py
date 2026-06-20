from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hey_robot.agents.task_run import TaskRun, TaskRunStore
from hey_robot.interaction import InteractionStateStore
from hey_robot.memory.scene import SceneMemoryStore
from hey_robot.skills import SkillStore


@dataclass(frozen=True)
class TaskTimelineItem:
    timestamp: float
    kind: str
    summary: str
    skill_id: str | None = None
    skill_name: str | None = None
    frame_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "kind": self.kind,
            "summary": self.summary,
            "skill_id": self.skill_id,
            "skill_name": self.skill_name,
            "frame_id": self.frame_id,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class TaskSessionView:
    episode_id: str
    task_id: str
    root_task: str
    status: str
    current_phase: str
    current_step: str | None
    active_skill_id: str | None
    active_skill_name: str | None
    skill_progress: float | None
    skill_ux: dict[str, Any] | None
    last_scene_summary: str | None
    latest_evidence: dict[str, Any] | None
    last_feedback_summary: str | None
    current_risk: str | None
    pending_confirmation: dict[str, Any] | None
    interaction_state: dict[str, Any] | None
    recovery_required: bool
    recovery_summary: str | None
    recovery_strategy: str | None = None
    next_recommended_actions: tuple[str, ...] = ()
    operator_actions: tuple[str, ...] = ()
    timeline: tuple[TaskTimelineItem, ...] = ()
    failure_reason: str | None = None
    task_success: bool | None = None
    robot_id: str | None = None
    retry_count: int = 0
    recovery_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "root_task": self.root_task,
            "status": self.status,
            "current_phase": self.current_phase,
            "current_step": self.current_step,
            "active_skill_id": self.active_skill_id,
            "active_skill_name": self.active_skill_name,
            "skill_progress": self.skill_progress,
            "skill_ux": self.skill_ux,
            "last_scene_summary": self.last_scene_summary,
            "latest_evidence": self.latest_evidence,
            "last_feedback_summary": self.last_feedback_summary,
            "current_risk": self.current_risk,
            "pending_confirmation": self.pending_confirmation,
            "interaction_state": self.interaction_state,
            "recovery_required": self.recovery_required,
            "recovery_summary": self.recovery_summary,
            "recovery_strategy": self.recovery_strategy,
            "next_recommended_actions": list(self.next_recommended_actions),
            "operator_actions": list(self.operator_actions),
            "timeline": [item.to_dict() for item in self.timeline],
            "failure_reason": self.failure_reason,
            "task_success": self.task_success,
            "robot_id": self.robot_id,
            "retry_count": self.retry_count,
            "recovery_count": self.recovery_count,
        }


class TaskSessionQueryService:
    """Composes a TaskSessionView from existing durable stores.

    This is a read-only query layer. All writes continue through TaskRunManager
    and the individual stores. This service only composes the unified view.
    """

    def __init__(
        self,
        task_store: TaskRunStore,
        scene_memory: SceneMemoryStore,
        skill_store: SkillStore,
        interaction_store: InteractionStateStore | None = None,
    ) -> None:
        self.task_store = task_store
        self.scene_memory = scene_memory
        self.skill_store = skill_store
        self.interaction_store = interaction_store

    def active_view(self, episode_id: str) -> TaskSessionView | None:
        state = self.task_store.load_active(episode_id)
        if state is None:
            return None
        return self._build_view(state)

    def view_for_episode(self, episode_id: str) -> TaskSessionView | None:
        """Return the active task view, or the latest task view after completion."""

        active = self.active_view(episode_id)
        if active is not None:
            return active
        tasks = self.task_store.list_for_episode(episode_id)
        if not tasks:
            return None
        return self._build_view(tasks[0])

    def _build_view(self, state: TaskRun) -> TaskSessionView:
        active_attempt = _find_active_attempt(state)
        active_skill_id = active_attempt.skill_id if active_attempt else None

        skill_name: str | None = None
        skill_record = None
        if active_skill_id:
            record = self.skill_store.get(active_skill_id)
            skill_record = record
            skill_name = record.name if record else None

        scene_records = self.scene_memory.recent(state.episode_id, limit=1)
        latest_scene = scene_records[0] if scene_records else None
        last_scene_summary = latest_scene.summary if latest_scene else None
        skill_ux = _latest_skill_ux(skill_record)
        latest_evidence = (
            {
                "summary": latest_scene.summary,
                "frame_id": latest_scene.frame_id,
                "confidence": latest_scene.confidence,
                "timestamp": latest_scene.timestamp,
                "metadata": latest_scene.metadata,
            }
            if latest_scene
            else None
        )
        if skill_ux is not None:
            latest_evidence = _merge_skill_evidence(latest_evidence, skill_ux)

        last_feedback_summary: str | None = None
        if active_attempt and "execution_feedback" in active_attempt.metadata:
            fb = active_attempt.metadata["execution_feedback"]
            if isinstance(fb, dict):
                last_feedback_summary = str(fb.get("summary", ""))

        recovery_required = state.status == "recovering"
        recovery_summary = state.recovery.get("summary") if state.recovery else None
        recovery_strategy = state.recovery.get("strategy") if state.recovery else None
        interaction_state = (
            self.interaction_store.get(state.episode_id)
            if self.interaction_store is not None
            else None
        )

        current_step = active_attempt.objective if active_attempt else None

        timeline = _build_timeline(
            state,
            self.scene_memory,
            self.skill_store,
        )

        return TaskSessionView(
            episode_id=state.episode_id,
            task_id=state.task_id,
            root_task=state.root_task,
            status=state.status,
            current_phase=_current_phase(state, active_attempt, skill_record),
            current_step=current_step,
            active_skill_id=active_skill_id,
            active_skill_name=skill_name,
            skill_progress=_skill_progress(state, skill_record),
            skill_ux=skill_ux,
            last_scene_summary=last_scene_summary,
            latest_evidence=latest_evidence,
            last_feedback_summary=last_feedback_summary,
            current_risk=_current_risk(state),
            pending_confirmation=(
                dict(state.pending_confirmation)
                if isinstance(state.pending_confirmation, dict)
                else None
            ),
            interaction_state=(
                interaction_state.to_dict() if interaction_state is not None else None
            ),
            recovery_required=recovery_required,
            recovery_summary=recovery_summary,
            recovery_strategy=recovery_strategy,
            next_recommended_actions=_recommended_actions(state),
            operator_actions=_operator_actions(state),
            timeline=timeline,
            failure_reason=state.failure_reason,
            task_success=state.task_success,
            robot_id=state.robot_id,
            retry_count=state.retry_count,
            recovery_count=state.recovery_count,
        )


def _find_active_attempt(state: TaskRun) -> Any | None:
    if not state.active_attempt_id:
        return None
    for attempt in state.attempts:
        if attempt.attempt_id == state.active_attempt_id:
            return attempt
    return None


def _recommended_actions(state: TaskRun) -> tuple[str, ...]:
    if isinstance(state.pending_confirmation, dict):
        return ("confirm", "decline")
    if state.status == "recovering":
        strategy = state.recovery.get("strategy", "") if state.recovery else ""
        if strategy == "reobserve":
            return ("inspect_scene",)
        if strategy == "reposition":
            return ("inspect_scene", "move_base")
        if strategy == "retry_with_adjustment":
            return ("inspect_scene",)
        if strategy == "degraded_continue":
            return ("get_task_context",)
        if strategy == "ask_operator":
            return ()
    return ()


def _operator_actions(state: TaskRun) -> tuple[str, ...]:
    if state.status in {"completed", "failed", "cancelled"}:
        return ()
    if isinstance(state.pending_confirmation, dict):
        return ("confirm", "decline", "new_task")
    if state.status == "recovering":
        return ("inspect_scene", "resume", "abort")
    if state.status in {"active", "executing", "feedback_pending", "paused"}:
        return ("stop_motion", "inspect_scene")
    return ("inspect_scene",)


def _current_phase(state: TaskRun, active_attempt: Any | None, skill_record) -> str:
    if isinstance(state.pending_confirmation, dict):
        return "awaiting_confirmation"
    if state.status == "recovering":
        return "recovering"
    if skill_record is not None:
        skill_ux = _latest_skill_ux(skill_record)
        if skill_ux and skill_ux.get("phase"):
            return str(skill_ux["phase"])
        return str(skill_record.phase)
    if active_attempt is not None:
        return str(active_attempt.status)
    return state.status


def _skill_progress(state: TaskRun, skill_record) -> float | None:
    if state.task_success is True:
        return 1.0
    if skill_record is None:
        return None
    return max(0.0, min(float(skill_record.progress), 1.0))


def _latest_skill_ux(skill_record) -> dict[str, Any] | None:
    if skill_record is None:
        return None
    for item in reversed(getattr(skill_record, "timeline", []) or []):
        metadata = item.get("metadata") if isinstance(item, dict) else None
        ux = metadata.get("ux") if isinstance(metadata, dict) else None
        if isinstance(ux, dict):
            return dict(ux)
    metadata = getattr(skill_record, "metadata", None)
    ux = metadata.get("ux") if isinstance(metadata, dict) else None
    if isinstance(ux, dict):
        return dict(ux)
    return None


def _merge_skill_evidence(
    latest_evidence: dict[str, Any] | None,
    skill_ux: dict[str, Any],
) -> dict[str, Any]:
    evidence = dict(latest_evidence or {})
    if skill_ux.get("frame_id") is not None:
        evidence["frame_id"] = skill_ux.get("frame_id")
    if skill_ux.get("confidence") is not None:
        evidence["confidence"] = skill_ux.get("confidence")
    evidence["skill_ux"] = skill_ux
    return evidence


def _current_risk(state: TaskRun) -> str | None:
    if state.failure_reason:
        return state.failure_reason
    if isinstance(state.recovery, dict):
        summary = state.recovery.get("summary")
        if summary:
            return str(summary)
    if isinstance(state.pending_confirmation, dict):
        reason = state.pending_confirmation.get("reason")
        if reason:
            return str(reason)
        objective = state.pending_confirmation.get("objective")
        if objective:
            return f"confirmation required: {objective}"
    return None


def _build_timeline(
    state: TaskRun,
    scene_memory: SceneMemoryStore,
    skill_store: SkillStore,
) -> tuple[TaskTimelineItem, ...]:
    items: list[TaskTimelineItem] = []

    scene_records = scene_memory.recent(state.episode_id, limit=20)
    items.extend(
        TaskTimelineItem(
            timestamp=record.timestamp,
            kind="scene_observed",
            summary=record.summary,
            frame_id=record.frame_id,
            metadata={"confidence": record.confidence},
        )
        for record in scene_records
    )

    for attempt in state.attempts:
        skill_name: str | None = None
        if attempt.skill_id:
            skill_record = skill_store.get(attempt.skill_id)
            skill_name = skill_record.name if skill_record else None

        kind_map = {
            "pending": "skill_accepted",
            "executing": "skill_accepted",
            "completed": "skill_completed",
            "failed": "skill_failed",
            "confirmed": "execution_feedback",
            "feedback_failed": "execution_feedback",
        }
        kind = kind_map.get(attempt.status, "skill_accepted")

        items.append(
            TaskTimelineItem(
                timestamp=attempt.created_at,
                kind=kind,
                summary=attempt.objective or attempt.text,
                skill_id=attempt.skill_id,
                skill_name=skill_name,
                metadata={
                    "status": attempt.status,
                    "success": attempt.success,
                },
            )
        )

        if attempt.skill_id:
            skill_record = skill_store.get(attempt.skill_id)
            for event in getattr(skill_record, "timeline", []) if skill_record else []:
                if not isinstance(event, dict):
                    continue
                metadata = event.get("metadata")
                ux = metadata.get("ux") if isinstance(metadata, dict) else None
                if not isinstance(ux, dict):
                    continue
                items.append(
                    TaskTimelineItem(
                        timestamp=event.get("timestamp") or attempt.created_at,
                        kind="skill_progress",
                        summary=event.get("summary") or ux.get("phase") or "",
                        skill_id=attempt.skill_id,
                        skill_name=skill_name,
                        frame_id=ux.get("frame_id"),
                        metadata={"ux": dict(ux), "step": event.get("step")},
                    )
                )

    if state.recovery:
        items.append(
            TaskTimelineItem(
                timestamp=state.recovery.get("timestamp", state.updated_at),
                kind="recovery_selected",
                summary=state.recovery.get("summary", ""),
                metadata={"strategy": state.recovery.get("strategy")},
            )
        )

    task_reported_kind = (
        "task_completed"
        if state.task_success
        else "task_failed"
        if state.task_success is False
        else "task_reported"
    )
    if state.status in {"completed", "failed", "cancelled"}:
        items.append(
            TaskTimelineItem(
                timestamp=state.finished_at or state.updated_at,
                kind=task_reported_kind,
                summary=state.failure_reason or state.root_task,
                metadata={"task_success": state.task_success},
            )
        )

    items.sort(key=lambda item: item.timestamp)
    return tuple(items)
