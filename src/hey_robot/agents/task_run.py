from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hey_robot.agents.task_events import RobotTaskEventLog
from hey_robot.logging import HeyRobotLogger

_logger = HeyRobotLogger(name="agent.task_run")

_ACTIVE_TASK_STATUSES = {
    "active",
    "awaiting_confirmation",
    "executing",
    "feedback_pending",
    "recovering",
    "paused",
}
_TERMINAL_TASK_STATUSES = {"completed", "failed", "cancelled"}


@dataclass
class TaskAttempt:
    attempt_id: str
    text: str
    status: str = "pending"
    skill_id: str | None = None
    objective: str | None = None
    success: bool | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    skill: str | None = None
    backend: str | None = None
    implementation_name: str | None = None
    implementation_kind: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskRun:
    task_id: str
    episode_id: str
    agent_id: str | None = None
    robot_id: str | None = None
    root_task: str = ""
    status: str = "active"
    task_success: bool | None = None
    last_step_success: bool | None = None
    failure_reason: str | None = None
    retry_count: int = 0
    recovery_count: int = 0
    operator_notes: list[dict[str, Any]] = field(default_factory=list)
    watchdog: dict[str, Any] = field(default_factory=dict)
    recovery: dict[str, Any] | None = None
    paused_reason: str | None = None
    pending_confirmation: dict[str, Any] | None = None
    finished_at: float | None = None
    active_attempt_id: str | None = None
    attempts: list[TaskAttempt] = field(default_factory=list)
    skill_ids: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    skill_trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["attempts"] = [attempt.to_dict() for attempt in self.attempts]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskRun:
        attempts = [
            TaskAttempt(**item)
            for item in data.get("attempts", [])
            if isinstance(item, dict)
        ]
        return cls(
            task_id=str(data["task_id"]),
            episode_id=str(data["episode_id"]),
            agent_id=data.get("agent_id"),
            robot_id=data.get("robot_id"),
            root_task=str(data.get("root_task") or ""),
            status=str(data.get("status") or "active"),
            task_success=data.get("task_success"),
            last_step_success=data.get("last_step_success"),
            failure_reason=data.get("failure_reason"),
            retry_count=int(data.get("retry_count") or 0),
            recovery_count=int(data.get("recovery_count") or 0),
            operator_notes=list(data.get("operator_notes", []) or []),
            watchdog=dict(data.get("watchdog", {}) or {}),
            recovery=data.get("recovery"),
            paused_reason=data.get("paused_reason"),
            pending_confirmation=data.get("pending_confirmation"),
            finished_at=data.get("finished_at"),
            active_attempt_id=data.get("active_attempt_id"),
            attempts=attempts,
            skill_ids=[str(item) for item in data.get("skill_ids", [])],
            created_at=float(data.get("created_at") or time.time()),
            updated_at=float(data.get("updated_at") or time.time()),
            metadata=dict(data.get("metadata", {}) or {}),
            skill_trace=list(data.get("skill_trace", []) or []),
        )


class TaskRunStore:
    """Durable task state for loop-first robot execution."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root) / "agent_tasks"
        self.root.mkdir(parents=True, exist_ok=True)
        self.events = RobotTaskEventLog(root)
        self._last_timestamp = 0.0

    def load_active(self, episode_id: str) -> TaskRun | None:
        tasks = [
            task
            for task in self.list_for_episode(episode_id)
            if task.status in _ACTIVE_TASK_STATUSES
        ]
        return tasks[0] if tasks else None

    def ensure_active(
        self,
        *,
        episode_id: str,
        task: str,
        agent_id: str | None,
        robot_id: str | None,
    ) -> TaskRun:
        existing = self.load_active(episode_id)
        if existing is not None:
            if task and existing.root_task != task:
                if existing.status not in _TERMINAL_TASK_STATUSES:
                    _logger.info(
                        f"取消旧任务：episode={episode_id} "
                        f"old_task={existing.root_task!r} old_status={existing.status} "
                        f"new_task={task!r}"
                    )
                    superseded_skill_ids = [
                        attempt.skill_id
                        for attempt in existing.attempts
                        if attempt.skill_id and attempt.status == "executing"
                    ]
                    carried = existing.metadata.get("_superseded_skill_ids")
                    if isinstance(carried, list):
                        for sid in carried:
                            if sid not in superseded_skill_ids:
                                superseded_skill_ids.append(sid)
                    existing.status = "cancelled"
                    existing.failure_reason = "superseded by a new user task"
                    existing.finished_at = time.time()
                    existing.metadata["superseded_by"] = task
                    self.save(existing)
                    new_task = self._create_task(
                        episode_id=episode_id,
                        task=task,
                        agent_id=agent_id,
                        robot_id=robot_id,
                    )
                    if superseded_skill_ids:
                        new_task.metadata["_superseded_skill_ids"] = (
                            superseded_skill_ids
                        )
                        self.save(new_task)
                    return new_task
                # Existing task is terminal — create a new task alongside it
                return self._create_task(
                    episode_id=episode_id,
                    task=task,
                    agent_id=agent_id,
                    robot_id=robot_id,
                )
            if existing.status in {"recovering", "paused"}:
                existing.status = "active"
                existing.paused_reason = None
                existing.failure_reason = None
            return self.save(existing)
        return self._create_task(
            episode_id=episode_id,
            task=task,
            agent_id=agent_id,
            robot_id=robot_id,
        )

    def _create_task(
        self,
        *,
        episode_id: str,
        task: str,
        agent_id: str | None,
        robot_id: str | None,
    ) -> TaskRun:
        task_id = f"task_{uuid.uuid4().hex[:16]}"
        state = TaskRun(
            task_id=task_id,
            episode_id=episode_id,
            agent_id=agent_id,
            robot_id=robot_id,
            root_task=task,
        )
        self.events.append(
            episode_id=episode_id,
            task_id=task_id,
            kind="task_started",
            summary=task,
            metadata={"agent_id": agent_id, "robot_id": robot_id},
        )
        return self.save(state)

    def bind_skill(
        self, episode_id: str, skill_id: str, objective: str
    ) -> TaskRun | None:
        return self.bind_skill_with_metadata(
            episode_id, skill_id, objective, metadata=None
        )

    def bind_skill_with_metadata(
        self,
        episode_id: str,
        skill_id: str,
        objective: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> TaskRun | None:
        state = self.load_active(episode_id)
        if state is None:
            return None
        state.pending_confirmation = None
        if skill_id not in state.skill_ids:
            state.skill_ids.append(skill_id)
        attempt = TaskAttempt(
            attempt_id=f"attempt_{uuid.uuid4().hex[:8]}",
            text=objective or state.root_task,
            status="executing",
            skill_id=skill_id,
            objective=objective or None,
            metadata=dict(metadata or {}),
            skill=_trace_value(metadata, "skill"),
            backend=_trace_value(metadata, "backend"),
            implementation_name=_trace_value(metadata, "implementation_name"),
            implementation_kind=_trace_value(metadata, "implementation_kind"),
        )
        state.attempts.append(attempt)
        state.active_attempt_id = attempt.attempt_id
        if state.status not in _TERMINAL_TASK_STATUSES:
            state.status = "executing"
        self.events.append(
            episode_id=episode_id,
            task_id=state.task_id,
            kind="skill_bound",
            summary=objective,
            skill_id=skill_id,
            metadata={
                "active_attempt_id": state.active_attempt_id,
                **_trace_metadata_from_attempt(attempt),
            },
        )
        self._upsert_skill_trace(
            state,
            skill_id=skill_id,
            status="executing",
            objective=objective,
            metadata=attempt.metadata,
        )
        return self.save(state)

    def mark_execution_feedback(
        self,
        episode_id: str,
        *,
        skill_id: str,
        success: bool,
        summary: str,
    ) -> TaskRun | None:
        state = self.load_active(episode_id)
        if state is None:
            return None
        attempt = self._attempt_for_skill(state, skill_id)
        if attempt is not None:
            attempt.status = "confirmed" if success else "feedback_failed"
            attempt.success = bool(success)
            attempt.metadata["execution_feedback_summary"] = summary
            attempt.updated_at = time.time()
        state.last_step_success = bool(success)
        if success:
            state.status = "active"
            state.failure_reason = None
            state.pending_confirmation = None
        else:
            state.status = "recovering"
            state.recovery_count += 1
            state.failure_reason = summary
        self.events.append(
            episode_id=episode_id,
            task_id=state.task_id,
            kind="execution_feedback",
            summary=summary,
            skill_id=skill_id,
            metadata={
                "success": bool(success),
                "status": state.status,
                **_trace_metadata_from_attempt(attempt),
            },
        )
        self._upsert_skill_trace(
            state,
            skill_id=skill_id,
            status=attempt.status if attempt is not None else state.status,
            objective=attempt.objective if attempt is not None else None,
            metadata=attempt.metadata if attempt is not None else {},
            success=success,
            summary=summary,
        )
        return self.save(state)

    def mark_task_reported(
        self, episode_id: str, *, success: bool, summary: str
    ) -> TaskRun | None:
        state = self.load_active(episode_id)
        if state is None:
            _logger.warning(f"mark_task_reported 未找到活跃任务：episode={episode_id}")
            return None
        _logger.info(
            f"标记任务完成：task={state.root_task!r} success={success} "
            f"old_status={state.status} episode={episode_id}"
        )
        state.task_success = bool(success)
        state.status = "completed" if success else "failed"
        state.failure_reason = None if success else summary
        if success:
            state.recovery = None
            state.paused_reason = None
            state.pending_confirmation = None
        state.finished_at = time.time()
        self.events.append(
            episode_id=episode_id,
            task_id=state.task_id,
            kind="task_reported",
            summary=summary,
            metadata={"success": bool(success), "status": state.status},
        )
        return self.save(state)

    def mark_skill_completed(
        self,
        episode_id: str,
        *,
        skill_id: str,
        summary: str,
        success: bool = True,
    ) -> TaskRun | None:
        state = self.load_active(episode_id)
        if state is None:
            return None
        attempt = self._attempt_for_skill(state, skill_id)
        if attempt is not None:
            attempt.status = "completed" if success else "failed"
            attempt.success = bool(success)
            attempt.metadata["completion_summary"] = summary
            attempt.updated_at = time.time()
        state.last_step_success = bool(success)
        if success:
            state.status = "active"
            state.failure_reason = None
            state.pending_confirmation = None
        else:
            state.status = "recovering"
            state.failure_reason = summary
        self.events.append(
            episode_id=episode_id,
            task_id=state.task_id,
            kind="skill_completed",
            summary=summary,
            skill_id=skill_id,
            metadata={
                "success": bool(success),
                "status": state.status,
                **_trace_metadata_from_attempt(attempt),
            },
        )
        self._upsert_skill_trace(
            state,
            skill_id=skill_id,
            status=attempt.status if attempt is not None else state.status,
            objective=attempt.objective if attempt is not None else None,
            metadata=attempt.metadata if attempt is not None else {},
            success=success,
            summary=summary,
        )
        return self.save(state)

    def bind_skill_trace_metadata(
        self,
        episode_id: str,
        *,
        skill_id: str,
        status: str,
        success: bool | None,
        summary: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskRun | None:
        state = self.load_active(episode_id)
        if state is None:
            tasks = self.list_for_episode(episode_id)
            state = tasks[0] if tasks else None
        if state is None:
            return None
        attempt = self._attempt_for_skill(state, skill_id)
        if attempt is not None:
            attempt.skill = _trace_value(metadata, "skill") or attempt.skill
            attempt.backend = _trace_value(metadata, "backend") or attempt.backend
            attempt.implementation_name = (
                _trace_value(metadata, "implementation_name")
                or attempt.implementation_name
            )
            attempt.implementation_kind = (
                _trace_value(metadata, "implementation_kind")
                or attempt.implementation_kind
            )
            attempt.updated_at = time.time()
            if summary:
                attempt.metadata["result_summary"] = summary
        self._upsert_skill_trace(
            state,
            skill_id=skill_id,
            status=status,
            objective=attempt.objective if attempt is not None else None,
            metadata={
                **dict(attempt.metadata if attempt is not None else {}),
                **dict(metadata or {}),
            },
            success=success,
            summary=summary,
        )
        return self.save(state)

    def append_attempt(
        self,
        episode_id: str,
        *,
        event: str,
        summary: str,
        skill_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskRun | None:
        state = self.load_active(episode_id)
        if state is None:
            return None
        self.events.append(
            episode_id=episode_id,
            task_id=state.task_id,
            kind=event,
            summary=summary,
            skill_id=skill_id,
            metadata=metadata,
        )
        return self.save(state)

    def record_scene_memory(
        self,
        episode_id: str,
        *,
        summary: str,
        frame_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskRun | None:
        state = self.load_active(episode_id)
        if state is None:
            return None
        self.events.append(
            episode_id=episode_id,
            task_id=state.task_id,
            kind="scene_observed",
            summary=summary,
            frame_id=frame_id,
            metadata=metadata,
        )
        state.metadata["latest_scene_event"] = {
            "summary": summary,
            "frame_id": frame_id,
            "metadata": dict(metadata or {}),
            "timestamp": time.time(),
        }
        return self.save(state)

    def update_watchdog(
        self,
        episode_id: str,
        *,
        health: str,
        summary: str,
        metadata: dict[str, Any] | None = None,
    ) -> TaskRun | None:
        state = self.load_active(episode_id)
        if state is None:
            return None
        previous = state.watchdog if isinstance(state.watchdog, dict) else {}
        event_changed = (
            previous.get("health") != health or previous.get("summary") != summary
        )
        state.watchdog = {
            "health": health,
            "summary": summary,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        if health in {"blocked", "stale", "unsafe", "lost"} and state.status not in {
            "paused",
            "recovering",
        }:
            state.status = "recovering"
            state.recovery_count += 1
            state.failure_reason = summary
        if event_changed:
            self.events.append(
                episode_id=episode_id,
                task_id=state.task_id,
                kind="watchdog",
                summary=summary,
                metadata={"health": health, **dict(metadata or {})},
            )
        return self.save(state)

    def set_recovery(
        self,
        episode_id: str,
        *,
        strategy: str,
        summary: str,
        metadata: dict[str, Any] | None = None,
    ) -> TaskRun | None:
        state = self.load_active(episode_id)
        if state is None:
            return None
        previous = state.recovery if isinstance(state.recovery, dict) else {}
        event_changed = (
            previous.get("strategy") != strategy or previous.get("summary") != summary
        )
        state.recovery = {
            "strategy": strategy,
            "summary": summary,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        if state.status not in _TERMINAL_TASK_STATUSES:
            state.status = "recovering"
        state.pending_confirmation = None
        if event_changed:
            self.events.append(
                episode_id=episode_id,
                task_id=state.task_id,
                kind="recovery_selected",
                summary=summary,
                metadata={"strategy": strategy, **dict(metadata or {})},
            )
        return self.save(state)

    def pause(
        self, episode_id: str, *, reason: str, operator: str | None = None
    ) -> TaskRun | None:
        state = self.load_active(episode_id)
        if state is None:
            return None
        state.status = "paused"
        state.paused_reason = reason
        state.pending_confirmation = None
        state.operator_notes.append(
            {
                "action": "pause",
                "reason": reason,
                "operator": operator,
                "timestamp": time.time(),
            }
        )
        return self.save(state)

    def resume(self, episode_id: str, *, operator: str | None = None) -> TaskRun | None:
        tasks = self.list_for_episode(episode_id)
        state = next(
            (task for task in tasks if task.status == "paused"), None
        ) or self.load_active(episode_id)
        if state is None:
            return None
        state.status = "active"
        state.paused_reason = None
        state.operator_notes.append(
            {"action": "resume", "operator": operator, "timestamp": time.time()}
        )
        return self.save(state)

    def abort(
        self, episode_id: str, *, reason: str, operator: str | None = None
    ) -> TaskRun | None:
        state = self.load_active(episode_id)
        if state is None:
            return None
        state.status = "cancelled"
        state.failure_reason = reason
        state.pending_confirmation = None
        state.finished_at = time.time()
        state.operator_notes.append(
            {
                "action": "abort",
                "reason": reason,
                "operator": operator,
                "timestamp": time.time(),
            }
        )
        return self.save(state)

    def mark_operator_feedback(
        self,
        episode_id: str,
        *,
        success: bool,
        summary: str,
        operator: str | None = None,
    ) -> TaskRun | None:
        state = self.load_active(episode_id)
        if state is None:
            return None
        state.last_step_success = bool(success)
        if success:
            state.status = "active"
            state.failure_reason = None
            state.pending_confirmation = None
        else:
            state.status = "recovering"
            state.failure_reason = summary
        state.operator_notes.append(
            {
                "action": "confirm_feedback",
                "success": success,
                "summary": summary,
                "operator": operator,
                "timestamp": time.time(),
            }
        )
        return self.save(state)

    def set_pending_confirmation(
        self,
        episode_id: str,
        *,
        capability: str,
        objective: str,
        prompt: str,
        slots: dict[str, Any] | None = None,
        interrupt: bool = False,
        proposal_id: str | None = None,
        agent_id: str | None = None,
        robot_id: str | None = None,
    ) -> TaskRun | None:
        state = self.load_active(episode_id)
        if state is None:
            state = self._create_task(
                episode_id=episode_id,
                task=objective,
                agent_id=agent_id,
                robot_id=robot_id,
            )
        state.status = "awaiting_confirmation"
        state.pending_confirmation = {
            "proposal_id": proposal_id or f"proposal_{uuid.uuid4().hex[:16]}",
            "capability": capability,
            "objective": objective,
            "prompt": prompt,
            "slots": dict(slots or {}),
            "interrupt": bool(interrupt),
            "timestamp": time.time(),
        }
        self.events.append(
            episode_id=episode_id,
            task_id=state.task_id,
            kind="awaiting_confirmation",
            summary=prompt,
            metadata={
                "capability": capability,
                "objective": objective,
                "interrupt": bool(interrupt),
            },
        )
        return self.save(state)

    def clear_pending_confirmation(self, episode_id: str) -> TaskRun | None:
        state = self.load_active(episode_id)
        if state is None:
            return None
        state.pending_confirmation = None
        if state.status == "awaiting_confirmation":
            state.status = "active"
        return self.save(state)

    def save(self, state: TaskRun) -> TaskRun:
        state.updated_at = self._next_timestamp()
        path = self._path(state.task_id)
        tmp = self.root / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(
                state.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True
            )
        for attempt in range(5):
            try:
                tmp.replace(path)
                break
            except PermissionError:
                if attempt >= 4:
                    raise
                time.sleep(0.01 * (attempt + 1))
        return state

    def list_for_episode(self, episode_id: str) -> list[TaskRun]:
        tasks: list[TaskRun] = []
        for path in self.root.glob("*.task.json"):
            try:
                with path.open(encoding="utf-8") as handle:
                    state = TaskRun.from_dict(json.load(handle))
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                continue
            if state.episode_id == episode_id:
                tasks.append(state)
        return sorted(tasks, key=lambda item: item.updated_at, reverse=True)

    def list_recent(self, limit: int = 100) -> list[TaskRun]:
        tasks: list[TaskRun] = []
        for path in self.root.glob("*.task.json"):
            try:
                with path.open(encoding="utf-8") as handle:
                    tasks.append(TaskRun.from_dict(json.load(handle)))
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                continue
        return sorted(tasks, key=lambda item: item.updated_at, reverse=True)[
            : max(1, limit)
        ]

    def _path(self, task_id: str) -> Path:
        return self.root / f"{_sanitize(task_id)}.task.json"

    def _next_timestamp(self) -> float:
        now = time.time()
        if now <= self._last_timestamp:
            now = self._last_timestamp + 1e-6
        self._last_timestamp = now
        return now

    @staticmethod
    def _attempt_for_skill(state: TaskRun, skill_id: str) -> TaskAttempt | None:
        for attempt in reversed(state.attempts):
            if attempt.skill_id == skill_id:
                return attempt
        return None

    @staticmethod
    def _upsert_skill_trace(
        state: TaskRun,
        *,
        skill_id: str,
        status: str,
        objective: str | None,
        metadata: dict[str, Any] | None,
        success: bool | None = None,
        summary: str | None = None,
    ) -> None:
        trace_item = {
            "skill_id": skill_id,
            "skill": _trace_value(metadata, "skill"),
            "backend": _trace_value(metadata, "backend"),
            "implementation_name": _trace_value(metadata, "implementation_name"),
            "implementation_kind": _trace_value(metadata, "implementation_kind"),
            "objective": objective or "",
            "status": status,
            "success": success,
            "summary": summary,
            "updated_at": time.time(),
        }
        for index, item in enumerate(state.skill_trace):
            if str(item.get("skill_id") or "") == skill_id:
                state.skill_trace[index] = {
                    **item,
                    **{
                        key: value
                        for key, value in trace_item.items()
                        if value is not None
                    },
                }
                return
        state.skill_trace.append(trace_item)


def _trace_value(metadata: dict[str, Any] | None, *keys: str) -> str | None:
    if not isinstance(metadata, dict):
        return None
    for key in keys:
        value = metadata.get(key)
        text = str(value or "").strip()
        if text:
            return text
    return None


def _trace_metadata_from_attempt(attempt: TaskAttempt | None) -> dict[str, Any]:
    if attempt is None:
        return {}
    return {
        "skill": attempt.skill,
        "backend": attempt.backend,
        "implementation_name": attempt.implementation_name,
        "implementation_kind": attempt.implementation_kind,
    }


def _sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value)
