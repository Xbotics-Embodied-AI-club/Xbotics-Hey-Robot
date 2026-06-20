from __future__ import annotations

import contextlib
import json
import os
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hey_robot.protocol import RobotObservation, RobotStatus, SkillEvent, SkillResult
from hey_robot.skills import SkillPhase

_LOCK_TIMEOUT_SEC = 10.0
_STALE_LOCK_SEC = 30.0

_TERMINAL_PHASES = {
    SkillPhase.COMPLETED.value,
    SkillPhase.FAILED.value,
    SkillPhase.INTERRUPTED.value,
    SkillPhase.CONFIRMED.value,
    SkillPhase.FEEDBACK_FAILED.value,
}


@dataclass
class ExecutionFeedbackSnapshot:
    skill_id: str
    subgoal_success: bool
    task_success: bool
    summary: str
    next_hint: str | None = None
    updated_at: float = field(default_factory=time.time)


@dataclass
class RobotEpisodeState:
    episode_id: str
    agent_id: str | None = None
    robot_id: str | None = None
    policy_id: str | None = None
    active_task: str | None = None
    active_skill_id: str | None = None
    active_skill_text: str | None = None
    active_skill_phase: str | None = None
    last_observation_frame: int | None = None
    last_observation_images: list[dict[str, Any]] = field(default_factory=list)
    last_status: dict[str, Any] = field(default_factory=dict)
    last_execution_feedback: dict[str, Any] | None = None
    recovery_required: bool = False
    recovery_reason: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def recovery_context(self) -> str | None:
        if not self.recovery_required:
            return None
        strategy = _recovery_strategy_hint(
            self.recovery_reason, self.last_execution_feedback
        )
        parts = [
            "Recovery context:",
            f"- episode_id: {self.episode_id}",
            f"- robot_id: {self.robot_id or 'unknown'}",
            f"- active_task: {self.active_task or 'unknown'}",
            f"- active_skill_id: {self.active_skill_id or 'none'}",
            f"- active_skill_phase: {self.active_skill_phase or 'unknown'}",
            f"- recovery_reason: {self.recovery_reason or 'unknown'}",
            f"- recovery_strategy_hint: {strategy}",
            f"- recommended_tools: {', '.join(_recommended_recovery_tools(strategy))}",
            f"- continuation_guidance: {_recovery_continuation_guidance(strategy)}",
        ]
        if self.last_observation_frame is not None:
            parts.append(f"- last_observation_frame: {self.last_observation_frame}")
        if self.last_execution_feedback:
            parts.append(
                f"- last_execution_feedback: {json.dumps(self.last_execution_feedback, ensure_ascii=False)}"
            )
        parts.append(
            "Recovery rule: inspect current status/observation and resolve the last execution feedback before "
            "submitting another robot skill."
        )
        return "\n".join(parts)


class RobotEpisodeStateStore:
    """Materialized robot runtime state stored beside conversation episodes."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def ensure(
        self,
        episode_id: str,
        *,
        agent_id: str | None = None,
        robot_id: str | None = None,
        policy_id: str | None = None,
    ) -> RobotEpisodeState:
        with self._episode_lock(episode_id):
            state = self._load_unlocked(episode_id) or RobotEpisodeState(
                episode_id=episode_id
            )
            changed = False
            for key, value in {
                "agent_id": agent_id,
                "robot_id": robot_id,
                "policy_id": policy_id,
            }.items():
                if value and getattr(state, key) != value:
                    setattr(state, key, value)
                    changed = True
            if changed or not self._path(episode_id).exists():
                self._save_unlocked(state)
            return state

    def load(self, episode_id: str) -> RobotEpisodeState | None:
        with self._episode_lock(episode_id):
            return self._load_unlocked(episode_id)

    def _load_unlocked(self, episode_id: str) -> RobotEpisodeState | None:
        path = self._path(episode_id)
        if not path.exists():
            return None
        try:
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            return RobotEpisodeState(**data)
        except (OSError, TypeError, json.JSONDecodeError):
            return None

    def save(self, state: RobotEpisodeState) -> None:
        with self._episode_lock(state.episode_id):
            self._save_unlocked(state)

    def _save_unlocked(self, state: RobotEpisodeState) -> None:
        state.updated_at = time.time()
        path = self._path(state.episode_id)
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = (
            self.root
            / f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
        )
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(
                state.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True
            )
        for attempt in range(5):
            try:
                tmp.replace(path)
                return
            except PermissionError:
                if attempt >= 4:
                    raise
                time.sleep(0.02 * (attempt + 1))

    def list_states(self) -> list[RobotEpisodeState]:
        states: list[RobotEpisodeState] = []
        for path in self.root.glob("*.robot_state.json"):
            try:
                with path.open(encoding="utf-8") as handle:
                    data = json.load(handle)
                states.append(RobotEpisodeState(**data))
            except (OSError, TypeError, json.JSONDecodeError):
                continue
        return sorted(states, key=lambda item: item.updated_at, reverse=True)

    def mark_recovery_required_for_nonterminal(self) -> list[RobotEpisodeState]:
        marked: list[RobotEpisodeState] = []
        for state in self.list_states():
            phase = state.active_skill_phase
            if (
                state.active_skill_id
                and phase
                and phase not in _TERMINAL_PHASES
                and not state.recovery_required
            ):
                state.recovery_required = True
                state.recovery_reason = (
                    f"service restarted with non-terminal skill phase {phase}"
                )
                self.save(state)
                marked.append(state)
        return marked

    def apply_skill_event(self, event: SkillEvent) -> RobotEpisodeState | None:
        episode_id = event.envelope.episode_id
        if not episode_id:
            return None
        with self._episode_lock(episode_id):
            state = self._load_unlocked(episode_id) or RobotEpisodeState(
                episode_id=episode_id
            )
            if event.envelope.agent_id:
                state.agent_id = event.envelope.agent_id
            if event.envelope.robot_id:
                state.robot_id = event.envelope.robot_id
            state.active_skill_id = event.skill_id
            state.active_skill_phase = event.phase
            state.active_skill_text = event.text or state.active_skill_text
            state.policy_id = event.policy_id or state.policy_id
            if event.frame_id is not None:
                state.last_observation_frame = event.frame_id
            if event.phase in {
                SkillPhase.ISSUED.value,
                SkillPhase.ACCEPTED.value,
                SkillPhase.EXECUTING.value,
            }:
                state.recovery_required = False
                state.recovery_reason = None
            elif event.phase == SkillPhase.FEEDBACK_PENDING.value:
                state.recovery_required = True
                state.recovery_reason = (
                    "skill completed and requires execution feedback"
                )
            elif event.phase == SkillPhase.CONFIRMED.value:
                state.recovery_required = False
                state.recovery_reason = None
            elif event.phase in {
                SkillPhase.FAILED.value,
                SkillPhase.INTERRUPTED.value,
                SkillPhase.FEEDBACK_FAILED.value,
            }:
                state.recovery_required = True
                state.recovery_reason = (
                    event.error or event.summary or f"skill phase {event.phase}"
                )
            self._save_unlocked(state)
            return state

    def apply_skill_result(self, result: SkillResult) -> RobotEpisodeState | None:
        episode_id = result.envelope.episode_id
        if not episode_id:
            return None
        with self._episode_lock(episode_id):
            state = self._load_unlocked(episode_id) or RobotEpisodeState(
                episode_id=episode_id
            )
            if result.envelope.agent_id:
                state.agent_id = result.envelope.agent_id
            if result.envelope.robot_id:
                state.robot_id = result.envelope.robot_id
            state.active_skill_id = result.skill_id
            state.active_skill_phase = _phase_from_result_status(result.status)
            if result.frame_id is not None:
                state.last_observation_frame = result.frame_id
            if result.status == SkillPhase.COMPLETED.value:
                state.recovery_required = False
                state.recovery_reason = None
            elif result.status in {
                SkillPhase.FAILED.value,
                SkillPhase.INTERRUPTED.value,
                SkillPhase.FEEDBACK_FAILED.value,
            }:
                state.recovery_required = True
                state.recovery_reason = (
                    result.error or result.summary or f"skill result {result.status}"
                )
            self._save_unlocked(state)
            return state

    def mark_task_started(
        self,
        episode_id: str,
        *,
        task: str,
        agent_id: str | None,
        robot_id: str | None,
    ) -> RobotEpisodeState:
        with self._episode_lock(episode_id):
            state = self._load_unlocked(episode_id) or RobotEpisodeState(
                episode_id=episode_id
            )
            if agent_id:
                state.agent_id = agent_id
            if robot_id:
                state.robot_id = robot_id
            state.active_task = task
            state.active_skill_id = None
            state.active_skill_text = None
            state.active_skill_phase = None
            state.recovery_required = False
            state.recovery_reason = None
            self._save_unlocked(state)
            return state

    def clear_recovery(
        self,
        episode_id: str,
        *,
        task: str | None = None,
    ) -> RobotEpisodeState | None:
        with self._episode_lock(episode_id):
            state = self._load_unlocked(episode_id)
            if state is None:
                return None
            if task:
                state.active_task = task
            state.recovery_required = False
            state.recovery_reason = None
            self._save_unlocked(state)
            return state

    def update_observation(
        self, episode_id: str, observation: RobotObservation
    ) -> RobotEpisodeState | None:
        with self._episode_lock(episode_id):
            state = self._load_unlocked(episode_id)
            if state is None:
                return None
            state.last_observation_frame = observation.frame_id
            state.last_observation_images = [
                {
                    "uri": image.uri,
                    "camera": image.camera,
                    "width": image.width,
                    "height": image.height,
                    "timestamp": image.timestamp,
                }
                for image in observation.images
            ]
            self._save_unlocked(state)
            return state

    def update_status(
        self, episode_id: str, status: RobotStatus
    ) -> RobotEpisodeState | None:
        with self._episode_lock(episode_id):
            state = self._load_unlocked(episode_id)
            if state is None:
                return None
            state.last_status = _status_snapshot(status)
            self._save_unlocked(state)
            return state

    def update_status_for_robot(
        self, robot_id: str, status: RobotStatus
    ) -> list[RobotEpisodeState]:
        updated: list[RobotEpisodeState] = []
        for state in self.list_states():
            if state.robot_id != robot_id:
                continue
            with self._episode_lock(state.episode_id):
                current = self._load_unlocked(state.episode_id)
                if current is None or current.robot_id != robot_id:
                    continue
                current.last_status = _status_snapshot(status)
                self._save_unlocked(current)
                updated.append(current)
        return updated

    def mark_execution_feedback(
        self,
        episode_id: str,
        *,
        skill_id: str,
        subgoal_success: bool,
        task_success: bool,
        summary: str,
        next_hint: str | None = None,
    ) -> RobotEpisodeState | None:
        with self._episode_lock(episode_id):
            state = self._load_unlocked(episode_id)
            if state is None:
                return None
            state.last_execution_feedback = ExecutionFeedbackSnapshot(
                skill_id=skill_id,
                subgoal_success=subgoal_success,
                task_success=task_success,
                summary=summary,
                next_hint=next_hint,
            ).__dict__
            state.recovery_required = not bool(subgoal_success)
            state.recovery_reason = (
                None if subgoal_success else "last execution feedback failed"
            )
            if subgoal_success:
                state.active_skill_phase = SkillPhase.CONFIRMED.value
            self._save_unlocked(state)
            return state

    def _path(self, episode_id: str) -> Path:
        return self.root / f"{_sanitize(episode_id)}.robot_state.json"

    @contextlib.contextmanager
    def _episode_lock(self, episode_id: str) -> Iterator[None]:
        lock_path = self.root / f"{_sanitize(episode_id)}.robot_state.lock"
        self.root.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + _LOCK_TIMEOUT_SEC
        fd: int | None = None
        while fd is None:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()} {time.time()}\n".encode("ascii"))
            except (FileExistsError, PermissionError):
                if _is_stale_lock(lock_path):
                    with contextlib.suppress(OSError):
                        lock_path.unlink()
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out waiting for robot episode state lock: {lock_path}"
                    ) from None
                time.sleep(0.01)
        try:
            yield
        finally:
            if fd is not None:
                with contextlib.suppress(OSError):
                    os.close(fd)
            with contextlib.suppress(OSError):
                lock_path.unlink()


def _sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value)


def _phase_from_result_status(status: str) -> str:
    if status in {
        SkillPhase.COMPLETED.value,
        SkillPhase.FAILED.value,
        SkillPhase.INTERRUPTED.value,
        SkillPhase.FEEDBACK_FAILED.value,
    }:
        return status
    return str(status or "unknown")


def _recovery_strategy_hint(reason: str | None, feedback: dict[str, Any] | None) -> str:
    next_hint = str((feedback or {}).get("next_hint") or "").lower()
    lowered_reason = str(reason or "").lower()
    if any(
        token in next_hint
        for token in ("reposition", "change viewpoint", "move closer")
    ):
        return "reposition"
    if any(
        token in next_hint
        for token in ("inspect", "observe", "reobserve", "look again")
    ):
        return "reobserve"
    if any(
        token in next_hint
        for token in ("retry", "adjust", "try again", "closer", "further")
    ):
        return "retry_with_adjustment"
    if any(token in next_hint for token in ("degraded", "continue anyway", "proceed")):
        return "degraded_continue"
    if any(token in lowered_reason for token in ("busy", "conflict", "clarify")):
        return "clarify"
    if any(
        token in lowered_reason
        for token in ("not visible", "occluded", "camera", "image")
    ):
        return "reobserve"
    return "reobserve"


def _recommended_recovery_tools(strategy: str) -> list[str]:
    mapping = {
        "clarify": ["get_task_context", "wait"],
        "reobserve": ["get_task_context", "request_perception"],
        "reposition": ["get_task_context", "request_perception"],
        "retry_with_adjustment": ["get_task_context", "request_perception"],
        "degraded_continue": ["get_task_context", "get_robot_status"],
    }
    return mapping.get(strategy, ["get_task_context", "wait"])


def _recovery_continuation_guidance(strategy: str) -> str:
    mapping = {
        "clarify": "Ask one focused clarification before the next robot action.",
        "reobserve": "Inspect again before retrying or reporting completion.",
        "reposition": "Improve viewpoint or pose first, then inspect again before continuing.",
        "retry_with_adjustment": "Retry the same skill with adjusted parameters based on the failure reason and fresh observation.",
        "degraded_continue": "Continue the task with reduced capability; avoid relying on the degraded resource.",
    }
    return mapping.get(
        strategy, "Resolve recovery before issuing another robot action."
    )


def _status_snapshot(status: RobotStatus) -> dict[str, Any]:
    return {
        "frame_id": status.frame_id,
        "state": status.state,
        "task": status.task,
        "skill_id": status.skill_id,
        "success": status.success,
        "error": status.error,
        "metrics": status.metrics,
        "timestamp": status.envelope.timestamp,
    }


def _is_stale_lock(path: Path) -> bool:
    with contextlib.suppress(OSError):
        return time.time() - path.stat().st_mtime > _STALE_LOCK_SEC
    return False
