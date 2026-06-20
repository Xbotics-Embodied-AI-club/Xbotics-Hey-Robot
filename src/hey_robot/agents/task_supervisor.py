from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hey_robot.agents.checkpoint import RobotAgentCheckpointStore
from hey_robot.agents.task_run import TaskRun, TaskRunStore
from hey_robot.bus.factory import create_bus_client
from hey_robot.config import DeploymentConfig
from hey_robot.episode import JsonlEpisodeStore, RobotEpisodeStateStore
from hey_robot.events import RuntimeEvent
from hey_robot.gateway.identity import IdentityResolver
from hey_robot.notifications import (
    Notification,
    NotificationPolicy,
    NotificationService,
)
from hey_robot.protocol import AgentReply, SkillEvent, Topics
from hey_robot.protocol.messages import from_payload, to_payload
from hey_robot.skills import SkillPhase, SkillStore


@dataclass(frozen=True)
class RobotWatchdogSnapshot:
    health: str
    summary: str
    episode_id: str
    task_id: str
    robot_id: str | None = None
    active_skill_id: str | None = None
    active_skill_age_sec: float | None = None
    last_status_age_sec: float | None = None
    last_observation_age_sec: float | None = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TaskSupervisorService:
    """Long-running task supervisor for the deployment-native robot flow."""

    def __init__(
        self, config: DeploymentConfig, *, episode_dir: str | Path | None = None
    ) -> None:
        self.config = config
        settings = _supervisor_settings(config)
        self.enabled = bool(settings.get("enabled", True))
        self.interval_sec = float(settings.get("interval_sec", 2.0))
        self.skill_timeout_sec = float(settings.get("skill_timeout_sec", 180.0))
        self.observation_stale_sec = float(settings.get("observation_stale_sec", 30.0))
        self.status_stale_sec = float(settings.get("status_stale_sec", 30.0))
        self.root = episode_dir or config.resources.episodes_root
        self.task_runs = TaskRunStore(self.root)
        self.checkpoints = RobotAgentCheckpointStore(self.root)
        self.episodes = JsonlEpisodeStore(self.root)
        self.robot_states = RobotEpisodeStateStore(self.root)
        self.identity = IdentityResolver(
            config.identity,
            state_path=Path(config.resources.runtime_dir)
            / "identity"
            / "bindings.json",
        )
        self.notifications = NotificationService(
            self.episodes,
            self._publish_notification_reply,
            policy=NotificationPolicy(config.notifications),
            linked_target_provider=self.identity.linked_channel_targets,
        )
        self.skill_store = SkillStore(
            Path(config.resources.runtime_dir) / "skills",
            max_items=config.resources.events_max_items,
        )
        self.topics = Topics()
        self.bus = create_bus_client(config.deployment.bus)
        self._running = False

    async def start(self) -> None:
        if not self.enabled:
            await asyncio.Event().wait()
        await self.bus.connect()
        await self.bus.subscribe([self.topics.skill_event], self._on_skill_event)
        self._running = True
        while self._running:
            await self.tick()
            await asyncio.sleep(self.interval_sec)

    async def stop(self) -> None:
        self._running = False
        await self.bus.close()

    async def _on_skill_event(self, _topic: str, payload: dict) -> None:
        event = from_payload(SkillEvent, payload)
        if event.envelope.episode_id:
            self.task_runs.append_attempt(
                event.envelope.episode_id,
                event=f"skill.{event.phase}",
                summary=event.summary or event.error or event.text or event.phase,
                skill_id=event.skill_id,
                metadata={"phase": event.phase, "frame_id": event.frame_id},
            )

    async def tick(self) -> list[RobotWatchdogSnapshot]:
        snapshots: list[RobotWatchdogSnapshot] = []
        for task in self.task_runs.list_recent(limit=500):
            if task.status in {"completed", "failed", "cancelled"}:
                continue
            snapshot = self._watchdog_for_task(task)
            snapshots.append(snapshot)
            self.task_runs.update_watchdog(
                task.episode_id,
                health=snapshot.health,
                summary=snapshot.summary,
                metadata=snapshot.to_dict(),
            )
            should_notify = self._should_notify_watchdog(task, snapshot)
            if snapshot.health != "healthy":
                self._apply_recovery(task, snapshot)
            await self._publish_watchdog(snapshot)
            if should_notify:
                await self._publish_watchdog_notification(task, snapshot)
        return snapshots

    def _watchdog_for_task(self, task: TaskRun) -> RobotWatchdogSnapshot:
        now = time.time()
        state = self.robot_states.load(task.episode_id)
        active_skill_id = _active_skill_id_for_watchdog(
            state_active_skill_id=state.active_skill_id if state else None,
            state_active_skill_phase=state.active_skill_phase if state else None,
            task_skill_ids=task.skill_ids,
        )
        skill = self.skill_store.get(active_skill_id) if active_skill_id else None
        monitor_robot_staleness = active_skill_id is not None and (
            skill is None or not skill.terminal
        )
        skill_age = now - skill.updated_at if skill is not None else None
        status_ts = None
        if state is not None:
            status_ts = ((state.last_status or {}).get("timestamp")) or state.updated_at
        status_age = now - float(status_ts) if status_ts else None
        obs_age = (
            now - float(state.updated_at)
            if state and state.last_observation_frame is not None
            else None
        )
        if task.status == "paused":
            health = "paused"
            summary = task.paused_reason or "task paused by operator"
        elif state and state.recovery_required:
            health = "blocked"
            summary = state.recovery_reason or "robot episode requires recovery"
        elif state is not None and _camera_quality_blocked(state):
            health = "blocked"
            summary = _camera_quality_blocked_summary(state)
        elif (
            skill is not None
            and not skill.terminal
            and skill_age is not None
            and skill_age > self.skill_timeout_sec
        ):
            health = "blocked"
            summary = f"active skill timed out after {skill_age:.1f}s"
        elif (
            monitor_robot_staleness
            and status_age is not None
            and status_age > self.status_stale_sec
        ):
            health = "stale"
            summary = f"robot status stale for {status_age:.1f}s"
        elif (
            monitor_robot_staleness
            and obs_age is not None
            and obs_age > self.observation_stale_sec
        ):
            health = "stale"
            summary = f"robot observation stale for {obs_age:.1f}s"
        else:
            health = "healthy"
            summary = "task supervisor checks passed"
        return RobotWatchdogSnapshot(
            health=health,
            summary=summary,
            episode_id=task.episode_id,
            task_id=task.task_id,
            robot_id=task.robot_id or (state.robot_id if state else None),
            active_skill_id=active_skill_id,
            active_skill_age_sec=skill_age,
            last_status_age_sec=status_age,
            last_observation_age_sec=obs_age,
        )

    def _apply_recovery(self, task: TaskRun, snapshot: RobotWatchdogSnapshot) -> None:
        if snapshot.health == "paused":
            return
        strategy = "ask_user"
        if "timed out" in snapshot.summary:
            strategy = "interrupt_then_continue"
        elif snapshot.health == "stale":
            strategy = "pause_for_operator"
        elif snapshot.health == "blocked":
            strategy = "continue_from_observation"
        self.task_runs.set_recovery(
            task.episode_id,
            strategy=strategy,
            summary=snapshot.summary,
            metadata=snapshot.to_dict(),
        )

    async def _publish_watchdog(self, snapshot: RobotWatchdogSnapshot) -> None:
        try:
            await self.bus.publish(
                self.topics.runtime_event,
                RuntimeEvent.make(
                    "task.watchdog",
                    source="task_supervisor",
                    episode_id=snapshot.episode_id,
                    robot_id=snapshot.robot_id,
                    payload=snapshot.to_dict(),
                ).to_dict(),
            )
        except RuntimeError:
            return

    @staticmethod
    def _should_notify_watchdog(task: TaskRun, snapshot: RobotWatchdogSnapshot) -> bool:
        if snapshot.health in {"healthy", "paused"}:
            return False
        previous = task.watchdog if isinstance(task.watchdog, dict) else {}
        return previous.get("health") != snapshot.health

    async def _publish_watchdog_notification(
        self, task: TaskRun, snapshot: RobotWatchdogSnapshot
    ) -> None:
        await self.notifications.publish(
            Notification(
                kind="task_watchdog",
                severity="warning" if snapshot.health == "stale" else "critical",
                body=f"任务监督告警：{snapshot.summary}",
                episode_id=task.episode_id,
                robot_id=task.robot_id,
                agent_id=task.agent_id,
                metadata={
                    "event": "task_watchdog",
                    "watchdog_health": snapshot.health,
                    "task_id": task.task_id,
                    "dedupe_key": f"task_watchdog:{task.episode_id}:{snapshot.health}:{snapshot.summary}",
                },
            )
        )

    async def _publish_notification_reply(self, reply: AgentReply) -> None:
        await self.bus.publish(self.topics.agent_reply, to_payload(reply))


def _active_skill_id_for_watchdog(
    *,
    state_active_skill_id: str | None,
    state_active_skill_phase: str | None,
    task_skill_ids: list[str],
) -> str | None:
    if state_active_skill_id and state_active_skill_phase not in _TERMINAL_SKILL_PHASES:
        return state_active_skill_id
    if state_active_skill_id:
        return None
    return task_skill_ids[-1] if task_skill_ids else None


_TERMINAL_SKILL_PHASES = {
    SkillPhase.COMPLETED.value,
    SkillPhase.FAILED.value,
    SkillPhase.INTERRUPTED.value,
    SkillPhase.CONFIRMED.value,
    SkillPhase.FEEDBACK_FAILED.value,
}


def _supervisor_settings(config: DeploymentConfig) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for spec in config.agents.values():
        raw = spec.settings.get("task_supervisor")
        if isinstance(raw, dict):
            values.update(raw)
    return values


_WATCHDOG_CAMERA_MAX_AGE_MS = 20_000


def _camera_quality_blocked(state: object) -> bool:
    last_status = getattr(state, "last_status", None) or {}
    if not isinstance(last_status, dict):
        return False
    metrics = (
        last_status.get("metrics")
        if isinstance(last_status.get("metrics"), dict)
        else {}
    )
    camera = metrics.get("camera") if isinstance(metrics, dict) else None
    if not isinstance(camera, dict):
        return False
    ok = camera.get("ok")
    frame_available = camera.get("frame_available")
    valid_count = camera.get("valid_image_count")
    quality_issues = camera.get("image_quality_issues") or []
    age_ms = camera.get("age_ms")
    if ok is False:
        return True
    if frame_available is False:
        return True
    if isinstance(valid_count, (int, float)) and valid_count <= 0:
        return True
    if quality_issues and all(str(i).strip() for i in quality_issues):
        return True
    return bool(
        isinstance(age_ms, (int, float)) and age_ms > _WATCHDOG_CAMERA_MAX_AGE_MS
    )


def _camera_quality_blocked_summary(state: object) -> str:
    last_status = getattr(state, "last_status", None) or {}
    if not isinstance(last_status, dict):
        return "camera quality degraded"
    metrics = (
        last_status.get("metrics")
        if isinstance(last_status.get("metrics"), dict)
        else {}
    )
    camera = metrics.get("camera") if isinstance(metrics, dict) else {}
    if not isinstance(camera, dict):
        return "camera quality degraded"
    ok = camera.get("ok")
    frame_available = camera.get("frame_available")
    valid_count = camera.get("valid_image_count")
    quality_issues = camera.get("image_quality_issues") or []
    age_ms = camera.get("age_ms")
    parts = [
        f"camera blocked: ok={ok}, frame_available={frame_available}, valid_image_count={valid_count}"
    ]
    if quality_issues:
        parts.append(f"issues={quality_issues}")
    if isinstance(age_ms, (int, float)):
        parts.append(f"age_ms={age_ms:.0f}")
    return ", ".join(parts)
