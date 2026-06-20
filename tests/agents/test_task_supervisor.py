from __future__ import annotations

import asyncio
import time
from typing import cast

from hey_robot.agents import TaskSupervisorService
from hey_robot.agents.task_run import TaskRunStore
from hey_robot.bus.client import BusClient
from hey_robot.config import DeploymentConfig
from hey_robot.episode import JsonlEpisodeStore, RobotEpisodeStateStore
from hey_robot.episode.scope import EpisodeScope
from hey_robot.protocol import Envelope, SkillEvent, UserTurn
from hey_robot.protocol.messages import to_payload
from hey_robot.skills import SkillStore
from hey_robot.tasks import build_task_report


def _config(tmp_path) -> DeploymentConfig:
    return DeploymentConfig.from_dict(
        {
            "resources": {
                "runtime_dir": str(tmp_path / "runtime"),
                "episodes": {"root": str(tmp_path / "episodes")},
                "media": {"root": str(tmp_path / "media")},
            },
            "agents": {
                "main": {
                    "type": "robot_agent",
                    "robot_id": "mock0",
                    "settings": {
                        "task_supervisor": {
                            "enabled": True,
                            "interval_sec": 0.01,
                            "skill_timeout_sec": 0.01,
                        }
                    },
                }
            },
            "robots": {"mock0": {"type": "mock"}},
        }
    )


def test_task_supervisor_marks_timed_out_skill_recovering(tmp_path) -> None:
    config = _config(tmp_path)
    task_store = TaskRunStore(config.resources.episodes_root)
    task_store.ensure_active(
        episode_id="s1", task="long task", agent_id="main", robot_id="mock0"
    )
    task_store.bind_skill("s1", "cmd1", "long task")
    episodes = JsonlEpisodeStore(config.resources.episodes_root)
    episodes.ensure("s1", scope=EpisodeScope(agent_id="main"), aliases=[])
    episodes.append_user_turn(
        "s1",
        UserTurn(
            envelope=Envelope(
                episode_id="s1",
                agent_id="main",
                robot_id="mock0",
                channel="feishu",
                chat_id="oc_chat_1",
                sender_id="ou_user_1",
            ),
            text="long task",
        ),
    )
    SkillStore(tmp_path / "runtime" / "skills").append(
        SkillEvent(
            envelope=Envelope(
                episode_id="s1",
                agent_id="main",
                robot_id="mock0",
                timestamp=time.time() - 10,
            ),
            skill_id="cmd1",
            phase="executing",
            text="long task",
        )
    )
    supervisor = TaskSupervisorService(config)
    fake_bus = FakeBus()
    supervisor.bus = fake_bus

    snapshots = asyncio.run(supervisor.tick())
    task = task_store.load_active("s1")
    replies = [
        payload
        for topic, payload in fake_bus.published
        if topic == supervisor.topics.agent_reply
    ]

    assert snapshots[0].health == "blocked"
    assert task is not None
    assert task.status == "recovering"
    assert task.recovery is not None
    assert task.recovery["strategy"] == "interrupt_then_continue"
    assert replies
    assert replies[-1]["metadata"]["event"] == "task_watchdog"
    assert replies[-1]["envelope"]["chat_id"] == "oc_chat_1"


def test_task_report_includes_task_skill_and_summary(tmp_path) -> None:
    config = _config(tmp_path)
    task_store = TaskRunStore(config.resources.episodes_root)
    checkpoint_store = supervisor_checkpoints = TaskSupervisorService(
        config
    ).checkpoints
    robot_states = RobotEpisodeStateStore(config.resources.episodes_root)
    skill_store = SkillStore(tmp_path / "runtime" / "skills")
    task_store.ensure_active(
        episode_id="s1", task="long task", agent_id="main", robot_id="mock0"
    )
    task_store.bind_skill("s1", "cmd1", "long task")
    checkpoint_store.mark_phase("s1", phase="skill_submitted", skill_id="cmd1")
    skill_store.append(
        SkillEvent(
            envelope=Envelope(episode_id="s1", agent_id="main", robot_id="mock0"),
            skill_id="cmd1",
            phase="issued",
        )
    )

    report = build_task_report(
        episode_id="s1",
        task_store=task_store,
        checkpoint_store=supervisor_checkpoints,
        robot_states=robot_states,
        skill_store=skill_store,
    )

    assert report["summary"]["skill_count"] == 1
    assert report["task"]["root_task"] == "long task"


class FakeBus(BusClient):
    def __init__(self) -> None:
        super().__init__(url="test://fake")
        self.published: list[tuple[str, dict]] = []
        self.subscriptions: list[tuple[str, object]] = []

    async def connect(self) -> None:
        self._connected = True

    async def close(self) -> None:
        self._connected = False

    async def publish(self, topic: str, payload: dict) -> None:
        self.published.append((topic, payload))

    async def subscribe(self, topics: list[str], handler: object) -> None:
        self.subscriptions.append((topics[0], handler))


def _make_supervisor(tmp_path, **overrides: float) -> TaskSupervisorService:
    config = _config(tmp_path)
    for key, value in overrides.items():
        setattr(config, key, value)
    svc = TaskSupervisorService(config)
    svc.bus = FakeBus()
    return svc


def _setup_episode_and_task(
    supervisor: TaskSupervisorService, episode_id: str, task_text: str
) -> None:
    supervisor.task_runs.ensure_active(
        episode_id=episode_id, task=task_text, agent_id="main", robot_id="mock0"
    )
    supervisor.episodes.ensure(
        episode_id, scope=EpisodeScope(agent_id="main"), aliases=[]
    )
    supervisor.episodes.append_user_turn(
        episode_id,
        UserTurn(
            envelope=Envelope(
                episode_id=episode_id,
                agent_id="main",
                robot_id="mock0",
                channel="feishu",
                chat_id="oc_chat_1",
                sender_id="ou_user_1",
            ),
            text=task_text,
        ),
    )


class TestWatchdogStates:
    def test_watchdog_paused_task(self, tmp_path) -> None:
        svc = _make_supervisor(tmp_path)
        _setup_episode_and_task(svc, "s1", "some task")
        svc.task_runs.pause("s1", reason="operator paused")

        snapshots = asyncio.run(svc.tick())
        assert snapshots[0].health == "paused"
        assert "operator paused" in snapshots[0].summary

    def test_watchdog_blocked_recovery_required(self, tmp_path) -> None:
        svc = _make_supervisor(tmp_path)
        _setup_episode_and_task(svc, "s2", "recovery task")
        robot_state = svc.robot_states.ensure("s2", agent_id="main", robot_id="mock0")
        robot_state.recovery_required = True
        robot_state.recovery_reason = "skill failed"
        svc.robot_states.save(robot_state)

        snapshots = asyncio.run(svc.tick())
        assert snapshots[0].health == "blocked"
        assert snapshots[0].summary == "skill failed"

    def test_watchdog_stale_status(self, tmp_path) -> None:
        svc = _make_supervisor(tmp_path)
        svc.status_stale_sec = 0.0
        _setup_episode_and_task(svc, "s3", "stale task")
        robot_state = svc.robot_states.ensure("s3", agent_id="main", robot_id="mock0")
        robot_state.active_skill_id = "cmd1"
        robot_state.last_status = {"battery": 0.8, "timestamp": time.time() - 100}
        svc.robot_states.save(robot_state)

        snapshots = asyncio.run(svc.tick())
        assert snapshots[0].health == "stale"
        assert "robot status stale" in snapshots[0].summary

    def test_watchdog_ignores_stale_status_without_active_skill(self, tmp_path) -> None:
        svc = _make_supervisor(tmp_path)
        svc.status_stale_sec = 0.0
        _setup_episode_and_task(svc, "s3-chat", "what is deepseek")
        robot_state = svc.robot_states.ensure(
            "s3-chat", agent_id="main", robot_id="mock0"
        )
        robot_state.last_status = {"battery": 0.8, "timestamp": time.time() - 100}
        svc.robot_states.save(robot_state)

        snapshots = asyncio.run(svc.tick())
        assert snapshots[0].health == "healthy"
        assert snapshots[0].summary == "task supervisor checks passed"

    def test_watchdog_stale_observation(self, tmp_path) -> None:
        svc = _make_supervisor(tmp_path)
        svc.status_stale_sec = 999  # prevent status staleness from firing first
        svc.observation_stale_sec = 0.0
        _setup_episode_and_task(svc, "s4", "obs task")
        robot_state = svc.robot_states.ensure("s4", agent_id="main", robot_id="mock0")
        robot_state.active_skill_id = "cmd1"
        robot_state.last_observation_frame = 42
        svc.robot_states.save(robot_state)
        # save() resets updated_at; rewrite with stale timestamp directly
        import json

        path = svc.robot_states._path("s4")
        data = json.loads(path.read_text(encoding="utf-8"))
        data["updated_at"] = time.time() - 100
        path.write_text(json.dumps(data), encoding="utf-8")

        snapshots = asyncio.run(svc.tick())
        assert snapshots[0].health == "stale"
        assert "robot observation stale" in snapshots[0].summary

    def test_watchdog_healthy(self, tmp_path) -> None:
        svc = _make_supervisor(tmp_path)
        _setup_episode_and_task(svc, "s5", "healthy task")
        robot_state = svc.robot_states.ensure("s5", agent_id="main", robot_id="mock0")
        robot_state.last_status = {"battery": 0.8, "timestamp": time.time()}
        robot_state.last_observation_frame = 99
        svc.robot_states.save(robot_state)

        snapshots = asyncio.run(svc.tick())
        assert snapshots[0].health == "healthy"
        assert snapshots[0].summary == "task supervisor checks passed"

    def test_watchdog_healthy_obs_without_frame(self, tmp_path) -> None:
        svc = _make_supervisor(tmp_path)
        _setup_episode_and_task(svc, "s6", "task no frame")
        robot_state = svc.robot_states.ensure("s6", agent_id="main", robot_id="mock0")
        robot_state.last_observation_frame = None
        svc.robot_states.save(robot_state)

        snapshots = asyncio.run(svc.tick())
        assert snapshots[0].health == "healthy"

    def test_watchdog_ignores_terminal_state_active_skill(self, tmp_path) -> None:
        svc = _make_supervisor(tmp_path)
        _setup_episode_and_task(svc, "s7", "already confirmed skill")
        robot_state = svc.robot_states.ensure("s7", agent_id="main", robot_id="mock0")
        robot_state.active_skill_id = "cmd_done"
        robot_state.active_skill_phase = "confirmed"
        robot_state.last_status = {"battery": 0.8, "timestamp": time.time() - 100}
        svc.robot_states.save(robot_state)

        snapshots = asyncio.run(svc.tick())
        assert snapshots[0].health == "healthy"
        assert snapshots[0].active_skill_id is None

    def test_watchdog_camera_quality_blocked_black_frame(self, tmp_path) -> None:
        svc = _make_supervisor(tmp_path)
        _setup_episode_and_task(svc, "s8", "camera task")
        robot_state = svc.robot_states.ensure("s8", agent_id="main", robot_id="mock0")
        robot_state.active_skill_id = "cmd1"
        robot_state.last_status = {
            "battery": 0.8,
            "timestamp": time.time(),
            "metrics": {
                "camera": {
                    "ok": False,
                    "frame_available": False,
                    "valid_image_count": 0,
                    "image_quality_issues": ["black_frame"],
                    "age_ms": 5000,
                }
            },
        }
        svc.robot_states.save(robot_state)

        snapshots = asyncio.run(svc.tick())
        assert snapshots[0].health == "blocked"
        assert "camera blocked" in snapshots[0].summary
        assert "black_frame" in snapshots[0].summary

    def test_watchdog_camera_quality_blocked_no_valid_images(self, tmp_path) -> None:
        svc = _make_supervisor(tmp_path)
        _setup_episode_and_task(svc, "s9", "obs task")
        robot_state = svc.robot_states.ensure("s9", agent_id="main", robot_id="mock0")
        robot_state.active_skill_id = "cmd1"
        robot_state.last_status = {
            "battery": 0.8,
            "timestamp": time.time(),
            "metrics": {
                "camera": {
                    "ok": True,
                    "frame_available": False,
                    "valid_image_count": 0,
                    "image_quality_issues": [],
                }
            },
        }
        svc.robot_states.save(robot_state)

        snapshots = asyncio.run(svc.tick())
        assert snapshots[0].health == "blocked"
        assert "camera blocked" in snapshots[0].summary

    def test_watchdog_camera_quality_blocked_stale_camera(self, tmp_path) -> None:
        svc = _make_supervisor(tmp_path)
        _setup_episode_and_task(svc, "s10", "stale camera task")
        robot_state = svc.robot_states.ensure("s10", agent_id="main", robot_id="mock0")
        robot_state.active_skill_id = "cmd1"
        robot_state.last_status = {
            "battery": 0.8,
            "timestamp": time.time(),
            "metrics": {
                "camera": {
                    "ok": True,
                    "frame_available": True,
                    "valid_image_count": 3,
                    "age_ms": 30000,
                }
            },
        }
        svc.robot_states.save(robot_state)

        snapshots = asyncio.run(svc.tick())
        assert snapshots[0].health == "blocked"
        assert "camera blocked" in snapshots[0].summary
        assert "age_ms=30000" in snapshots[0].summary

    def test_watchdog_healthy_with_good_camera(self, tmp_path) -> None:
        svc = _make_supervisor(tmp_path)
        _setup_episode_and_task(svc, "s11", "good camera task")
        robot_state = svc.robot_states.ensure("s11", agent_id="main", robot_id="mock0")
        robot_state.last_status = {
            "battery": 0.8,
            "timestamp": time.time(),
            "metrics": {
                "camera": {
                    "ok": True,
                    "frame_available": True,
                    "valid_image_count": 3,
                }
            },
        }
        robot_state.last_observation_frame = 99
        svc.robot_states.save(robot_state)

        snapshots = asyncio.run(svc.tick())
        assert snapshots[0].health == "healthy"

    def test_watchdog_healthy_when_camera_metrics_absent(self, tmp_path) -> None:
        svc = _make_supervisor(tmp_path)
        _setup_episode_and_task(svc, "s12", "no camera metrics")
        robot_state = svc.robot_states.ensure("s12", agent_id="main", robot_id="mock0")
        robot_state.active_skill_id = "cmd1"
        robot_state.last_status = {"battery": 0.8, "timestamp": time.time()}
        svc.robot_states.save(robot_state)

        snapshots = asyncio.run(svc.tick())
        assert snapshots[0].health == "healthy"


class TestRecoveryStrategies:
    def test_recovery_paused_skips_recovery_decision(self, tmp_path) -> None:
        svc = _make_supervisor(tmp_path)
        _setup_episode_and_task(svc, "s1", "paused task")
        svc.task_runs.pause("s1", reason="operator paused")

        asyncio.run(svc.tick())
        tasks = svc.task_runs.list_recent(limit=10)
        task = next(t for t in tasks if t.episode_id == "s1")
        assert task.recovery is None
        assert task.status == "paused"

    def test_recovery_stale_uses_pause_for_operator(self, tmp_path) -> None:
        svc = _make_supervisor(tmp_path)
        svc.status_stale_sec = 0.0
        _setup_episode_and_task(svc, "s2", "stale recovery")
        robot_state = svc.robot_states.ensure("s2", agent_id="main", robot_id="mock0")
        robot_state.active_skill_id = "cmd1"
        robot_state.last_status = {"battery": 0.8, "timestamp": time.time() - 100}
        svc.robot_states.save(robot_state)

        asyncio.run(svc.tick())
        task = svc.task_runs.load_active("s2")
        assert task is not None
        assert task.recovery is not None
        assert task.recovery["strategy"] == "pause_for_operator"

    def test_recovery_blocked_continue_from_observation(self, tmp_path) -> None:
        svc = _make_supervisor(tmp_path)
        _setup_episode_and_task(svc, "s3", "blocked recovery")
        robot_state = svc.robot_states.ensure("s3", agent_id="main", robot_id="mock0")
        robot_state.recovery_required = True
        robot_state.recovery_reason = "grasp failed"
        svc.robot_states.save(robot_state)

        asyncio.run(svc.tick())
        task = svc.task_runs.load_active("s3")
        assert task is not None
        assert task.recovery is not None
        assert task.recovery["strategy"] == "continue_from_observation"

    def test_repeated_blocked_ticks_do_not_append_duplicate_watchdog_or_recovery_events(
        self, tmp_path
    ) -> None:
        svc = _make_supervisor(tmp_path)
        _setup_episode_and_task(svc, "s4", "blocked recovery dedupe")
        robot_state = svc.robot_states.ensure("s4", agent_id="main", robot_id="mock0")
        robot_state.recovery_required = True
        robot_state.recovery_reason = "grasp failed"
        svc.robot_states.save(robot_state)

        asyncio.run(svc.tick())
        events_after_first = svc.task_runs.events.recent("s4", limit=20)
        asyncio.run(svc.tick())
        events_after_second = svc.task_runs.events.recent("s4", limit=20)

        first_watchdogs = [
            event for event in events_after_first if event.kind == "watchdog"
        ]
        second_watchdogs = [
            event for event in events_after_second if event.kind == "watchdog"
        ]
        first_recoveries = [
            event for event in events_after_first if event.kind == "recovery_selected"
        ]
        second_recoveries = [
            event for event in events_after_second if event.kind == "recovery_selected"
        ]

        assert len(first_watchdogs) == 1
        assert len(second_watchdogs) == 1
        assert len(first_recoveries) == 1
        assert len(second_recoveries) == 1

    def test_health_transition_appends_new_watchdog_and_recovery_events(
        self, tmp_path
    ) -> None:
        svc = _make_supervisor(tmp_path)
        svc.status_stale_sec = 0.0
        _setup_episode_and_task(svc, "s5", "health transition")
        robot_state = svc.robot_states.ensure("s5", agent_id="main", robot_id="mock0")
        robot_state.active_skill_id = "cmd1"
        robot_state.last_status = {"battery": 0.8, "timestamp": time.time() - 100}
        svc.robot_states.save(robot_state)

        asyncio.run(svc.tick())
        loaded_state = svc.robot_states.load("s5")
        assert loaded_state is not None
        loaded_state.recovery_required = True
        loaded_state.recovery_reason = "grasp failed"
        svc.robot_states.save(loaded_state)

        asyncio.run(svc.tick())
        watchdog_events = svc.task_runs.events.recent("s5", limit=20, kind="watchdog")
        recovery_events = svc.task_runs.events.recent(
            "s5", limit=20, kind="recovery_selected"
        )

        assert len(watchdog_events) == 2
        assert watchdog_events[0].summary.startswith("robot status stale")
        assert watchdog_events[-1].summary == "grasp failed"
        assert len(recovery_events) == 2
        assert recovery_events[0].metadata["strategy"] == "pause_for_operator"
        assert recovery_events[-1].metadata["strategy"] == "continue_from_observation"


class TestWatchdogNotifications:
    def test_should_notify_when_health_changes(self, tmp_path) -> None:
        svc = _make_supervisor(tmp_path)
        svc.status_stale_sec = 0.0
        _setup_episode_and_task(svc, "s1", "notify change")
        robot_state = svc.robot_states.ensure("s1", agent_id="main", robot_id="mock0")
        robot_state.active_skill_id = "cmd1"
        robot_state.last_status = {"battery": 0.8, "timestamp": time.time() - 100}
        svc.robot_states.save(robot_state)

        asyncio.run(svc.tick())
        bus = cast(FakeBus, svc.bus)
        replies = [p for t, p in bus.published if t == svc.topics.agent_reply]
        assert len(replies) >= 1
        assert replies[-1]["metadata"]["event"] == "task_watchdog"

    def test_should_notify_suppressed_when_health_unchanged(self, tmp_path) -> None:
        svc = _make_supervisor(tmp_path)
        svc.status_stale_sec = 0.0
        _setup_episode_and_task(svc, "s2", "suppress notify")
        robot_state = svc.robot_states.ensure("s2", agent_id="main", robot_id="mock0")
        robot_state.active_skill_id = "cmd1"
        robot_state.last_status = {"battery": 0.8, "timestamp": time.time() - 100}
        svc.robot_states.save(robot_state)

        asyncio.run(svc.tick())
        bus = cast(FakeBus, svc.bus)
        bus.published.clear()
        asyncio.run(svc.tick())
        second_notify_count = len(
            [p for t, p in bus.published if t == svc.topics.agent_reply]
        )
        assert second_notify_count == 0

    def test_should_notify_suppressed_for_healthy(self, tmp_path) -> None:
        svc = _make_supervisor(tmp_path)
        _setup_episode_and_task(svc, "s3", "healthy notify")
        robot_state = svc.robot_states.ensure("s3", agent_id="main", robot_id="mock0")
        robot_state.last_status = {"battery": 0.8, "timestamp": time.time()}
        svc.robot_states.save(robot_state)

        asyncio.run(svc.tick())
        bus = cast(FakeBus, svc.bus)
        replies = [p for t, p in bus.published if t == svc.topics.agent_reply]
        assert len(replies) == 0


class TestTickEdgeCases:
    def test_tick_skips_completed_tasks(self, tmp_path) -> None:
        svc = _make_supervisor(tmp_path)
        _setup_episode_and_task(svc, "s1", "done task")
        task = svc.task_runs.load_active("s1")
        assert task is not None
        task.status = "completed"
        svc.task_runs.save(task)

        snapshots = asyncio.run(svc.tick())
        assert len(snapshots) == 0

    def test_tick_skips_failed_tasks(self, tmp_path) -> None:
        svc = _make_supervisor(tmp_path)
        _setup_episode_and_task(svc, "s2", "failed task")
        task = svc.task_runs.load_active("s2")
        assert task is not None
        task.status = "failed"
        svc.task_runs.save(task)

        snapshots = asyncio.run(svc.tick())
        assert len(snapshots) == 0

    def test_tick_skips_cancelled_tasks(self, tmp_path) -> None:
        svc = _make_supervisor(tmp_path)
        _setup_episode_and_task(svc, "s3", "cancelled task")
        task = svc.task_runs.load_active("s3")
        assert task is not None
        task.status = "cancelled"
        svc.task_runs.save(task)

        snapshots = asyncio.run(svc.tick())
        assert len(snapshots) == 0


class TestSkillEventHandling:
    def test_on_skill_event_without_episode_id_is_noop(self, tmp_path) -> None:
        svc = _make_supervisor(tmp_path)
        _setup_episode_and_task(svc, "s1", "a task")
        svc.task_runs.ensure_active(
            episode_id="s1", task="a task", agent_id="main", robot_id="mock0"
        )

        event = SkillEvent(
            envelope=Envelope(episode_id="", agent_id="main"),
            skill_id="cmd_x",
            phase="executing",
        )
        asyncio.run(svc._on_skill_event("skill.event", to_payload(event)))
        task = svc.task_runs.load_active("s1")
        assert task is not None
        assert svc.task_runs.events.recent("s1", kind="skill_event") == []


def test_publish_watchdog_runtime_error_is_silent(tmp_path) -> None:
    svc = _make_supervisor(tmp_path)
    _setup_episode_and_task(svc, "s1", "task")

    class FailingBus(BusClient):
        def __init__(self) -> None:
            super().__init__(url="test://fake")

        async def connect(self) -> None:
            self._connected = True

        async def close(self) -> None:
            self._connected = False

        async def subscribe(self, topics, handler) -> None:
            pass

        async def publish(self, _topic: str, _payload: dict) -> None:
            raise RuntimeError("bus down")

    svc.bus = FailingBus()
    snapshots = asyncio.run(svc.tick())
    assert len(snapshots) > 0
