from __future__ import annotations

import asyncio

from hey_robot.agents.robot_agent import RobotAgentService
from hey_robot.agents.session import AgentTurnState
from hey_robot.config import DeploymentConfig
from hey_robot.episode.scope import EpisodeScope
from hey_robot.protocol import (
    Envelope,
    RobotStatus,
    SkillEvent,
    SkillIntent,
    SkillResult,
    UserTurn,
)
from hey_robot.protocol.messages import to_payload


def _config(tmp_path) -> DeploymentConfig:
    return DeploymentConfig.from_dict(
        {
            "resources": {
                "runtime_dir": str(tmp_path / "runtime"),
                "media": {"root": str(tmp_path / "runtime" / "media")},
                "episodes": {"root": str(tmp_path / "runtime" / "episodes")},
            },
            "agents": {
                "main": {
                    "type": "robot_agent",
                    "robot_id": "mock0",
                    "settings": {
                        "mode": "direct",
                        "execution_feedback": {"backend": "status"},
                    },
                }
            },
            "robots": {"mock0": {"type": "mock"}},
        }
    )


def test_agent_completed_result_commits_execution_feedback(tmp_path) -> None:
    class FakeBus:
        def __init__(self) -> None:
            self.published: list[tuple[str, dict]] = []

        async def publish(self, topic: str, payload: dict) -> None:
            self.published.append((topic, payload))

    config = _config(tmp_path)
    service = RobotAgentService(config, agent_id="main", episode_dir=tmp_path)
    fake_bus = FakeBus()
    service.bus = fake_bus  # type: ignore[assignment]
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr1", episode_id="s1", agent_id="main", robot_id="mock0"
        ),
        text="move",
    )
    skill = SkillIntent(envelope=turn.envelope, skill_id="cmd1", objective="move")
    service.turn_sessions.active_turns["tr1"] = AgentTurnState(
        turn=turn, skill_id="cmd1"
    )
    service.core.skill_state.submit(skill)
    service.robot_states.mark_task_started(
        "s1", task="move", agent_id="main", robot_id="mock0"
    )
    service.task_runtime.task_runs.ensure_active(
        episode_id="s1", task="move", agent_id="main", robot_id="mock0"
    )
    service.task_runtime.task_runs.bind_skill("s1", "cmd1", "move")
    service.robot_states.apply_skill_event(
        SkillEvent(
            envelope=turn.envelope, skill_id="cmd1", phase="executing", frame_id=3
        )
    )

    asyncio.run(
        service._on_skill_result(
            service.topics.skill_result,
            to_payload(
                SkillResult(
                    envelope=turn.envelope,
                    skill_id="cmd1",
                    status="completed",
                    frame_id=4,
                )
            ),
        )
    )

    phases = [
        payload.get("phase")
        for topic, payload in fake_bus.published
        if topic == service.topics.skill_event
    ]
    state = service.robot_states.load("s1")
    task = service.task_runtime.task_runs.list_for_episode("s1")[0]
    active_turn = service.turn_sessions.active_turns.get("tr1")

    assert "feedback_pending" not in phases
    assert "confirmed" in phases
    assert state is not None
    assert state.active_skill_phase == "confirmed"
    assert state.last_observation_frame == 4
    assert state.recovery_required is False
    assert task.status == "active"
    assert task.last_step_success is True
    assert task.task_success is None
    assert active_turn is not None
    assert active_turn.status == "completed"


def test_agent_completed_result_does_not_publish_task_completion_notification(
    tmp_path,
) -> None:
    class FakeBus:
        def __init__(self) -> None:
            self.published: list[tuple[str, dict]] = []

        async def publish(self, topic: str, payload: dict) -> None:
            self.published.append((topic, payload))

    config = _config(tmp_path)
    service = RobotAgentService(config, agent_id="main", episode_dir=tmp_path)
    fake_bus = FakeBus()
    service.bus = fake_bus  # type: ignore[assignment]
    service.episodes.ensure("s1", scope=EpisodeScope(agent_id="main"), aliases=[])
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr_notify",
            episode_id="s1",
            agent_id="main",
            robot_id="mock0",
            channel="feishu",
            chat_id="oc_chat_1",
            sender_id="ou_user_1",
        ),
        text="move",
    )
    service.episodes.append_user_turn("s1", turn)
    skill = SkillIntent(envelope=turn.envelope, skill_id="cmd_notify", objective="move")
    service.turn_sessions.active_turns["tr_notify"] = AgentTurnState(
        turn=turn, skill_id="cmd_notify"
    )
    service.core.skill_state.submit(skill)
    service.latest_status["mock0"] = RobotStatus(
        envelope=Envelope(robot_id="mock0"),
        frame_id=4,
        state="terminated",
        success=True,
        metrics={"is_success": True},
    )
    service.robot_states.mark_task_started(
        "s1", task="move", agent_id="main", robot_id="mock0"
    )
    service.task_runtime.task_runs.ensure_active(
        episode_id="s1", task="move", agent_id="main", robot_id="mock0"
    )
    service.task_runtime.task_runs.bind_skill("s1", "cmd_notify", "move")

    asyncio.run(
        service._on_skill_result(
            service.topics.skill_result,
            to_payload(
                SkillResult(
                    envelope=turn.envelope,
                    skill_id="cmd_notify",
                    status="completed",
                    frame_id=4,
                )
            ),
        )
    )

    replies = [
        payload
        for topic, payload in fake_bus.published
        if topic == service.topics.agent_reply
    ]

    assert not any(item["metadata"].get("proactive") is True for item in replies)


def test_publish_task_result_is_task_completion_path(tmp_path) -> None:
    class FakeBus:
        def __init__(self) -> None:
            self.published: list[tuple[str, dict]] = []

        async def publish(self, topic: str, payload: dict) -> None:
            self.published.append((topic, payload))

    config = _config(tmp_path)
    service = RobotAgentService(config, agent_id="main", episode_dir=tmp_path)
    fake_bus = FakeBus()
    service.bus = fake_bus  # type: ignore[assignment]
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr_report", episode_id="s1", agent_id="main", robot_id="mock0"
        ),
        text="move",
    )
    service.core._current_turn = turn
    service.task_runtime.task_runs.ensure_active(
        episode_id="s1", task="move", agent_id="main", robot_id="mock0"
    )

    asyncio.run(service.publish_task_result(success=True, summary="task complete"))

    task = service.task_runtime.task_runs.list_for_episode("s1")[0]
    replies = [
        payload
        for topic, payload in fake_bus.published
        if topic == service.topics.agent_reply
    ]

    assert task.status == "completed"
    assert task.task_success is True
    assert task.finished_at is not None
    assert replies[-1]["metadata"]["task_success"] is True


def test_agent_arm_status_result_updates_turn_state_without_reply(tmp_path) -> None:
    class FakeBus:
        def __init__(self) -> None:
            self.published: list[tuple[str, dict]] = []

        async def publish(self, topic: str, payload: dict) -> None:
            self.published.append((topic, payload))

    config = _config(tmp_path)
    service = RobotAgentService(config, agent_id="main", episode_dir=tmp_path)
    fake_bus = FakeBus()
    service.bus = fake_bus  # type: ignore[assignment]
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr_arm", episode_id="s1", agent_id="main", robot_id="mock0"
        ),
        text="What is the arm status?",
    )
    service.turn_sessions.active_turns["tr_arm"] = AgentTurnState(
        turn=turn, skill_id="cmd_arm"
    )
    service.latest_status["mock0"] = RobotStatus(
        envelope=Envelope(robot_id="mock0"),
        metrics={
            "arm_status": {
                "enabled": True,
                "initialized": True,
                "joint_states": {
                    "base": 83.7,
                    "shoulder": 0.0,
                    "elbow": 158.6,
                    "gripper": 0.7,
                },
            }
        },
    )

    asyncio.run(
        service._on_skill_result(
            service.topics.skill_result,
            to_payload(
                SkillResult(
                    envelope=turn.envelope,
                    skill_id="cmd_arm",
                    name="arm_status",
                    status="completed",
                )
            ),
        )
    )

    active_turn = service.turn_sessions.active_turns.get("tr_arm")
    assert active_turn is not None
    assert active_turn.status == "completed"


def test_agent_failed_result_publishes_recovery_notification(tmp_path) -> None:
    class FakeBus:
        def __init__(self) -> None:
            self.published: list[tuple[str, dict]] = []

        async def publish(self, topic: str, payload: dict) -> None:
            self.published.append((topic, payload))

    config = _config(tmp_path)
    service = RobotAgentService(config, agent_id="main", episode_dir=tmp_path)
    fake_bus = FakeBus()
    service.bus = fake_bus  # type: ignore[assignment]
    service.episodes.ensure("s1", scope=EpisodeScope(agent_id="main"), aliases=[])
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr_recovery",
            episode_id="s1",
            agent_id="main",
            robot_id="mock0",
            channel="feishu",
            chat_id="oc_chat_2",
            sender_id="ou_user_2",
        ),
        text="pick up cup",
    )
    service.episodes.append_user_turn("s1", turn)
    service.robot_states.mark_task_started(
        "s1", task="pick up cup", agent_id="main", robot_id="mock0"
    )
    service.task_runtime.task_runs.ensure_active(
        episode_id="s1",
        task="pick up cup",
        agent_id="main",
        robot_id="mock0",
    )
    service.task_runtime.task_runs.bind_skill("s1", "cmd_fail", "pick up cup")

    asyncio.run(
        service._on_skill_result(
            service.topics.skill_result,
            to_payload(
                SkillResult(
                    envelope=turn.envelope,
                    skill_id="cmd_fail",
                    status="failed",
                    error="gripper blocked",
                )
            ),
        )
    )

    replies = [
        payload
        for topic, payload in fake_bus.published
        if topic == service.topics.agent_reply
    ]
    proactive = next(
        item for item in replies if item["metadata"].get("proactive") is True
    )

    assert proactive["envelope"]["chat_id"] == "oc_chat_2"
    assert proactive["metadata"]["task_status"] == "recovering"
    assert proactive["metadata"]["active_task"] == "pick up cup"
    assert proactive["metadata"]["continuation_goal"] == "pick up cup"
    assert proactive["metadata"]["recovery_strategy"] == "reobserve"
    assert proactive["metadata"]["recovery_actions"]
    assert "pick up cup" in proactive["text"]
