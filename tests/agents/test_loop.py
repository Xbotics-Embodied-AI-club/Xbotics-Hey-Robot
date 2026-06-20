from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from hey_robot.agents.core import RobotAgentCore
from hey_robot.agents.execution_feedback import status_feedback_from_result
from hey_robot.agents.injection import RobotTurnInjector
from hey_robot.agents.interaction import classify_user_interaction
from hey_robot.agents.loop import RobotAgentLoop
from hey_robot.agents.perception_query import SceneEvidence
from hey_robot.agents.robot_agent import RobotAgentService
from hey_robot.agents.task_run import TaskAttempt, TaskRun
from hey_robot.agents.task_runtime import TaskRunManager
from hey_robot.agents.types import RobotSnapshot
from hey_robot.config import AgentSpec, DeploymentConfig
from hey_robot.episode import EpisodeRecord, RobotEpisodeStateStore
from hey_robot.events.bus import BusEventPublisher
from hey_robot.perception.scene import SceneUnderstanding
from hey_robot.protocol import (
    AgentReply,
    Envelope,
    ImageRef,
    RobotObservation,
    RobotStatus,
    SkillIntent,
    SkillResult,
    UserTurn,
)


@dataclass
class FakeAgentIO:
    skills: list[SkillIntent] = field(default_factory=list)
    replies: list[AgentReply] = field(default_factory=list)

    async def submit_skill(self, skill: SkillIntent) -> None:
        self.skills.append(skill)

    async def publish_reply(self, reply: AgentReply) -> None:
        self.replies.append(reply)

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
        del channel, chat_id, sender_id, message_id, reply_to_current, metadata
        self.replies.append(AgentReply(envelope=Envelope(), text=text))

    async def publish_task_result(self, *, success: bool, summary: str) -> None:
        self.replies.append(
            AgentReply(envelope=Envelope(), text=summary, metadata={"success": success})
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
        del robot_id, question, baseline_frame_id, freshness, timeout_sec
        return SceneEvidence(
            status="no_observation", summary="No observation available."
        )


def test_robot_agent_loop_runs_core_and_writes_checkpoint(tmp_path) -> None:
    io = FakeAgentIO()
    core = RobotAgentCore(
        agent_id="main",
        spec=AgentSpec(type="robot_agent", settings={"mode": "direct"}),
        io=io,
    )
    task_runtime = TaskRunManager(
        episode_root=tmp_path,
        runtime_dir=tmp_path,
        events_max_items=100,
        robot_states=RobotEpisodeStateStore(tmp_path),
    )
    loop = RobotAgentLoop(core, task_runtime=task_runtime)
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr1", episode_id="s1", agent_id="main", robot_id="mock0"
        ),
        text="pick up the bowl",
    )
    history = [
        EpisodeRecord(role="user", content="previous task", timestamp=1.0, payload={})
    ]

    async def progress_callback(_progress) -> None:
        return None

    _result, trace = asyncio.run(
        loop.run_turn(
            turn=turn,
            snapshot=RobotSnapshot(robot_id="mock0"),
            history=history,
            recovery_context=None,
            progress_callback=progress_callback,
        )
    )

    checkpoint = task_runtime.checkpoints.load("s1")

    assert io.skills[0].objective == "pick up the bowl"
    assert [entry.state for entry in trace] == ["restore", "build", "run", "save"]
    assert checkpoint is not None
    assert checkpoint.phase == "responded"
    assert checkpoint.skill_id == io.skills[0].skill_id
    task = task_runtime.task_runs.load_active("s1")
    assert task is not None
    assert task.skill_ids == [io.skills[0].skill_id]
    assert task.attempts[0].status == "executing"
    assert task.root_task == "pick up the bowl"


def test_robot_turn_injector_merges_correction_into_active_task() -> None:
    injector = RobotTurnInjector()
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr2", episode_id="s2", agent_id="main", robot_id="mock0"
        ),
        text="\u4e0d\u662f\u5de6\u8fb9\uff0c\u662f\u53f3\u8fb9",
    )
    task = TaskRun(
        task_id="task1",
        episode_id="s2",
        root_task="pick up the bowl",
        active_attempt_id="attempt1",
        attempts=[TaskAttempt(attempt_id="attempt1", text="move to the bowl")],
    )

    injected = injector.inject(
        turn=turn,
        intent=classify_user_interaction(turn.text, robot_busy=True),
        task=task,
        snapshot=RobotSnapshot(robot_id="mock0"),
    )

    assert "pick up the bowl" in injected.text
    assert "\u4e0d\u662f\u5de6\u8fb9\uff0c\u662f\u53f3\u8fb9" in injected.text


def test_status_feedback_uses_robot_success_metric() -> None:
    result = SkillResult(
        envelope=Envelope(robot_id="mock0"), skill_id="cmd1", status="completed"
    )
    status = RobotStatus(
        envelope=Envelope(robot_id="mock0"),
        frame_id=10,
        state="terminated",
        success=True,
        metrics={"is_success": True},
    )

    feedback = status_feedback_from_result(result, status)

    assert feedback.subgoal_success is True
    assert feedback.task_success is True
    assert feedback.outcome == "confirmed"


def test_status_feedback_fails_unsuccessful_get_robot_status() -> None:
    result = SkillResult(
        envelope=Envelope(robot_id="mock0"), skill_id="cmd1", status="completed"
    )
    status = RobotStatus(
        envelope=Envelope(robot_id="mock0"),
        frame_id=10,
        state="acting",
        success=False,
        metrics={"is_success": False},
    )

    feedback = status_feedback_from_result(result, status)

    assert feedback.subgoal_success is False
    assert feedback.task_success is False
    assert feedback.outcome == "failed"


def test_robot_agent_service_status_feedback_updates_task_run(tmp_path) -> None:
    class FakeBus:
        def __init__(self) -> None:
            self.published: list[tuple[str, dict]] = []

        async def publish(self, topic: str, payload: dict) -> None:
            self.published.append((topic, payload))

    config = DeploymentConfig.from_dict(
        {
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
    service = RobotAgentService(config, agent_id="main", episode_dir=tmp_path)
    fake_bus = FakeBus()
    service.bus = fake_bus  # type: ignore[assignment]
    service.latest_status["mock0"] = RobotStatus(
        envelope=Envelope(robot_id="mock0"),
        frame_id=5,
        state="terminated",
        success=True,
        metrics={"is_success": True},
    )
    service.task_runtime.task_runs.ensure_active(
        episode_id="s1",
        task="put bowl down",
        agent_id="main",
        robot_id="mock0",
    )
    service.task_runtime.task_runs.bind_skill("s1", "cmd1", "put bowl down")
    result = SkillResult(
        envelope=Envelope(
            trace_id="tr1", episode_id="s1", agent_id="main", robot_id="mock0"
        ),
        skill_id="cmd1",
        status="completed",
    )

    feedback = asyncio.run(service._evaluate_execution_feedback(result))
    task = asyncio.run(service._commit_execution_feedback(result, feedback))
    asyncio.run(service._publish_execution_feedback_event(result, feedback))
    state = service.robot_states.load("s1")
    phases = [
        payload["phase"]
        for topic, payload in fake_bus.published
        if topic == service.topics.skill_event
    ]

    assert feedback.successful is True
    assert phases == ["confirmed"]
    assert state is None or state.recovery_required is False
    assert task is not None
    assert task.status == "active"
    assert task.task_success is None


def test_robot_agent_service_queues_busy_correction(tmp_path) -> None:
    class FakeBus:
        def __init__(self) -> None:
            self.published: list[tuple[str, dict]] = []

        async def publish(self, topic: str, payload: dict) -> None:
            self.published.append((topic, payload))

    config = DeploymentConfig.from_dict(
        {
            "agents": {
                "main": {
                    "type": "robot_agent",
                    "robot_id": "mock0",
                    "settings": {"mode": "direct"},
                }
            },
            "robots": {"mock0": {"type": "mock"}},
        }
    )
    service = RobotAgentService(config, agent_id="main", episode_dir=tmp_path)
    service.bus = FakeBus()  # type: ignore[assignment]
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr1", episode_id="s1", agent_id="main", robot_id="mock0"
        ),
        text="not that one, use the left bowl",
    )

    handled = asyncio.run(service._handle_busy_turn(turn, active_skill_id="cmd1"))
    checkpoint = service.task_runtime.checkpoints.load("s1")

    assert handled is True
    assert checkpoint is not None
    assert len(checkpoint.pending_turns) == 1
    assert checkpoint.pending_turns[0]["text"] == "not that one, use the left bowl"
    assert checkpoint.pending_turns[0]["metadata"]["_pending_reason"] == "follow_up"


def test_robot_agent_service_busy_executing_task_keeps_correction_intent(
    tmp_path,
) -> None:
    class FakeBus:
        def __init__(self) -> None:
            self.published: list[tuple[str, dict]] = []

        async def publish(self, topic: str, payload: dict) -> None:
            self.published.append((topic, payload))

    config = DeploymentConfig.from_dict(
        {
            "agents": {
                "main": {
                    "type": "robot_agent",
                    "robot_id": "mock0",
                    "settings": {"mode": "direct"},
                }
            },
            "robots": {"mock0": {"type": "mock"}},
        }
    )
    service = RobotAgentService(config, agent_id="main", episode_dir=tmp_path)
    service.bus = FakeBus()  # type: ignore[assignment]
    service.task_runtime.task_runs.ensure_active(
        episode_id="s1",
        task="pick the left bowl",
        agent_id="main",
        robot_id="mock0",
    )
    service.task_runtime.task_runs.bind_skill("s1", "cmd1", "approach bowl")
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr1", episode_id="s1", agent_id="main", robot_id="mock0"
        ),
        text="not that one, use the left bowl",
    )

    handled = asyncio.run(service._handle_busy_turn(turn, active_skill_id="cmd1"))
    checkpoint = service.task_runtime.checkpoints.load("s1")

    assert handled is True
    assert checkpoint is not None
    assert checkpoint.pending_turns[0]["metadata"]["_pending_reason"] == "correction"


def test_robot_agent_service_busy_recovering_task_downgrades_correction_to_follow_up(
    tmp_path,
) -> None:
    class FakeBus:
        def __init__(self) -> None:
            self.published: list[tuple[str, dict]] = []

        async def publish(self, topic: str, payload: dict) -> None:
            self.published.append((topic, payload))

    config = DeploymentConfig.from_dict(
        {
            "agents": {
                "main": {
                    "type": "robot_agent",
                    "robot_id": "mock0",
                    "settings": {"mode": "direct"},
                }
            },
            "robots": {"mock0": {"type": "mock"}},
        }
    )
    service = RobotAgentService(config, agent_id="main", episode_dir=tmp_path)
    service.bus = FakeBus()  # type: ignore[assignment]
    task = service.task_runtime.task_runs.ensure_active(
        episode_id="s1",
        task="pick the bowl",
        agent_id="main",
        robot_id="mock0",
    )
    assert task is not None
    task.status = "recovering"
    service.task_runtime.task_runs.save(task)
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr1", episode_id="s1", agent_id="main", robot_id="mock0"
        ),
        text="not that one, use the left bowl",
    )

    handled = asyncio.run(service._handle_busy_turn(turn, active_skill_id="cmd1"))
    checkpoint = service.task_runtime.checkpoints.load("s1")

    assert handled is True
    assert checkpoint is not None
    assert checkpoint.pending_turns[0]["metadata"]["_pending_reason"] == "follow_up"


def test_robot_agent_service_answers_busy_readonly_status(tmp_path) -> None:
    class FakeBus:
        def __init__(self) -> None:
            self.published: list[tuple[str, dict]] = []

        async def publish(self, topic: str, payload: dict) -> None:
            self.published.append((topic, payload))

    config = DeploymentConfig.from_dict(
        {
            "agents": {
                "main": {
                    "type": "robot_agent",
                    "robot_id": "mock0",
                    "settings": {"mode": "direct"},
                }
            },
            "robots": {"mock0": {"type": "mock"}},
        }
    )
    service = RobotAgentService(config, agent_id="main", episode_dir=tmp_path)
    fake_bus = FakeBus()
    service.bus = fake_bus  # type: ignore[assignment]
    service.latest_status["mock0"] = RobotStatus(
        envelope=Envelope(robot_id="mock0"),
        state="acting",
        metrics={
            "battery": {"status": "normal", "percentage": 50.0, "voltage": 10.8},
            "arm": {"state": "ready", "joint_count": 6},
        },
    )
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr1", episode_id="s1", agent_id="main", robot_id="mock0"
        ),
        text="what is the battery status?",
    )

    handled = asyncio.run(service._handle_busy_turn(turn, active_skill_id="cmd1"))
    replies = [
        payload
        for topic, payload in fake_bus.published
        if topic == service.topics.agent_reply
    ]
    checkpoint = service.task_runtime.checkpoints.load("s1")

    assert handled is True
    assert replies
    assert replies[-1]["metadata"]["readonly"] is True
    assert "Battery: 50.0%, 10.8V, normal." in replies[-1]["text"]
    assert checkpoint is None or checkpoint.pending_turns == []


def test_robot_agent_service_publishes_final_response(tmp_path) -> None:
    from tests.conftest import FakeProvider

    class FakeBus:
        def __init__(self) -> None:
            self.published: list[tuple[str, dict]] = []

        async def publish(self, topic: str, payload: dict) -> None:
            self.published.append((topic, payload))

    config = DeploymentConfig.from_dict(
        {
            "agents": {
                "main": {
                    "type": "robot_agent",
                    "robot_id": "mock0",
                    "settings": {
                        "max_iterations": 1,
                        "providers": {
                            "planner": {
                                "type": "openai_compat",
                                "model": "test-model",
                                "api_key": "test-key",
                            }
                        },
                    },
                }
            },
            "robots": {"mock0": {"type": "mock"}},
        }
    )
    service = RobotAgentService(config, agent_id="main", episode_dir=tmp_path)
    fake_bus = FakeBus()
    service.bus = fake_bus  # type: ignore[assignment]
    service.events = BusEventPublisher(fake_bus, service.topics)  # type: ignore[arg-type]
    provider = FakeProvider(["I am Hey Robot.", "I am Hey Robot."])
    service.core.provider = provider
    service.core.runtime.provider = provider
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr1",
            episode_id="s1",
            agent_id="main",
            robot_id="mock0",
            channel="web",
        ),
        text="who are you?",
    )

    asyncio.run(service._handle_user_turn_locked(turn))

    replies = [
        payload
        for topic, payload in fake_bus.published
        if topic == service.topics.agent_reply
    ]
    assert replies
    assert replies[-1]["text"] == "I am Hey Robot."
    assert replies[-1]["metadata"]["source"] == "final_response"


def test_robot_agent_service_scene_turn_uses_scene_evidence_tool_call(tmp_path) -> None:
    class FakeBus:
        def __init__(self) -> None:
            self.published: list[tuple[str, dict]] = []

        async def publish(self, topic: str, payload: dict) -> None:
            self.published.append((topic, payload))

    from tests.conftest import FakeProvider

    config = DeploymentConfig.from_dict(
        {
            "agents": {
                "main": {
                    "type": "robot_agent",
                    "robot_id": "mock0",
                    "settings": {
                        "skill_timeout_sec": 1.0,
                        "providers": {
                            "planner": {
                                "type": "openai_compat",
                                "model": "test-model",
                                "api_key": "test-key",
                            }
                        },
                    },
                }
            },
            "robots": {"mock0": {"type": "xlerobot"}},
        }
    )
    service = RobotAgentService(config, agent_id="main", episode_dir=tmp_path)
    fake_bus = FakeBus()
    service.bus = fake_bus  # type: ignore[assignment]
    service.events = BusEventPublisher(fake_bus, service.topics)  # type: ignore[arg-type]
    provider = FakeProvider(
        [
            "I can see a white wall ahead, and a door on the left.",
            "I can see a white wall ahead, and a door on the left.",
        ]
    )
    service.core.provider = provider
    service.core.runtime.provider = provider

    async def submit_skill(skill: SkillIntent) -> None:
        await service._publish_skill_progress_reply(skill)
        await fake_bus.publish(
            service.topics.skill_intent,
            skill.__dict__ | {"envelope": skill.envelope.__dict__},
        )

        async def complete() -> None:
            await asyncio.sleep(0)
            await service._on_observation(
                service.topics.robot_observation,
                RobotObservation(
                    envelope=skill.envelope,
                    frame_id=2,
                    images=[ImageRef(uri="memory://frame2")],
                ).__dict__
                | {"envelope": skill.envelope.__dict__},
            )
            await service._on_skill_result(
                service.topics.skill_result,
                SkillResult(
                    envelope=skill.envelope,
                    skill_id=skill.skill_id,
                    name=skill.name,
                    status="completed",
                    success=True,
                    summary="perception refreshed",
                    frame_id=2,
                ).__dict__
                | {"envelope": skill.envelope.__dict__},
            )

        service._completion_task = asyncio.create_task(complete())  # type: ignore[attr-defined]

    service.submit_skill = submit_skill  # type: ignore[method-assign]
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr1",
            episode_id="s1",
            agent_id="main",
            robot_id="mock0",
            channel="web",
        ),
        text="what do you see ahead?",
    )

    asyncio.run(service._handle_user_turn_locked(turn))

    replies = [
        payload
        for topic, payload in fake_bus.published
        if topic == service.topics.agent_reply
    ]
    intents = [
        payload
        for topic, payload in fake_bus.published
        if topic == service.topics.skill_intent
    ]

    assert intents
    assert intents[0]["name"] == "inspect_scene"
    assert (
        replies[-1]["text"] == "I can see a white wall ahead, and a door on the left."
    )
    assert replies[-1]["metadata"]["source"] == "final_response"
    assert provider.last_messages is not None
    assert any(
        "request_perception" in message.content
        for message in provider.last_messages
        if message.role == "user"
    )


def test_robot_agent_service_motion_turn_uses_request_capability_tool_call(
    tmp_path,
) -> None:
    class FakeBus:
        def __init__(self) -> None:
            self.published: list[tuple[str, dict]] = []

        async def publish(self, topic: str, payload: dict) -> None:
            self.published.append((topic, payload))

    from tests.conftest import FakeProvider

    config = DeploymentConfig.from_dict(
        {
            "agents": {
                "main": {
                    "type": "robot_agent",
                    "robot_id": "mock0",
                    "settings": {
                        "skill_timeout_sec": 1.0,
                        "providers": {
                            "planner": {
                                "type": "openai_compat",
                                "model": "test-model",
                                "api_key": "test-key",
                            }
                        },
                    },
                }
            },
            "robots": {"mock0": {"type": "xlerobot"}},
        }
    )
    service = RobotAgentService(config, agent_id="main", episode_dir=tmp_path)
    fake_bus = FakeBus()
    service.bus = fake_bus  # type: ignore[assignment]
    service.events = BusEventPublisher(fake_bus, service.topics)  # type: ignore[arg-type]
    provider = FakeProvider(
        [
            {
                "tool": "request_capability",
                "args": {
                    "capability": "move_base",
                    "objective": "move forward 10cm",
                    "slots": {"direction": "forward", "distance_cm": 10.0},
                },
            },
            "Moved forward about 10.0 cm.",
        ]
    )
    service.core.provider = provider
    service.core.runtime.provider = provider

    async def submit_skill(skill: SkillIntent) -> None:
        await service._publish_skill_progress_reply(skill)
        await fake_bus.publish(
            service.topics.skill_intent,
            skill.__dict__ | {"envelope": skill.envelope.__dict__},
        )

        async def complete() -> None:
            await asyncio.sleep(0)
            await service._on_skill_result(
                service.topics.skill_result,
                SkillResult(
                    envelope=skill.envelope,
                    skill_id=skill.skill_id,
                    name=skill.name,
                    status="completed",
                    success=True,
                    summary="motion completed",
                ).__dict__
                | {"envelope": skill.envelope.__dict__},
            )

        service._completion_task = asyncio.create_task(complete())  # type: ignore[attr-defined]

    service.submit_skill = submit_skill  # type: ignore[method-assign]
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr1",
            episode_id="s1",
            agent_id="main",
            robot_id="mock0",
            channel="web",
        ),
        text="move forward 10cm",
    )

    asyncio.run(service._handle_user_turn_locked(turn))

    replies = [
        payload
        for topic, payload in fake_bus.published
        if topic == service.topics.agent_reply
    ]
    intents = [
        payload
        for topic, payload in fake_bus.published
        if topic == service.topics.skill_intent
    ]

    assert intents
    assert intents[0]["name"] == "move_base"
    assert intents[0]["arguments"] == {"direction": "forward", "distance_cm": 10.0}
    assert replies[-1]["text"] == "Moved forward about 10.0 cm."
    assert replies[-1]["metadata"]["source"] == "final_response"
    assert provider.last_messages is not None
    assert any(
        message.tool_name == "request_capability"
        for message in provider.last_messages
        if message.role == "tool"
    )


def test_robot_agent_service_publishes_skill_boundary_progress_reply(tmp_path) -> None:
    class FakeBus:
        def __init__(self) -> None:
            self.published: list[tuple[str, dict]] = []

        async def publish(self, topic: str, payload: dict) -> None:
            self.published.append((topic, payload))

    config = DeploymentConfig.from_dict(
        {
            "agents": {
                "main": {
                    "type": "robot_agent",
                    "robot_id": "mock0",
                    "settings": {
                        "providers": {
                            "planner": {
                                "type": "openai_compat",
                                "model": "test-model",
                                "api_key": "test-key",
                            }
                        }
                    },
                }
            },
            "robots": {"mock0": {"type": "mock"}},
        }
    )
    service = RobotAgentService(config, agent_id="main", episode_dir=tmp_path)
    fake_bus = FakeBus()
    service.bus = fake_bus  # type: ignore[assignment]
    service.events = BusEventPublisher(fake_bus, service.topics)  # type: ignore[arg-type]
    skill = SkillIntent(
        envelope=Envelope(
            trace_id="tr1", episode_id="s1", agent_id="main", robot_id="mock0"
        ),
        skill_id="cmd1",
        name="inspect_scene",
        objective="inspect current scene",
    )

    asyncio.run(service.submit_skill(skill))

    replies = [
        payload
        for topic, payload in fake_bus.published
        if topic == service.topics.agent_reply
    ]
    intents = [
        payload
        for topic, payload in fake_bus.published
        if topic == service.topics.skill_intent
    ]
    assert replies == []
    assert intents[0]["skill_id"] == "cmd1"


def test_robot_agent_service_publishes_motion_skill_progress_reply(tmp_path) -> None:
    class FakeBus:
        def __init__(self) -> None:
            self.published: list[tuple[str, dict]] = []

        async def publish(self, topic: str, payload: dict) -> None:
            self.published.append((topic, payload))

    config = DeploymentConfig.from_dict(
        {
            "agents": {
                "main": {
                    "type": "robot_agent",
                    "robot_id": "mock0",
                    "settings": {
                        "providers": {
                            "planner": {
                                "type": "openai_compat",
                                "model": "test-model",
                                "api_key": "test-key",
                            }
                        }
                    },
                }
            },
            "robots": {"mock0": {"type": "mock"}},
        }
    )
    service = RobotAgentService(config, agent_id="main", episode_dir=tmp_path)
    fake_bus = FakeBus()
    service.bus = fake_bus  # type: ignore[assignment]
    service.events = BusEventPublisher(fake_bus, service.topics)  # type: ignore[arg-type]
    skill = SkillIntent(
        envelope=Envelope(
            trace_id="tr1", episode_id="s1", agent_id="main", robot_id="mock0"
        ),
        skill_id="cmd1",
        name="move_base",
        objective="move forward 10cm",
    )

    asyncio.run(service.submit_skill(skill))

    replies = [
        payload
        for topic, payload in fake_bus.published
        if topic == service.topics.agent_reply
    ]
    intents = [
        payload
        for topic, payload in fake_bus.published
        if topic == service.topics.skill_intent
    ]
    assert len(replies) == 1
    assert replies[0]["final"] is False
    assert replies[0]["metadata"]["source"] == "skill_progress"
    assert replies[0]["metadata"]["skill_id"] == "cmd1"
    assert intents[0]["skill_id"] == "cmd1"


def test_skill_result_text_for_agent_includes_scene_summary(tmp_path) -> None:
    class FakeCaptioner:
        async def caption(self, _observation, _status=None):
            return SceneUnderstanding(
                summary="There is a small round table and a folded chair in view.",
                confidence=0.9,
            )

    config = DeploymentConfig.from_dict(
        {
            "agents": {
                "main": {
                    "type": "robot_agent",
                    "robot_id": "mock0",
                    "settings": {
                        "providers": {
                            "planner": {
                                "type": "openai_compat",
                                "model": "test-model",
                                "api_key": "test-key",
                            }
                        }
                    },
                }
            },
            "robots": {"mock0": {"type": "mock"}},
        }
    )
    service = RobotAgentService(config, agent_id="main", episode_dir=tmp_path)
    service.scene_runtime.captioner = FakeCaptioner()
    service.latest_observation["mock0"] = RobotObservation(
        envelope=Envelope(robot_id="mock0"),
        frame_id=7,
        images=[
            ImageRef(
                uri="media://local/images/mock0/frame.jpg",
                camera="front",
                width=10,
                height=10,
            )
        ],
    )
    result = SkillResult(
        envelope=Envelope(robot_id="mock0"),
        skill_id="skill1",
        name="inspect_scene",
        status="completed",
        success=True,
        summary="perception refreshed",
    )

    text = asyncio.run(service.scene_runtime.skill_result_text_for_agent(result))

    assert "perception refreshed" in text
    assert "Observation frame=7" in text
    assert "There is a small round table and a folded chair in view." in text


def test_robot_agent_service_new_turn_clears_stale_recovery_state(tmp_path) -> None:
    class FakeBus:
        def __init__(self) -> None:
            self.published: list[tuple[str, dict]] = []

        async def publish(self, topic: str, payload: dict) -> None:
            self.published.append((topic, payload))

    config = DeploymentConfig.from_dict(
        {
            "agents": {
                "main": {
                    "type": "robot_agent",
                    "robot_id": "mock0",
                    "settings": {"mode": "direct"},
                }
            },
            "robots": {"mock0": {"type": "mock"}},
        }
    )
    service = RobotAgentService(config, agent_id="main", episode_dir=tmp_path)
    fake_bus = FakeBus()
    service.bus = fake_bus  # type: ignore[assignment]
    service.events = BusEventPublisher(fake_bus, service.topics)  # type: ignore[arg-type]
    stale = service.robot_states.ensure("s1", agent_id="main", robot_id="mock0")
    stale.active_task = "old failed task"
    stale.active_skill_id = "cmd_old"
    stale.active_skill_phase = "failed"
    stale.recovery_required = True
    stale.recovery_reason = "old failure"
    service.robot_states.save(stale)
    service.task_runtime.task_runs.ensure_active(
        episode_id="s1",
        task="old failed task",
        agent_id="main",
        robot_id="mock0",
    )
    old_task = service.task_runtime.task_runs.load_active("s1")
    assert old_task is not None
    old_task.status = "recovering"
    service.task_runtime.task_runs.save(old_task)
    service.task_runtime.checkpoints.enqueue_pending_turn(
        UserTurn(
            envelope=Envelope(
                trace_id="tr_pending",
                episode_id="s1",
                agent_id="main",
                robot_id="mock0",
            ),
            text="stale correction",
        ),
        reason="correction",
    )
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr_new", episode_id="s1", agent_id="main", robot_id="mock0"
        ),
        text="new clean task",
    )

    asyncio.run(service._handle_user_turn_locked(turn))

    skills = [
        payload
        for topic, payload in fake_bus.published
        if topic == service.topics.skill_intent
    ]
    state = service.robot_states.load("s1")
    active_task = service.task_runtime.task_runs.load_active("s1")

    assert skills
    assert skills[0]["objective"] == "new clean task"
    assert state is not None
    assert state.recovery_required is False
    assert state.active_skill_id is None
    assert service.task_runtime.checkpoints.load("s1") is not None
    assert service.task_runtime.checkpoints.load("s1").pending_turns == []  # type: ignore[union-attr]
    assert active_task is not None
    assert active_task.root_task == "new clean task"


def test_robot_agent_service_replayed_pending_turn_keeps_remaining_queue(
    tmp_path,
) -> None:
    class FakeBus:
        def __init__(self) -> None:
            self.published: list[tuple[str, dict]] = []

        async def publish(self, topic: str, payload: dict) -> None:
            self.published.append((topic, payload))

    config = DeploymentConfig.from_dict(
        {
            "agents": {
                "main": {
                    "type": "robot_agent",
                    "robot_id": "mock0",
                    "settings": {"mode": "direct"},
                }
            },
            "robots": {"mock0": {"type": "mock"}},
        }
    )
    service = RobotAgentService(config, agent_id="main", episode_dir=tmp_path)
    fake_bus = FakeBus()
    service.bus = fake_bus  # type: ignore[assignment]
    service.events = BusEventPublisher(fake_bus, service.topics)  # type: ignore[arg-type]
    service.task_runtime.checkpoints.enqueue_pending_turn(
        UserTurn(
            envelope=Envelope(
                trace_id="tr_pending_1",
                episode_id="s1",
                agent_id="main",
                robot_id="mock0",
            ),
            text="first queued update",
        ),
        reason="follow_up",
    )
    service.task_runtime.checkpoints.enqueue_pending_turn(
        UserTurn(
            envelope=Envelope(
                trace_id="tr_pending_2",
                episode_id="s1",
                agent_id="main",
                robot_id="mock0",
            ),
            text="second queued update",
        ),
        reason="follow_up",
    )
    asyncio.run(service._continue_pending_turn_for_episode("s1"))

    checkpoint = service.task_runtime.checkpoints.load("s1")
    assert checkpoint is not None
    assert [item["text"] for item in checkpoint.pending_turns] == [
        "second queued update"
    ]


def test_robot_agent_service_interrupt_publishes_interrupt_skill(tmp_path) -> None:
    class FakeBus:
        def __init__(self) -> None:
            self.published: list[tuple[str, dict]] = []

        async def publish(self, topic: str, payload: dict) -> None:
            self.published.append((topic, payload))

    config = DeploymentConfig.from_dict(
        {
            "agents": {
                "main": {
                    "type": "robot_agent",
                    "robot_id": "mock0",
                    "settings": {"mode": "direct"},
                }
            },
            "robots": {"mock0": {"type": "mock"}},
        }
    )
    service = RobotAgentService(config, agent_id="main", episode_dir=tmp_path)
    fake_bus = FakeBus()
    service.bus = fake_bus  # type: ignore[assignment]
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr1", episode_id="s1", agent_id="main", robot_id="mock0"
        ),
        text="stop now",
    )

    handled = asyncio.run(service._handle_busy_turn(turn, active_skill_id="cmd1"))
    skills = [
        payload
        for topic, payload in fake_bus.published
        if topic == service.topics.skill_intent
    ]

    assert handled is True
    assert skills
    assert skills[0]["skill_id"] == "cmd1"
    assert skills[0]["metadata"]["mode"] == "interrupt"
    assert skills[0]["interrupt"] is True


def test_robot_agent_service_confirmed_pending_confirmation_restores_objective(
    tmp_path,
) -> None:

    class FakeBus:
        def __init__(self) -> None:
            self.published: list[tuple[str, dict]] = []

        async def publish(self, topic: str, payload: dict) -> None:
            self.published.append((topic, payload))

    config = DeploymentConfig.from_dict(
        {
            "agents": {
                "main": {
                    "type": "robot_agent",
                    "robot_id": "mock0",
                    "settings": {"mode": "direct"},
                }
            },
            "robots": {"mock0": {"type": "mock"}},
        }
    )
    service = RobotAgentService(config, agent_id="main", episode_dir=tmp_path)
    fake_bus = FakeBus()
    service.bus = fake_bus  # type: ignore[assignment]
    service.events = BusEventPublisher(fake_bus, service.topics)  # type: ignore[arg-type]
    service.task_runtime.store_pending_confirmation(
        "s1",
        {
            "proposal_id": "proposal_1",
            "capability": "turn_base",
            "objective": "向左转一点，然后看看左侧是什么",
            "prompt": "要向左转一点，然后看看左侧是什么吗？",
        },
    )
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr1", episode_id="s1", agent_id="main", robot_id="mock0"
        ),
        text="好",
    )

    asyncio.run(service._handle_user_turn_locked(turn))

    task = service.task_runtime.task_runs.load_active("s1")
    checkpoint = service.task_runtime.pending_confirmation("s1")
    skills = [
        payload
        for topic, payload in fake_bus.published
        if topic == service.topics.skill_intent
    ]

    assert task is not None
    expected_objective = "\u5411\u5de6\u8f6c\u4e00\u70b9\uff0c\u7136\u540e\u770b\u770b\u5de6\u4fa7\u662f\u4ec0\u4e48"
    assert task.root_task == expected_objective
    assert checkpoint is None
    assert skills
    assert skills[0]["objective"] == expected_objective


def test_robot_agent_service_declined_pending_confirmation_does_not_create_task(
    tmp_path,
) -> None:

    class FakeBus:
        def __init__(self) -> None:
            self.published: list[tuple[str, dict]] = []

        async def publish(self, topic: str, payload: dict) -> None:
            self.published.append((topic, payload))

    config = DeploymentConfig.from_dict(
        {
            "agents": {
                "main": {
                    "type": "robot_agent",
                    "robot_id": "mock0",
                    "settings": {"mode": "direct"},
                }
            },
            "robots": {"mock0": {"type": "mock"}},
        }
    )
    service = RobotAgentService(config, agent_id="main", episode_dir=tmp_path)
    fake_bus = FakeBus()
    service.bus = fake_bus  # type: ignore[assignment]
    service.events = BusEventPublisher(fake_bus, service.topics)  # type: ignore[arg-type]
    service.task_runtime.store_pending_confirmation(
        "s1",
        {
            "proposal_id": "proposal_2",
            "capability": "turn_base",
            "objective": "\u5411\u5de6\u8f6c\u4e00\u70b9\uff0c\u7136\u540e\u770b\u770b\u5de6\u4fa7\u662f\u4ec0\u4e48",
            "prompt": "要向左转一点，然后看看左侧是什么吗？",
        },
    )
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr2", episode_id="s1", agent_id="main", robot_id="mock0"
        ),
        text="不要",
    )

    asyncio.run(service._handle_user_turn_locked(turn))

    replies = [
        payload
        for topic, payload in fake_bus.published
        if topic == service.topics.agent_reply
    ]

    task = service.task_runtime.task_runs.load_active("s1")
    assert task is not None
    expected_objective = "\u5411\u5de6\u8f6c\u4e00\u70b9\uff0c\u7136\u540e\u770b\u770b\u5de6\u4fa7\u662f\u4ec0\u4e48"
    assert task.root_task == expected_objective
    assert service.task_runtime.pending_confirmation("s1") is None
    assert replies
    assert replies[-1]["metadata"]["pending_confirmation_declined"] is True


def test_robot_agent_service_new_task_reply_clears_pending_confirmation_and_runs_new_objective(
    tmp_path,
) -> None:

    class FakeBus:
        def __init__(self) -> None:
            self.published: list[tuple[str, dict]] = []

        async def publish(self, topic: str, payload: dict) -> None:
            self.published.append((topic, payload))

    config = DeploymentConfig.from_dict(
        {
            "agents": {
                "main": {
                    "type": "robot_agent",
                    "robot_id": "mock0",
                    "settings": {"mode": "direct"},
                }
            },
            "robots": {"mock0": {"type": "mock"}},
        }
    )
    service = RobotAgentService(config, agent_id="main", episode_dir=tmp_path)
    fake_bus = FakeBus()
    service.bus = fake_bus  # type: ignore[assignment]
    service.events = BusEventPublisher(fake_bus, service.topics)  # type: ignore[arg-type]
    service.task_runtime.store_pending_confirmation(
        "s1",
        {
            "proposal_id": "proposal_3",
            "capability": "turn_base",
            "objective": "向左转一点，然后看看左侧是什么",
            "prompt": "要向左转一点，然后看看左侧是什么吗？",
        },
    )
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr3", episode_id="s1", agent_id="main", robot_id="mock0"
        ),
        text="去前面看看桌子上有什么",
    )

    asyncio.run(service._handle_user_turn_locked(turn))

    task = service.task_runtime.task_runs.load_active("s1")
    skills = [
        payload
        for topic, payload in fake_bus.published
        if topic == service.topics.skill_intent
    ]

    assert service.task_runtime.pending_confirmation("s1") is None
    assert task is not None
    assert task.root_task == "去前面看看桌子上有什么"
    assert skills
    assert skills[0]["objective"] == "去前面看看桌子上有什么"
