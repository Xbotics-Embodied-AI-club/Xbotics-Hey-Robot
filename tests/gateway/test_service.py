from __future__ import annotations

import asyncio
from typing import cast

import pytest

from hey_robot.config import DeploymentConfig
from hey_robot.episode.scope import EpisodeScope
from hey_robot.events import EventKind, RuntimeEvent
from hey_robot.events.bus import BusEventPublisher
from hey_robot.gateway import GatewayService
from hey_robot.protocol import (
    AgentReply,
    Envelope,
    RobotStatus,
    SkillEvent,
    SkillResult,
    UserTurn,
)
from hey_robot.protocol.messages import to_payload


class FakeBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict]] = []
        self.subscriptions: list[list[str]] = []
        self.connected = False
        self.closed = False

    async def connect(self) -> None:
        self.connected = True

    async def subscribe(self, topics, _handler) -> None:
        self.subscriptions.append(list(topics))

    async def publish(self, topic: str, payload: dict) -> None:
        self.published.append((topic, payload))

    async def close(self) -> None:
        self.closed = True


class FakeChannels:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self._items = [("web", object())]

    def items(self):
        return list(self._items)

    async def start_all(self, _handler) -> None:
        self.started = True

    async def stop_all(self) -> None:
        self.stopped = True


def _gateway(tmp_path, *, unified_user_episodes: bool = True) -> GatewayService:
    config = DeploymentConfig.from_dict(
        {
            "deployment": {"id": "d1"},
            "resources": {
                "runtime_dir": str(tmp_path / "runtime"),
                "episodes": {"root": str(tmp_path / "episodes")},
            },
            "identity": {
                "enabled": True,
                "unified_user_episodes": unified_user_episodes,
                "bindings": {
                    "web:sender:web-user": "owner",
                    "voice:sender:voice-user": "owner",
                },
            },
            "robots": {"mock0": {"type": "mock"}},
            "agents": {"main": {"type": "robot_agent", "robot_id": "mock0"}},
            "channels": {"web": {"type": "web", "enabled": True}},
        }
    )
    gateway = GatewayService(config, episode_dir=tmp_path / "episodes")
    gateway.bus = cast(object, FakeBus())  # type: ignore[assignment]
    gateway.events = BusEventPublisher(gateway.bus, gateway.topics)
    return gateway


def test_gateway_publishes_goal_prefixed_text_as_ordinary_turn(
    tmp_path,
) -> None:
    gateway = _gateway(tmp_path)
    turn = UserTurn(
        envelope=Envelope(
            channel="web", chat_id="chat-1", chat_type="web", sender_id="u1"
        ),
        text="pick up the cup",
    )

    gateway.episodes.ensure(
        "ep-owner", EpisodeScope(dimensions=("user",), values={"user": "owner"}), ()
    )
    gateway.episodes.append_user_turn("ep-owner", turn)
    asyncio.run(gateway._on_user_turn(turn))

    fake_bus = cast(FakeBus, gateway.bus)
    assert all(topic != gateway.topics.user_turn for topic, _ in fake_bus.published)
    conversation = next(
        payload
        for topic, payload in fake_bus.published
        if topic == gateway.topics.conversation_turn
    )
    assert conversation["text"] == "pick up the cup"
    assert conversation["session_key"] == "d1:main:web:chat-1"

    create = UserTurn(
        envelope=turn.envelope,
        text="""/goal create {"objective":"inspect","success_criteria":[{"criterion_id":"seen","criterion_type":"evidence_present","subject_id":"room:front","predicate":"observed","object_id":"scene","max_age_sec":20}]}""",
    )
    asyncio.run(gateway._on_user_turn(create))
    turns = [
        payload
        for topic, payload in fake_bus.published
        if topic == gateway.topics.conversation_turn
    ]
    assert [item["text"] for item in turns] == ["pick up the cup", create.text]


def test_gateway_never_routes_natural_language_directly_to_skill_intent(
    tmp_path,
) -> None:
    gateway = _gateway(tmp_path)
    asyncio.run(
        gateway._on_user_turn(
            UserTurn(Envelope(channel="web", sender_id="u1"), "look around")
        )
    )
    fake_bus = cast(FakeBus, gateway.bus)
    assert any(
        topic == gateway.topics.conversation_turn for topic, _ in fake_bus.published
    )
    assert all(topic != gateway.topics.skill_intent for topic, _ in fake_bus.published)


def test_non_unified_identity_uses_chat_scoped_session_key(tmp_path) -> None:
    gateway = _gateway(tmp_path, unified_user_episodes=False)
    envelope = Envelope(
        channel="web",
        chat_id="trial-2",
        sender_id="web-user",
        user_id="owner",
        agent_id="main",
    )

    assert gateway._session_key(envelope) == "d1:main:web:trial-2"


def test_gateway_routes_turns_even_when_task_store_has_active_task(tmp_path) -> None:
    gateway = _gateway(tmp_path)
    gateway.task_store.create_task(
        session_key="d1:main:web:u1",
        envelope=Envelope(robot_id="mock0"),
        objective="current task",
    )

    asyncio.run(
        gateway._on_user_turn(
            UserTurn(Envelope(channel="web", sender_id="u1"), "actually inspect here")
        )
    )

    fake_bus = cast(FakeBus, gateway.bus)
    assert any(
        topic == gateway.topics.conversation_turn for topic, _ in fake_bus.published
    )


def test_gateway_deduplicates_replayed_transport_message_before_model_routing(
    tmp_path,
) -> None:
    gateway = _gateway(tmp_path)
    turn = UserTurn(
        Envelope(channel="web", sender_id="u1", message_id="replayed-message"),
        "look around",
    )

    asyncio.run(gateway._on_user_turn(turn))
    asyncio.run(gateway._on_user_turn(turn))

    fake_bus = cast(FakeBus, gateway.bus)
    turns = [
        payload
        for topic, payload in fake_bus.published
        if topic == gateway.topics.conversation_turn
    ]
    assert len(turns) == 1


def test_gateway_routes_natural_language_emergency_stop_without_provider(
    tmp_path,
) -> None:
    gateway = _gateway(tmp_path)

    asyncio.run(
        gateway._on_user_turn(
            UserTurn(
                Envelope(channel="web", sender_id="u1", message_id="stop-1"),
                "emergency stop",
            )
        )
    )

    fake_bus = cast(FakeBus, gateway.bus)
    commands = [
        payload
        for topic, payload in fake_bus.published
        if topic == gateway.topics.skill_control
    ]
    assert len(commands) == 1
    assert commands[0]["action"] == "emergency_stop"


def test_gateway_deduplicates_own_runtime_event_echo(tmp_path) -> None:
    gateway = _gateway(tmp_path)
    event = RuntimeEvent.make("goal.created", source="gateway", robot_id="mock0")
    gateway.event_store.append(event)
    asyncio.run(
        gateway._on_runtime_event(gateway.topics.runtime_event, event.to_dict())
    )

    stored = [
        item
        for item in gateway.event_store.recent(20)
        if item.get("event_id") == event.event_id
    ]
    assert len(stored) == 1


def test_gateway_web_history_uses_user_identity_scope(tmp_path) -> None:
    gateway = _gateway(tmp_path)
    turn = UserTurn(
        envelope=Envelope(
            channel="web",
            account_id="web",
            chat_id="chat-web",
            chat_type="web",
            sender_id="web-user",
        ),
        text="remember this task",
    )

    asyncio.run(gateway._on_user_turn(turn))
    history = asyncio.run(
        gateway._web_history(
            Envelope(
                channel="web",
                account_id="web",
                sender_id="web-user",
                chat_id="other-web-chat",
            ),
            20,
        )
    )

    assert history["user_id"] == "owner"
    assert history["continuity"]["user_id"] == "owner"
    assert history["continuity"]["shared_episode_scope"] is True
    assert "voice" in history["continuity"]["linked_channels"]
    assert "web" in history["continuity"]["linked_channels"]
    assert history["records"][-1]["content"] == "remember this task"


def test_gateway_web_cockpit_exposes_sustained_task_view(tmp_path) -> None:
    gateway = _gateway(tmp_path)
    task = gateway.task_store.create_task(
        session_key="session",
        envelope=Envelope(robot_id="mock0"),
        objective="follow me",
    )

    payload = asyncio.run(gateway._web_cockpit("ep1"))

    assert payload is not None
    assert payload["health"]["robot_id"] == "mock0"
    assert payload["task"]["task_id"] == task.task_id
    assert payload["task"]["objective"] == "follow me"


def test_gateway_identity_binding_links_web_and_feishu_without_forwarding_task(
    tmp_path,
) -> None:
    gateway = _gateway(tmp_path)
    replies: list[AgentReply] = []

    async def capture(reply: AgentReply) -> None:
        replies.append(reply)

    gateway.channels.send = capture  # type: ignore[method-assign]

    created = asyncio.run(
        gateway.create_identity_binding(
            Envelope(
                channel="web",
                account_id="web",
                chat_id="chat-web",
                chat_type="web",
                sender_id="web-new",
            ),
            ttl_sec=300.0,
        )
    )

    asyncio.run(
        gateway._on_user_turn(
            UserTurn(
                envelope=Envelope(
                    channel="feishu",
                    account_id="feishu",
                    chat_id="oc_chat_1",
                    chat_type="group",
                    sender_id="ou_user_1",
                    message_id="om_1",
                ),
                text=f"绑定 {created['code']}",
            )
        )
    )

    fake_bus = cast(FakeBus, gateway.bus)
    forwarded_payloads = [
        payload
        for topic, payload in fake_bus.published
        if topic == gateway.topics.user_turn
    ]
    status = asyncio.run(gateway.identity_binding_status(created["code"]))
    resolved = gateway.identity.resolve(
        Envelope(
            channel="feishu",
            account_id="feishu",
            chat_id="oc_chat_1",
            sender_id="ou_user_1",
        )
    )
    state_path = tmp_path / "runtime" / "identity" / "bindings.json"
    web = gateway.channels.get("web")

    assert forwarded_payloads == []
    assert status["status"] == "claimed"
    assert status["user_id"] == created["user_id"]
    assert status["continuity"]["user_id"] == created["user_id"]
    assert "feishu" in status["continuity"]["linked_channels"]
    assert "web" in status["continuity"]["linked_channels"]
    assert resolved.user_id == created["user_id"]
    assert state_path.exists()
    assert web is not None
    assert replies[-1].metadata["identity_binding"] is True
    assert replies[-1].envelope.channel == "feishu"


def test_gateway_identity_binding_rejects_invalid_code(tmp_path) -> None:
    gateway = _gateway(tmp_path)
    replies: list[AgentReply] = []

    async def capture(reply: AgentReply) -> None:
        replies.append(reply)

    gateway.channels.send = capture  # type: ignore[method-assign]

    asyncio.run(
        gateway._on_user_turn(
            UserTurn(
                envelope=Envelope(
                    channel="feishu",
                    chat_id="oc_chat_1",
                    chat_type="group",
                    sender_id="ou_user_1",
                ),
                text="bind BAD999",
            )
        )
    )

    fake_bus = cast(FakeBus, gateway.bus)
    user_turn_payloads = [
        payload
        for topic, payload in fake_bus.published
        if topic == gateway.topics.user_turn
    ]

    assert user_turn_payloads == []
    assert "无效" in replies[-1].text


def test_gateway_forwards_agent_reply_back_to_web_channel(tmp_path) -> None:
    gateway = _gateway(tmp_path)
    envelope = Envelope(
        trace_id="tr1",
        episode_id="ep1",
        channel="web",
        chat_id="chat-1",
        chat_type="web",
        sender_id="u1",
        robot_id="mock0",
        agent_id="main",
    )
    gateway.episodes.ensure("ep1", scope=EpisodeScope(agent_id="main"), aliases=[])
    reply = AgentReply(envelope=envelope, text="done")

    asyncio.run(gateway._on_agent_reply(gateway.topics.agent_reply, to_payload(reply)))

    web = gateway.channels.get("web")
    history = gateway.episodes.history("ep1")

    assert web is not None
    assert web._replies[-1]["text"] == "done"  # type: ignore[attr-defined]
    assert history[-1].role == "assistant"


def test_gateway_allocates_episode_for_proactive_agent_reply(tmp_path) -> None:
    gateway = _gateway(tmp_path)
    reply = AgentReply(
        envelope=Envelope(
            trace_id="tr_notify",
            channel="web",
            chat_id="chat-42",
            chat_type="web",
            sender_id="u42",
            robot_id="mock0",
            agent_id="main",
        ),
        text="proactive check-in",
        metadata={"proactive": True},
    )

    asyncio.run(gateway._on_agent_reply(gateway.topics.agent_reply, to_payload(reply)))

    web = gateway.channels.get("web")
    assert web is not None
    stored = web._replies[-1]  # type: ignore[attr-defined]
    episode_id = stored["envelope"]["episode_id"]
    history = gateway.episodes.history(episode_id)

    assert episode_id is not None
    assert history[-1].role == "assistant"
    assert history[-1].content == "proactive check-in"


def test_gateway_publishes_runtime_and_skill_events_to_channels(tmp_path) -> None:
    gateway = _gateway(tmp_path)
    web = gateway.channels.get("web")
    event = RuntimeEvent.make(
        EventKind.ROBOT_STATUS,
        source="robot",
        robot_id="mock0",
        payload={"state": "idle"},
    )
    skill_event = SkillEvent(
        envelope=Envelope(
            trace_id="tr1",
            episode_id="ep1",
            channel="web",
            robot_id="mock0",
            agent_id="main",
        ),
        skill_id="cmd1",
        phase="executing",
        summary="moving to cup",
    )

    asyncio.run(
        gateway._on_runtime_event(gateway.topics.runtime_event, event.to_dict())
    )
    asyncio.run(
        gateway._on_skill_event(gateway.topics.skill_event, to_payload(skill_event))
    )
    asyncio.run(
        gateway._on_skill_result(
            gateway.topics.skill_result,
            to_payload(
                SkillResult(
                    envelope=skill_event.envelope,
                    skill_id="cmd1",
                    status="completed",
                    success=True,
                )
            ),
        )
    )

    assert web is not None
    assert any(item["kind"] == "robot.status" for item in web._events)  # type: ignore[attr-defined]
    assert any(item["kind"] == "skill.lifecycle" for item in web._events)  # type: ignore[attr-defined]
    assert gateway.skill_store.get("cmd1") is not None


def test_gateway_compacts_robot_status_motion_trace_for_event_stream(tmp_path) -> None:
    gateway = _gateway(tmp_path)
    status = RobotStatus(
        envelope=Envelope(
            trace_id="tr1",
            episode_id="ep1",
            channel="web",
            robot_id="mock0",
            agent_id="main",
        ),
        frame_id=7,
        state="idle",
        metrics={
            "last_skill_result": {
                "success": True,
                "motion_trace": {
                    "kind": "pulse_velocity",
                    "duration_sec": 3.0,
                    "iterations": [
                        {
                            "index": 1,
                            "elapsed_sec": 0.1,
                            "success": True,
                            "control": {"wheel_writes": [1, 2, 3]},
                        },
                        {
                            "index": 2,
                            "elapsed_sec": 0.2,
                            "success": True,
                            "control": {"wheel_writes": [4, 5, 6]},
                        },
                    ],
                },
            },
            "base_control": {
                "last_motion_report": {
                    "kind": "pulse_velocity",
                    "iterations": [{"index": 1, "elapsed_sec": 0.1, "success": True}],
                }
            },
        },
    )

    asyncio.run(
        gateway._on_robot_status(gateway.topics.robot_status, to_payload(status))
    )

    web = gateway.channels.get("web")
    assert web is not None
    event = web._events[-1]  # type: ignore[attr-defined]
    metrics = event["payload"]["metrics"]

    assert "iterations" not in metrics["last_skill_result"]["motion_trace"]
    assert metrics["last_skill_result"]["motion_trace"]["iteration_count"] == 2
    assert (
        "control" not in metrics["last_skill_result"]["motion_trace"]["first_iteration"]
    )
    assert "iterations" not in metrics["base_control"]["last_motion_report"]


def test_gateway_throttles_robot_status_disk_persistence(tmp_path) -> None:
    gateway = _gateway(tmp_path)
    status = RobotStatus(
        envelope=Envelope(robot_id="mock0"),
        frame_id=1,
        state="idle",
    )
    asyncio.run(
        gateway._on_robot_status(gateway.topics.robot_status, to_payload(status))
    )
    asyncio.run(
        gateway._on_robot_status(gateway.topics.robot_status, to_payload(status))
    )

    assert gateway.event_store.count() == 1
    web = gateway.channels.get("web")
    assert web is not None
    assert len(web._events) == 2  # type: ignore[attr-defined]


def test_gateway_runtime_summary_normalizes_robot_and_event_content(tmp_path) -> None:
    gateway = _gateway(tmp_path)
    asyncio.run(
        gateway._on_robot_status(
            gateway.topics.robot_status,
            to_payload(
                RobotStatus(
                    envelope=Envelope(robot_id="mock0"),
                    frame_id=42,
                    state="idle",
                    success=True,
                    metrics={"battery": {"percentage": 85}},
                )
            ),
        )
    )
    gateway.event_store.append(
        RuntimeEvent.make(
            EventKind.ROBOT_STATUS,
            source="robot",
            payload={"state": "idle", "frame_id": 42},
        )
    )

    summary = asyncio.run(gateway._web_runtime_summary(10))

    assert summary["robots"][0]["state"] == "idle"
    assert summary["robots"][0]["status"]["frame_id"] == 42
    assert summary["events"][0]["summary"] == "state=idle, frame=42"


def test_gateway_ignores_invalid_runtime_event_and_rejects_unknown_channel_type(
    tmp_path,
) -> None:
    gateway = _gateway(tmp_path)
    web = gateway.channels.get("web")

    asyncio.run(
        gateway._on_runtime_event(gateway.topics.runtime_event, {"bad": "payload"})
    )

    assert web is not None
    assert web._events == []  # type: ignore[attr-defined]
    assert gateway.event_store.recent(5) == []

    config = DeploymentConfig.from_dict(
        {
            "resources": {"episodes": {"root": str(tmp_path / "episodes2")}},
            "channels": {"bad": {"type": "unknown", "enabled": True}},
        }
    )

    with pytest.raises(ValueError, match="unsupported channel type"):
        GatewayService(config, episode_dir=tmp_path / "episodes2")


def test_gateway_start_and_stop_publish_lifecycle_and_manage_channels(
    tmp_path, monkeypatch
) -> None:
    gateway = _gateway(tmp_path)
    fake_bus = cast(FakeBus, gateway.bus)
    fake_channels = FakeChannels()
    gateway.channels = fake_channels  # type: ignore[assignment]

    class StopLoopError(Exception):
        pass

    class OneShotEvent:
        def set(self) -> None:
            return None

        async def wait(self) -> None:
            raise StopLoopError()

    monkeypatch.setattr("hey_robot.gateway.service.asyncio.Event", OneShotEvent)

    with pytest.raises(StopLoopError):
        asyncio.run(gateway.start())

    stored = gateway.event_store.recent(10)

    assert fake_bus.connected is True
    assert fake_channels.started is True
    assert [topic for topic, _payload in fake_bus.published[:2]] == [
        gateway.topics.runtime_event,
        gateway.topics.runtime_event,
    ]
    assert fake_bus.subscriptions == [
        [gateway.topics.agent_reply],
        [gateway.topics.conversation_result],
        [gateway.topics.runtime_event],
        [gateway.topics.robot_status],
        [gateway.topics.skill_event],
        [gateway.topics.skill_result],
    ]
    assert {event["kind"] for event in stored} >= {"gateway.start", "gateway.ready"}

    asyncio.run(gateway.stop())

    stopped = gateway.event_store.recent(10)
    assert fake_channels.stopped is True
    assert fake_bus.closed is True
    assert any(event["kind"] == "gateway.shutdown" for event in stopped)


def test_gateway_routes_natural_confirmation_to_agent(tmp_path) -> None:
    gateway = _gateway(tmp_path)
    asyncio.run(
        gateway._on_user_turn(
            UserTurn(
                Envelope(channel="web", sender_id="web-user", message_id="confirm-msg"),
                "confirm",
            )
        )
    )
    fake_bus = cast(FakeBus, gateway.bus)
    confirmations = [
        payload
        for topic, payload in fake_bus.published
        if topic == gateway.topics.conversation_turn and payload["text"] == "confirm"
    ]
    assert len(confirmations) == 1
