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


def _gateway(tmp_path) -> GatewayService:
    config = DeploymentConfig.from_dict(
        {
            "deployment": {"id": "d1"},
            "resources": {
                "runtime_dir": str(tmp_path / "runtime"),
                "episodes": {"root": str(tmp_path / "episodes")},
            },
            "identity": {
                "enabled": True,
                "unified_user_episodes": True,
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


def test_gateway_forwards_user_turn_and_persists_episode(tmp_path) -> None:
    gateway = _gateway(tmp_path)
    turn = UserTurn(
        envelope=Envelope(
            channel="web", chat_id="chat-1", chat_type="web", sender_id="u1"
        ),
        text="pick up the cup",
    )

    asyncio.run(gateway._on_user_turn(turn))

    fake_bus = cast(FakeBus, gateway.bus)
    published_topics = [topic for topic, _payload in fake_bus.published]
    forwarded_payload = next(
        payload
        for topic, payload in fake_bus.published
        if topic == gateway.topics.user_turn
    )
    history = gateway.episodes.history(forwarded_payload["envelope"]["episode_id"])
    stored_events = gateway.event_store.recent(10)

    assert gateway.topics.user_turn in published_topics
    assert gateway.topics.runtime_event in published_topics
    assert forwarded_payload["envelope"]["agent_id"] == "main"
    assert forwarded_payload["envelope"]["robot_id"] == "mock0"
    assert forwarded_payload["envelope"]["user_id"] is None
    assert forwarded_payload["envelope"]["episode_id"] is not None
    assert history[-1].role == "user"
    assert {event["kind"] for event in stored_events} >= {"episode.allocated"}
    interaction = gateway.interaction_states.get(
        forwarded_payload["envelope"]["episode_id"]
    )
    assert interaction is not None
    assert interaction.active_channel == "web"
    assert interaction.last_user_intent == "new_task"


def test_gateway_deduplicates_own_runtime_event_echo(tmp_path) -> None:
    gateway = _gateway(tmp_path)
    turn = UserTurn(
        envelope=Envelope(channel="web", chat_id="chat-1", chat_type="web"),
        text="inspect",
    )

    asyncio.run(gateway._on_user_turn(turn))
    event = next(
        item
        for item in gateway.event_store.recent(10)
        if item["kind"] == EventKind.EPISODE_ALLOCATED
    )
    asyncio.run(gateway._on_runtime_event(gateway.topics.runtime_event, event))

    stored = [
        item
        for item in gateway.event_store.recent(20)
        if item.get("event_id") == event["event_id"]
    ]
    assert len(stored) == 1


def test_gateway_maps_bound_sender_to_internal_user_and_unifies_episodes(
    tmp_path,
) -> None:
    gateway = _gateway(tmp_path)
    web_turn = UserTurn(
        envelope=Envelope(
            channel="web",
            account_id="web",
            chat_id="chat-web",
            chat_type="web",
            sender_id="web-user",
        ),
        text="inspect table",
    )
    voice_turn = UserTurn(
        envelope=Envelope(
            channel="voice",
            account_id="voice",
            chat_id="chat-voice",
            chat_type="voice",
            sender_id="voice-user",
        ),
        text="continue",
    )

    asyncio.run(gateway._on_user_turn(web_turn))
    asyncio.run(gateway._on_user_turn(voice_turn))

    fake_bus = cast(FakeBus, gateway.bus)
    user_turn_payloads = [
        payload
        for topic, payload in fake_bus.published
        if topic == gateway.topics.user_turn
    ]

    assert len(user_turn_payloads) == 2
    assert user_turn_payloads[0]["envelope"]["user_id"] == "owner"
    assert user_turn_payloads[1]["envelope"]["user_id"] == "owner"
    assert (
        user_turn_payloads[0]["envelope"]["episode_id"]
        == user_turn_payloads[1]["envelope"]["episode_id"]
    )


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


def test_gateway_web_cockpit_exposes_task_session_view(tmp_path) -> None:
    gateway = _gateway(tmp_path)
    envelope = Envelope(
        trace_id="tr1",
        episode_id="ep1",
        channel="web",
        robot_id="mock0",
        agent_id="main",
    )
    gateway.task_runs.ensure_active(
        episode_id="ep1",
        task="follow me",
        agent_id="main",
        robot_id="mock0",
    )
    gateway.task_runs.bind_skill("ep1", "skill1", "follow me")
    gateway.skill_store.append(
        SkillEvent(
            envelope=envelope,
            skill_id="skill1",
            name="human_follow",
            phase="executing",
            progress=0.4,
            summary="following target",
            step="following",
            metadata={
                "ux": {
                    "skill": "human_follow",
                    "phase": "following",
                    "bbox": [10, 20, 30, 40],
                    "confidence": 0.8,
                    "frame_id": 12,
                    "camera": "front",
                    "command": {"vx": 0.1, "vy": 0.0, "wz": 0.2},
                }
            },
        )
    )

    payload = asyncio.run(gateway._web_cockpit("ep1"))

    assert payload is not None
    assert payload["health"]["robot_id"] == "mock0"
    view = payload["view"]
    assert view["root_task"] == "follow me"
    assert view["active_skill_id"] == "skill1"
    assert view["active_skill_name"] == "human_follow"
    assert view["current_phase"] == "following"
    assert view["skill_progress"] == 0.4
    assert view["skill_ux"]["bbox"] == [10, 20, 30, 40]
    assert view["latest_evidence"]["skill_ux"]["confidence"] == 0.8
    assert view["interaction_state"] is None
    assert any(item["kind"] == "skill_progress" for item in view["timeline"])
    assert "stop_motion" in view["operator_actions"]
    assert asyncio.run(gateway._web_cockpit("missing")) is None


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
                    envelope=skill_event.envelope, skill_id="cmd1", status="completed"
                )
            ),
        )
    )

    state = gateway.robot_states.load("ep1")

    assert web is not None
    assert any(item["kind"] == "robot.status" for item in web._events)  # type: ignore[attr-defined]
    assert any(item["kind"] == "skill.lifecycle" for item in web._events)  # type: ignore[attr-defined]
    assert gateway.skill_store.get("cmd1") is not None
    assert state is not None
    assert state.active_skill_id == "cmd1"


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
        state="skill_completed",
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
    gateway.robot_states.ensure("ep1", agent_id="main", robot_id="mock0")
    gateway.robot_states.update_status(
        "ep1",
        RobotStatus(
            envelope=Envelope(robot_id="mock0"),
            frame_id=42,
            state="idle",
            success=True,
            metrics={"battery": {"percentage": 85}},
        ),
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
