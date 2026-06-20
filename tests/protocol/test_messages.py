from __future__ import annotations

import pytest

from hey_robot.protocol.messages import (
    AgentReply,
    ArtifactRef,
    Envelope,
    ImageRef,
    MediaRef,
    RobotAction,
    RobotObservation,
    RobotStatus,
    SkillEvent,
    SkillIntent,
    SkillResult,
    UserTurn,
    _new_id,
    from_payload,
    to_payload,
)
from hey_robot.protocol.topics import Topics


class TestEnvelope:
    def test_default_trace_id(self) -> None:
        env = Envelope()
        assert env.trace_id.startswith("tr_")
        assert len(env.trace_id) > 3

    def test_explicit_trace_id(self) -> None:
        env = Envelope(trace_id="custom")
        assert env.trace_id == "custom"

    def test_child_preserves_trace_id(self) -> None:
        env = Envelope(trace_id="tr-parent", episode_id="ep1")
        child = env.child(episode_id="ep2")
        assert child.trace_id == "tr-parent"
        assert child.episode_id == "ep2"

    def test_child_creates_trace_id_when_empty_string(self) -> None:
        env = Envelope(trace_id="tr-parent")
        child = env.child(trace_id="")
        assert child.trace_id != ""
        assert child.trace_id != "tr-parent"
        assert child.trace_id.startswith("tr_")

    def test_child_creates_trace_id_when_none_in_updates(self) -> None:
        env = Envelope(trace_id="tr-parent")
        child = env.child()  # No trace_id in updates
        assert child.trace_id == "tr-parent"

    def test_child_merges_multiple_fields(self) -> None:
        env = Envelope(trace_id="tr1", channel="slack", robot_id="r1")
        child = env.child(channel="feishu", agent_id="main")
        assert child.trace_id == "tr1"
        assert child.channel == "feishu"
        assert child.robot_id == "r1"
        assert child.agent_id == "main"

    def test_child_preserves_and_updates_user_id(self) -> None:
        env = Envelope(trace_id="tr1", user_id="user_a", sender_id="sender_a")
        child = env.child(channel="web")
        updated = child.child(user_id="user_b")
        assert child.user_id == "user_a"
        assert updated.user_id == "user_b"

    def test_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        env = Envelope(trace_id="tr1")
        with pytest.raises(FrozenInstanceError):
            env.trace_id = "other"  # type: ignore[misc]


class TestMediaRef:
    def test_minimal_creation(self) -> None:
        ref = MediaRef(uri="http://x.com/img.jpg", media_type="image")
        assert ref.uri == "http://x.com/img.jpg"
        assert ref.media_type == "image"

    def test_with_optional_fields(self) -> None:
        ref = MediaRef(
            uri="http://x.com/video.mp4",
            media_type="video",
            name="demo",
            size_bytes=1024,
        )
        assert ref.name == "demo"
        assert ref.size_bytes == 1024

    def test_to_payload_round_trip(self) -> None:
        ref = MediaRef(uri="http://x.com/img.jpg", media_type="image", name="photo")
        payload = to_payload(ref)
        restored = MediaRef(**payload)
        assert restored == ref


class TestImageRef:
    def test_minimal_creation(self) -> None:
        ref = ImageRef(uri="http://x.com/frame.jpg")
        assert ref.uri == "http://x.com/frame.jpg"

    def test_with_camera_info(self) -> None:
        ref = ImageRef(
            uri="http://x.com/frame.jpg",
            camera="cam0",
            width=640,
            height=480,
            timestamp=123.0,
        )
        assert ref.camera == "cam0"
        assert ref.width == 640
        assert ref.height == 480
        assert ref.timestamp == 123.0

    def test_to_payload_round_trip(self) -> None:
        ref = ImageRef(
            uri="http://x.com/frame.jpg", camera="cam0", width=640, height=480
        )
        payload = to_payload(ref)
        restored = ImageRef(**payload)
        assert restored == ref


class TestArtifactRef:
    def test_minimal_creation(self) -> None:
        ref = ArtifactRef(uri="data/recording.mp4", artifact_type="video")
        assert ref.uri == "data/recording.mp4"
        assert ref.artifact_type == "video"


class TestUserTurn:
    def test_creation(self) -> None:
        env = Envelope()
        turn = UserTurn(envelope=env, text="hello")
        assert turn.text == "hello"

    def test_from_payload_round_trip(self) -> None:
        env = Envelope(trace_id="tr1")
        turn = UserTurn(envelope=env, text="hello world")
        payload = to_payload(turn)
        restored = from_payload(UserTurn, payload)
        assert restored.text == "hello world"
        assert restored.envelope.trace_id == "tr1"

    def test_from_payload_with_media(self) -> None:
        env = Envelope(trace_id="tr2")
        turn = UserTurn(
            envelope=env,
            text="look at this",
            media=[MediaRef(uri="http://img", media_type="image")],
        )
        payload = to_payload(turn)
        restored = from_payload(UserTurn, payload)
        assert len(restored.media) == 1
        assert restored.media[0].uri == "http://img"


class TestAgentReply:
    def test_from_payload_round_trip(self) -> None:
        env = Envelope(trace_id="tr1")
        reply = AgentReply(envelope=env, text="got it", final=False)
        payload = to_payload(reply)
        restored = from_payload(AgentReply, payload)
        assert restored.text == "got it"
        assert restored.final is False

    def test_default_final(self) -> None:
        env = Envelope()
        reply = AgentReply(envelope=env, text="done")
        assert reply.final is True


class TestRobotObservation:
    def test_from_payload_with_images_and_artifacts(self) -> None:
        env = Envelope(trace_id="tr1")
        obs = RobotObservation(
            envelope=env,
            frame_id=42,
            images=[ImageRef(uri="http://img1", camera="cam0")],
            artifacts=[ArtifactRef(uri="http://art1", artifact_type="depth")],
            task="inspect",
        )
        payload = to_payload(obs)
        restored = from_payload(RobotObservation, payload)
        assert restored.frame_id == 42
        assert len(restored.images) == 1
        assert restored.images[0].camera == "cam0"
        assert isinstance(restored.images[0], ImageRef)
        assert len(restored.artifacts) == 1
        assert isinstance(restored.artifacts[0], ArtifactRef)
        assert restored.task == "inspect"

    def test_from_payload_empty_lists(self) -> None:
        env = Envelope()
        obs = RobotObservation(envelope=env, frame_id=0)
        payload = to_payload(obs)
        restored = from_payload(RobotObservation, payload)
        assert restored.images == []
        assert restored.artifacts == []


class TestRobotStatus:
    def test_defaults(self) -> None:
        env = Envelope()
        status = RobotStatus(envelope=env)
        assert status.state == "unknown"
        assert status.frame_id is None

    def test_from_payload_round_trip(self) -> None:
        env = Envelope(trace_id="tr1")
        status = RobotStatus(
            envelope=env, frame_id=10, state="running", skill_id="sk1", success=True
        )
        payload = to_payload(status)
        restored = from_payload(RobotStatus, payload)
        assert restored.frame_id == 10
        assert restored.state == "running"
        assert restored.skill_id == "sk1"
        assert restored.success is True


class TestSkillIntent:
    def test_default_skill_id(self) -> None:
        env = Envelope()
        intent = SkillIntent(envelope=env, name="move_arm")
        assert intent.skill_id.startswith("skill_")

    def test_from_payload_round_trip(self) -> None:
        env = Envelope(trace_id="tr1")
        intent = SkillIntent(
            envelope=env, name="move", arguments={"x": 1}, priority=5, timeout_sec=30.0
        )
        payload = to_payload(intent)
        restored = from_payload(SkillIntent, payload)
        assert restored.name == "move"
        assert restored.arguments == {"x": 1}
        assert restored.priority == 5
        assert restored.timeout_sec == 30.0
        assert isinstance(restored.envelope, Envelope)


class TestSkillEvent:
    def test_defaults(self) -> None:
        env = Envelope()
        event = SkillEvent(envelope=env, skill_id="sk1")
        assert event.phase == "created"
        assert event.name == ""

    def test_from_payload_round_trip(self) -> None:
        env = Envelope(trace_id="tr1")
        event = SkillEvent(
            envelope=env,
            skill_id="sk1",
            name="move",
            phase="executing",
            steps_executed=10,
            frame_id=5,
            progress=0.5,
        )
        payload = to_payload(event)
        restored = from_payload(SkillEvent, payload)
        assert restored.skill_id == "sk1"
        assert restored.phase == "executing"
        assert restored.steps_executed == 10
        assert restored.progress == 0.5


class TestRobotAction:
    def test_default_action_id(self) -> None:
        env = Envelope()
        action = RobotAction(envelope=env, values=[1.0, 2.0, 3.0])
        assert action.action_id.startswith("act_")

    def test_from_payload_round_trip(self) -> None:
        env = Envelope(trace_id="tr1")
        action = RobotAction(
            envelope=env,
            values=[0.5, -0.3],
            action_id="act1",
            skill_id="sk1",
            timestamp=100.0,
        )
        payload = to_payload(action)
        restored = from_payload(RobotAction, payload)
        assert restored.values == [0.5, -0.3]
        assert restored.skill_id == "sk1"


class TestSkillResult:
    def test_defaults(self) -> None:
        env = Envelope()
        result = SkillResult(envelope=env, skill_id="sk1")
        assert result.status == "unknown"
        assert result.success is None

    def test_from_payload_with_observations(self) -> None:
        env = Envelope(trace_id="tr1")
        result = SkillResult(
            envelope=env,
            skill_id="sk1",
            status="completed",
            success=True,
            steps_executed=5,
            progress=1.0,
            summary="all done",
            observations=[ImageRef(uri="http://final.jpg", camera="cam0")],
        )
        payload = to_payload(result)
        restored = from_payload(SkillResult, payload)
        assert restored.status == "completed"
        assert restored.success is True
        assert restored.steps_executed == 5
        assert len(restored.observations) == 1
        assert isinstance(restored.observations[0], ImageRef)
        assert restored.observations[0].uri == "http://final.jpg"

    def test_from_payload_minimal(self) -> None:
        restored = from_payload(
            SkillResult, {"envelope": {"trace_id": "tr1"}, "skill_id": "sk2"}
        )
        assert restored.skill_id == "sk2"
        assert restored.status == "unknown"
        assert restored.success is None
        assert isinstance(restored.envelope, Envelope)
        assert restored.envelope.trace_id == "tr1"

    def test_round_trip_preserves_equality(self) -> None:
        env = Envelope(trace_id="tr1", episode_id="ep1")
        result = SkillResult(
            envelope=env,
            skill_id="sk1",
            name="test",
            status="completed",
            success=True,
            steps_executed=5,
            progress=1.0,
            summary="Done",
            observations=[ImageRef(uri="http://img1", camera="cam0")],
        )
        payload = to_payload(result)
        restored = from_payload(SkillResult, payload)
        assert restored == result


class TestNewId:
    def test_prefix_short(self) -> None:
        uid = _new_id("test")
        assert uid.startswith("test_")
        assert len(uid) > len("test_")

    def test_uniqueness(self) -> None:
        ids = {_new_id("test") for _ in range(100)}
        assert len(ids) == 100


class TestTopics:
    def test_default_topics(self) -> None:
        t = Topics()
        assert t.user_turn == "user.turn"
        assert t.agent_reply == "agent.reply"
        assert t.runtime_event == "runtime.event"

    def test_for_robot_with_id(self) -> None:
        t = Topics()
        assert t.for_robot("robot.status", "r1") == "robot.status.r1"

    def test_for_robot_with_none(self) -> None:
        t = Topics()
        assert t.for_robot("robot.status", None) == "robot.status"

    def test_for_robot_with_empty_string(self) -> None:
        t = Topics()
        assert t.for_robot("robot.status", "") == "robot.status"

    def test_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        t = Topics()
        with pytest.raises(FrozenInstanceError):
            t.user_turn = "other"  # type: ignore[misc]
