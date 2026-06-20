from __future__ import annotations

import json

from hey_robot.agents.context import (
    RobotAgentContext,
    RobotContextBuilder,
    RobotMemoryContextBuilder,
)
from hey_robot.agents.types import RobotSnapshot
from hey_robot.capability.catalog.models import (
    CapabilityManifest,
    RobotSkillCapability,
    ToolCapability,
)
from hey_robot.episode import EpisodeRecord
from hey_robot.memory import LongTermMemoryStore, MemoryRuntime
from hey_robot.protocol import (
    ArtifactRef,
    Envelope,
    ImageRef,
    RobotObservation,
    UserTurn,
)


def test_robot_agent_context_memory_context_orders_available_sections() -> None:
    context = RobotAgentContext(
        task="pick cup",
        robot_state="robot_id=mock0",
        episode_context="history",
        pending_context="pending",
        metadata={
            "capability_context": "capabilities",
            "scene_context": "scene",
        },
    )

    assert context.memory_context() == "capabilities\n\nhistory\n\nscene\n\npending"


def test_robot_agent_context_memory_context_ignores_empty_and_non_string_metadata() -> (
    None
):
    context = RobotAgentContext(
        task="pick cup",
        robot_state="robot_id=mock0",
        episode_context="",
        pending_context=None,
        metadata={
            "capability_context": "",
            "scene_context": "scene",
        },
    )

    assert context.memory_context() == "scene"


def test_context_builder_limits_history_pending_and_records_observation_metadata() -> (
    None
):
    builder = RobotContextBuilder(max_history=2, max_pending=1)
    envelope = Envelope(
        trace_id="tr1",
        episode_id="ep1",
        robot_id="mock0",
        agent_id="main",
        message_id="m0",
    )
    turn = UserTurn(envelope=envelope, text="pick up the red cup")
    history = [
        EpisodeRecord(role="user", content="old message", timestamp=1.0, payload={}),
        EpisodeRecord(
            role="assistant", content="  spaced\nreply  ", timestamp=2.0, payload={}
        ),
        EpisodeRecord(role="user", content="new message", timestamp=3.0, payload={}),
    ]
    pending = [
        UserTurn(
            envelope=envelope.child(trace_id="tr2", message_id="m1"),
            text="old correction",
        ),
        UserTurn(
            envelope=envelope.child(trace_id="tr3", message_id="m2"),
            text="use the left cup",
        ),
    ]
    observation = RobotObservation(
        envelope=envelope,
        frame_id=9,
        images=[ImageRef(uri="file://frame.jpg")],
        artifacts=[ArtifactRef(uri="file://depth.bin", artifact_type="depth")],
        task="inspect table",
    )

    context = builder.build(
        turn=turn,
        snapshot=RobotSnapshot(robot_id="mock0", observation=observation),
        history=history,
        pending_turns=pending,
    )

    assert context.task == "pick up the red cup"
    assert context.episode_context is not None
    assert "old message" not in context.episode_context
    assert "assistant: spaced reply" in context.episode_context
    assert "user: new message" in context.episode_context

    assert context.pending_context is not None
    assert "old correction" not in context.pending_context
    assert "unverified" in context.pending_context
    pending_payload = json.loads(context.pending_context.split("- ", 3)[3])
    assert pending_payload == {
        "text": "use the left cup",
        "trace_id": "tr3",
        "message_id": "m2",
        "source": "operator",
        "verification": "unverified",
    }
    assert context.metadata["latest_observation"] == {
        "frame_id": 9,
        "image_count": 1,
        "artifact_count": 1,
        "task": "inspect table",
    }


def test_context_builder_returns_none_for_empty_history_pending_and_observation() -> (
    None
):
    builder = RobotContextBuilder(max_history=0, max_pending=0)
    envelope = Envelope(episode_id="ep1", robot_id="mock0", agent_id="main")
    turn = UserTurn(envelope=envelope, text="status?")

    context = builder.build(
        turn=turn,
        snapshot=RobotSnapshot(robot_id="mock0"),
        history=[],
        pending_turns=[],
    )

    assert builder.max_history == 1
    assert builder.max_pending == 1
    assert context.episode_context is None
    assert context.pending_context is None
    assert context.metadata["latest_observation"] is None


def test_context_builder_includes_capability_summary_without_prompt_skill_context() -> (
    None
):
    manifest = CapabilityManifest(
        tools=(
            ToolCapability(
                name="get_robot_status", source="local", safety_level="observe"
            ),
        ),
        robot_skill_actions=(RobotSkillCapability(name="inspect_scene"),),
    )
    builder = RobotContextBuilder(capability_manifest_provider=lambda: manifest)
    envelope = Envelope(episode_id="ep1", robot_id="mock0", agent_id="main")
    turn = UserTurn(envelope=envelope, text="look")

    context = builder.build(
        turn=turn,
        snapshot=RobotSnapshot(robot_id="mock0"),
        history=[],
    )

    assert context.metadata["capability_context"] == (
        "Current capability summary:\n- Tools: get_robot_status(local,observe)\n- Robot skill actions: inspect_scene"
    )
    assert "prompt_skill_context" not in context.metadata
    assert context.memory_context() == context.metadata["capability_context"]


def test_robot_memory_context_builder_orders_catalog_current_and_long_term_memory(
    tmp_path,
) -> None:
    store = LongTermMemoryStore(tmp_path / "long_term.jsonl")
    store.remember_entity(
        name="cup", summary="cup is on the workbench", location="workbench"
    )
    store.remember_user_preference(
        name="response_language",
        value="zh-CN",
        summary="用户偏好中文回答",
    )
    builder = RobotMemoryContextBuilder(
        memory=MemoryRuntime(store),
        robot_skill_catalog_context_provider=lambda: "Robot capability catalog",
    )

    context = builder.build(
        task="where is the cup",
        task_context="Task context",
        perception_context="Perception context",
    )

    assert context is not None
    assert context.startswith(
        "Robot capability catalog\n\nTask context\n\nPerception context\n\nLong-term memory:\n"
    )
    assert "response_language=zh-CN" in context
    assert "entity:cup: cup is on the workbench" in context
    assert (
        "Reply in zh-CN unless the user explicitly asks for another language."
        in context
    )
