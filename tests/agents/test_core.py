from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np
import pytest

from hey_robot.agents.core import RobotAgentCore
from hey_robot.agents.perception_query import SceneEvidence
from hey_robot.agents.skill_state import SkillPhase, SkillStateMachine
from hey_robot.agents.turn_policy import RobotTurnPolicy
from hey_robot.agents.types import AgentTurnInput, RobotSnapshot
from hey_robot.config import AgentSpec, DeploymentConfig
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
    notifications: list[
        tuple[str, str | None, str | None, str | None, str | None, bool, dict]
    ] = field(default_factory=list)
    task_results: list[tuple[bool, str]] = field(default_factory=list)
    scene_queries: list[dict[str, object]] = field(default_factory=list)

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
        metadata: dict | None = None,
    ) -> None:
        self.notifications.append(
            (
                text,
                channel,
                chat_id,
                sender_id,
                message_id,
                reply_to_current,
                dict(metadata or {}),
            )
        )

    async def publish_task_result(self, *, success: bool, summary: str) -> None:
        self.task_results.append((success, summary))

    async def query_scene_evidence(
        self,
        *,
        robot_id: str | None,
        question: str,
        baseline_frame_id: int | None = None,
        freshness: str = "fresh",
        timeout_sec: float = 2.0,
    ) -> SceneEvidence:
        assert timeout_sec > 0
        self.scene_queries.append(
            {
                "robot_id": robot_id,
                "question": question,
                "baseline_frame_id": baseline_frame_id,
                "freshness": freshness,
                "timeout_sec": timeout_sec,
            }
        )
        return SceneEvidence(
            status="ok",
            frame_id=(baseline_frame_id or 0) + 1,
            image_count=1,
            summary="Scene summary: desk ahead.",
            confidence=0.9,
            metadata={
                "robot_id": robot_id,
                "question": question,
                "freshness": freshness,
            },
        )


@dataclass
class FakeMediaResolver:
    images: list[np.ndarray] = field(default_factory=list)

    def resolve_images(self, _refs):
        return list(self.images)


async def _run_core_turn_with_policy(
    core: RobotAgentCore,
    *,
    turn: UserTurn,
    snapshot: RobotSnapshot,
    memory_context: str | None = None,
    recovery_context: str | None = None,
):
    policy_engine = RobotTurnPolicy(core.spec)
    payload = AgentTurnInput(
        turn=turn,
        snapshot=snapshot,
        memory_context=memory_context,
        recovery_context=recovery_context,
    )
    policy = policy_engine.build(payload)
    perception_context = await policy_engine.collect_perception_context(
        core=core,
        payload=payload,
        policy=policy,
    )
    return await core.handle_turn(
        AgentTurnInput(
            turn=turn,
            snapshot=snapshot,
            memory_context=memory_context,
            recovery_context=recovery_context,
            perception_context=perception_context,
            allowed_tools=policy.allowed_tools,
        )
    )


def test_robot_agent_core_direct_mode_issues_skill() -> None:
    io = FakeAgentIO()
    core = RobotAgentCore(
        agent_id="main",
        spec=AgentSpec(type="robot_agent", settings={"mode": "direct"}),
        io=io,
    )
    turn = UserTurn(
        envelope=Envelope(agent_id="main", robot_id="mock0", episode_id="s1"),
        text="pick up the block",
    )

    result = asyncio.run(
        _run_core_turn_with_policy(
            core, turn=turn, snapshot=RobotSnapshot(robot_id="mock0")
        )
    )

    assert result.skill_submitted is True
    assert io.skills[0].objective == "pick up the block"
    assert core.skill_state.snapshot.phase == SkillPhase.ISSUED


def test_robot_agent_core_agent_mode_uses_runtime_tool() -> None:
    from tests.conftest import FakeProvider

    provider = FakeProvider("open drawer task acknowledged.")
    io = FakeAgentIO()
    core = RobotAgentCore(
        agent_id="main",
        spec=AgentSpec(type="robot_agent", settings={"max_iterations": 1}),
        io=io,
        provider=provider,
    )
    turn = UserTurn(
        envelope=Envelope(agent_id="main", robot_id="mock0"), text="open drawer"
    )

    result = asyncio.run(
        _run_core_turn_with_policy(
            core, turn=turn, snapshot=RobotSnapshot(robot_id="mock0")
        )
    )

    assert result.task_finished is True
    assert result.metadata["stop_reason"] == "text_response"
    assert result.tool == "final_response"
    assert result.reply_text == "open drawer task acknowledged."


def test_robot_agent_core_chat_mode_identity_query_stays_text_only() -> None:
    from tests.conftest import FakeProvider

    io = FakeAgentIO()
    core = RobotAgentCore(
        agent_id="main",
        spec=AgentSpec(type="robot_agent", settings={"max_iterations": 1}),
        io=io,
        provider=FakeProvider(
            "我是小白，一个有真实身体、传感器和行动边界的机器人助手。"
        ),
    )
    turn = UserTurn(
        envelope=Envelope(agent_id="main", robot_id="mock0"),
        text="你叫什么名字",
        metadata={"interaction_mode": "chat"},
    )

    result = asyncio.run(
        _run_core_turn_with_policy(
            core, turn=turn, snapshot=RobotSnapshot(robot_id="mock0")
        )
    )

    assert result.tool == "final_response"
    assert result.reply_text is not None
    assert "我是小白" in result.reply_text
    assert io.skills == []


def test_robot_agent_core_chat_mode_status_query_uses_read_only_tool() -> None:
    from tests.conftest import FakeProvider

    io = FakeAgentIO()
    core = RobotAgentCore(
        agent_id="main",
        spec=AgentSpec(type="robot_agent", settings={"max_iterations": 1}),
        io=io,
        provider=FakeProvider(
            {
                "tool": "get_robot_status",
                "args": {"include_observation": True},
            }
        ),
    )
    turn = UserTurn(
        envelope=Envelope(agent_id="main", robot_id="mock0"),
        text="电池电量是多少",
        metadata={"interaction_mode": "chat"},
    )
    snapshot = RobotSnapshot(
        robot_id="mock0",
        status=RobotStatus(
            envelope=Envelope(robot_id="mock0"),
            state="failed",
            metrics={
                "battery": {"status": "normal", "percentage": 66.7, "voltage": 11.4}
            },
        ),
    )

    result = asyncio.run(_run_core_turn_with_policy(core, turn=turn, snapshot=snapshot))

    assert result.tool == "final_response"
    assert result.reply_text is not None
    assert "电池约 66.7%" in result.reply_text
    assert io.skills == []


def test_robot_agent_core_blocks_unsupported_door_task_before_planning() -> None:
    from tests.conftest import FakeProvider

    provider = FakeProvider("should not be called")
    io = FakeAgentIO()
    core = RobotAgentCore(
        agent_id="main",
        spec=AgentSpec(
            type="robot_agent", robot_id="xlerobot", settings={"max_iterations": 1}
        ),
        io=io,
        provider=provider,
    )
    turn = UserTurn(
        envelope=Envelope(agent_id="main", robot_id="xlerobot", channel="voice"),
        text="打开门",
    )

    result = asyncio.run(
        _run_core_turn_with_policy(
            core, turn=turn, snapshot=RobotSnapshot(robot_id="xlerobot")
        )
    )

    assert result.tool == "task_safety"
    assert result.task_finished is True
    assert result.metadata["safety_rule"] == "unsupported_physical_task"
    assert result.reply_text is not None
    assert "不能安全" in result.reply_text
    assert io.skills == []
    assert provider.last_messages is None


def test_robot_agent_core_blocks_voice_motion_skill_request() -> None:
    from tests.conftest import FakeProvider

    io = FakeAgentIO()
    core = RobotAgentCore(
        agent_id="main",
        spec=AgentSpec(
            type="robot_agent", robot_id="xlerobot", settings={"skill_timeout_sec": 0.1}
        ),
        io=io,
        provider=FakeProvider("should not be used"),
    )
    turn = UserTurn(
        envelope=Envelope(agent_id="main", robot_id="xlerobot", channel="voice"),
        text="前进十厘米",
    )
    core.bind_turn_context(
        AgentTurnInput(turn=turn, snapshot=RobotSnapshot(robot_id="xlerobot"))
    )

    with pytest.raises(RuntimeError, match="语音指令不能直接触发移动"):
        asyncio.run(
            core.request_capability(
                "move_base",
                "前进十厘米",
                {"direction": "forward", "distance_cm": 10},
            )
        )

    assert io.skills == []


def test_robot_agent_core_injects_robot_skill_catalog_context() -> None:
    from tests.conftest import FakeProvider

    provider = FakeProvider("ok")
    io = FakeAgentIO()
    core = RobotAgentCore(
        agent_id="main",
        spec=AgentSpec(
            type="robot_agent", robot_id="xlerobot", settings={"max_iterations": 1}
        ),
        io=io,
        provider=provider,
    )
    turn = UserTurn(
        envelope=Envelope(agent_id="main", robot_id="xlerobot"), text="打开夹爪"
    )

    asyncio.run(
        _run_core_turn_with_policy(
            core, turn=turn, snapshot=RobotSnapshot(robot_id="xlerobot")
        )
    )

    assert provider.last_messages is not None
    turn_prompt = provider.last_messages[1].content
    assert "Robot capability catalog for request_capability.capability" in turn_prompt
    assert "set_gripper" in turn_prompt
    assert "inspect_scene" in turn_prompt
    assert "camera_capture" not in turn_prompt
    assert (
        "Choose request_capability.capability exactly from this catalog" in turn_prompt
    )


def test_robot_agent_core_prompt_uses_enabled_skill_surface() -> None:
    from tests.conftest import FakeProvider

    config = DeploymentConfig.from_dict({"skills": {"enabled": ["inspect_scene"]}})
    provider = FakeProvider("ok")
    core = RobotAgentCore(
        agent_id="main",
        spec=AgentSpec(
            type="robot_agent",
            robot_id="xlerobot",
            settings={"max_iterations": 1, "_deployment_config": config},
        ),
        io=FakeAgentIO(),
        provider=provider,
    )
    turn = UserTurn(
        envelope=Envelope(agent_id="main", robot_id="xlerobot"), text="look"
    )

    asyncio.run(
        _run_core_turn_with_policy(
            core, turn=turn, snapshot=RobotSnapshot(robot_id="xlerobot")
        )
    )

    assert provider.last_messages is not None
    turn_prompt = provider.last_messages[1].content
    assert "inspect_scene" in turn_prompt
    assert "set_gripper" not in turn_prompt


def test_robot_agent_core_rejects_disabled_skill_request() -> None:
    from tests.conftest import FakeProvider

    config = DeploymentConfig.from_dict({"skills": {"enabled": ["inspect_scene"]}})
    core = RobotAgentCore(
        agent_id="main",
        spec=AgentSpec(
            type="robot_agent",
            robot_id="xlerobot",
            settings={"_deployment_config": config},
        ),
        io=FakeAgentIO(),
        provider=FakeProvider("should not be used"),
    )
    core.bind_turn_context(
        AgentTurnInput(
            turn=UserTurn(
                envelope=Envelope(agent_id="main", robot_id="xlerobot"), text="open"
            ),
            snapshot=RobotSnapshot(robot_id="xlerobot"),
        )
    )

    with pytest.raises(KeyError):
        asyncio.run(
            core.request_capability("set_gripper", "open gripper", {"action": "open"})
        )


def test_robot_agent_core_accepts_atomic_camera_skill_submission() -> None:
    from tests.conftest import FakeProvider

    io = FakeAgentIO()
    core = RobotAgentCore(
        agent_id="main",
        spec=AgentSpec(
            type="robot_agent", robot_id="xlerobot", settings={"max_iterations": 1}
        ),
        io=io,
        provider=FakeProvider("should not be used"),
    )
    turn = UserTurn(
        envelope=Envelope(agent_id="main", robot_id="xlerobot"), text="look"
    )
    core.bind_turn_context(
        AgentTurnInput(turn=turn, snapshot=RobotSnapshot(robot_id="xlerobot"))
    )

    async def submit_and_resolve(skill: SkillIntent) -> None:
        io.skills.append(skill)
        assert core.resolve_skill(skill.skill_id, "captured")

    io.submit_skill = submit_and_resolve  # type: ignore[method-assign]

    result = asyncio.run(core.request_capability("inspect_scene", "look"))

    assert result == "captured"
    assert io.skills[0].name == "inspect_scene"


def test_robot_agent_core_resolves_fast_skill_result() -> None:
    from tests.conftest import FakeProvider

    io = FakeAgentIO()
    core = RobotAgentCore(
        agent_id="main",
        spec=AgentSpec(
            type="robot_agent", settings={"max_iterations": 1, "skill_timeout_sec": 1.0}
        ),
        io=io,
        provider=FakeProvider("should not be used"),
    )
    turn = UserTurn(
        envelope=Envelope(agent_id="main", robot_id="mock0"), text="close gripper"
    )
    core.bind_turn_context(
        AgentTurnInput(turn=turn, snapshot=RobotSnapshot(robot_id="mock0"))
    )

    async def submit_and_resolve(skill: SkillIntent) -> None:
        io.skills.append(skill)
        assert core.resolve_skill(skill.skill_id, "gripper closed")

    io.submit_skill = submit_and_resolve  # type: ignore[method-assign]

    result = asyncio.run(
        core.request_capability("set_gripper", "close gripper", {"action": "close"})
    )

    assert result == "gripper closed"
    assert io.skills[0].name == "set_gripper"


def test_robot_agent_core_request_perception_uses_inspect_skill_and_captions_result() -> (
    None
):
    from tests.conftest import FakeProvider

    io = FakeAgentIO()
    core = RobotAgentCore(
        agent_id="main",
        spec=AgentSpec(
            type="robot_agent", settings={"max_iterations": 1, "skill_timeout_sec": 1.0}
        ),
        io=io,
        provider=FakeProvider("done"),
    )
    turn = UserTurn(
        envelope=Envelope(agent_id="main", robot_id="mock0"), text="你看到了什么"
    )
    core.bind_turn_context(
        AgentTurnInput(turn=turn, snapshot=RobotSnapshot(robot_id="mock0"))
    )

    async def submit_and_resolve(skill: SkillIntent) -> None:
        io.skills.append(skill)
        assert skill.name == "inspect_scene"
        assert core.resolve_skill(skill.skill_id, "perception refreshed")
        core.observe_skill_result(skill.skill_id, "completed", None)
        core.observe_skill_result(skill.skill_id, "completed", None)

    io.submit_skill = submit_and_resolve  # type: ignore[method-assign]

    result = asyncio.run(core.request_perception(question="你看到了什么"))

    assert "Scene summary: desk ahead." in result
    assert "Execution feedback" not in result
    assert io.skills[0].name == "inspect_scene"
    assert io.skills[0].arguments == {"question": turn.text}


def test_robot_agent_core_scene_question_requires_perception_evidence() -> None:
    from tests.conftest import FakeProvider

    provider = FakeProvider(["I see a desk and chair.", "I see a desk and chair."])
    io = FakeAgentIO()
    core = RobotAgentCore(
        agent_id="main",
        spec=AgentSpec(
            type="robot_agent", settings={"max_iterations": 2, "skill_timeout_sec": 1.0}
        ),
        io=io,
        provider=provider,
    )
    turn = UserTurn(
        envelope=Envelope(agent_id="main", robot_id="mock0"), text="你看到了什么"
    )

    async def submit_and_resolve(skill: SkillIntent) -> None:
        io.skills.append(skill)
        assert skill.name == "inspect_scene"
        assert core.resolve_skill(skill.skill_id, "perception refreshed")

    io.submit_skill = submit_and_resolve  # type: ignore[method-assign]

    result = asyncio.run(
        _run_core_turn_with_policy(
            core, turn=turn, snapshot=RobotSnapshot(robot_id="mock0")
        )
    )

    assert result.reply_text == "I see a desk and chair."
    assert result.tool == "final_response"
    assert io.skills[0].name == "inspect_scene"
    assert provider.last_messages is not None
    assert "Scene summary: desk ahead." in provider.last_messages[-1].content


def test_robot_agent_core_skips_active_perception_when_observation_is_fresh() -> None:
    from tests.conftest import FakeProvider

    provider = FakeProvider("The latest frame is available.")
    io = FakeAgentIO()
    core = RobotAgentCore(
        agent_id="main",
        spec=AgentSpec(
            type="robot_agent",
            settings={"max_iterations": 1, "active_perception_max_age_sec": 30.0},
        ),
        io=io,
        provider=provider,
    )
    observation = RobotObservation(
        envelope=Envelope(agent_id="main", robot_id="mock0"),
        frame_id=5,
        images=[ImageRef(uri="memory://frame5")],
    )
    turn = UserTurn(
        envelope=Envelope(agent_id="main", robot_id="mock0"), text="你看到了什么"
    )

    result = asyncio.run(
        _run_core_turn_with_policy(
            core,
            turn=turn,
            snapshot=RobotSnapshot(robot_id="mock0", observation=observation),
        )
    )

    assert result.reply_text == "The latest frame is available."
    assert io.skills == []


def test_robot_agent_core_uses_tool_result_as_reply_when_runtime_budget_expires() -> (
    None
):
    from tests.conftest import FakeProvider

    provider = FakeProvider(
        [
            {
                "tool": "request_capability",
                "args": {
                    "capability": "set_gripper",
                    "objective": "close",
                    "slots": {"action": "close"},
                },
            },
            "The gripper is closed.",
        ]
    )
    io = FakeAgentIO()
    core = RobotAgentCore(
        agent_id="main",
        spec=AgentSpec(
            type="robot_agent", settings={"max_iterations": 1, "skill_timeout_sec": 1.0}
        ),
        io=io,
        provider=provider,
    )

    async def submit_and_resolve(skill: SkillIntent) -> None:
        io.skills.append(skill)
        core.resolve_skill(skill.skill_id, "gripper closed")

    io.submit_skill = submit_and_resolve  # type: ignore[method-assign]
    turn = UserTurn(
        envelope=Envelope(agent_id="main", robot_id="mock0"), text="close gripper"
    )

    result = asyncio.run(
        _run_core_turn_with_policy(
            core, turn=turn, snapshot=RobotSnapshot(robot_id="mock0")
        )
    )

    assert result.reply_text == "The gripper is closed."
    assert result.metadata["stop_reason"] == "text_response"


def test_robot_agent_core_marks_capability_backed_final_response_as_task_finished() -> (
    None
):
    from tests.conftest import FakeProvider

    provider = FakeProvider(
        [
            {
                "tool": "request_capability",
                "args": {
                    "capability": "set_gripper",
                    "objective": "close",
                    "slots": {"action": "close"},
                },
            },
            "The gripper is closed.",
        ]
    )
    io = FakeAgentIO()
    core = RobotAgentCore(
        agent_id="main",
        spec=AgentSpec(
            type="robot_agent", settings={"max_iterations": 2, "skill_timeout_sec": 1.0}
        ),
        io=io,
        provider=provider,
    )

    async def submit_and_resolve(skill: SkillIntent) -> None:
        io.skills.append(skill)
        core.resolve_skill(skill.skill_id, "gripper closed")

    io.submit_skill = submit_and_resolve  # type: ignore[method-assign]
    turn = UserTurn(
        envelope=Envelope(agent_id="main", robot_id="mock0"), text="close gripper"
    )

    result = asyncio.run(
        _run_core_turn_with_policy(
            core, turn=turn, snapshot=RobotSnapshot(robot_id="mock0")
        )
    )

    assert result.reply_text == "The gripper is closed."
    assert result.task_finished is True


def test_robot_agent_core_does_not_finish_when_latest_feedback_says_continue() -> None:
    from tests.conftest import FakeProvider

    internal_feedback = (
        "Execution feedback for skill skill_move:\n"
        "- outcome: confirmed\n"
        "- subgoal_success: True\n"
        "- task_success: False\n"
        "- summary: base moved forward 25.0cm; robot_state=idle\n"
        "- confidence: 0.75\n"
        "- next_hint: continue with the next useful step\n"
        "- recommended_action: continue\n"
        "\n"
        "Task continuation:\n"
        "- remaining_goal: Continue advancing the original task until you can report completion."
    )
    provider = FakeProvider(
        [
            {
                "tool": "request_capability",
                "args": {
                    "capability": "move_base",
                    "objective": "move closer",
                    "slots": {"direction": "forward", "distance_cm": 25},
                },
            },
            internal_feedback,
        ]
    )
    io = FakeAgentIO()
    core = RobotAgentCore(
        agent_id="main",
        spec=AgentSpec(
            type="robot_agent",
            settings={"max_iterations": 2, "skill_timeout_sec": 1.0},
        ),
        io=io,
        provider=provider,
    )

    async def submit_and_resolve(skill: SkillIntent) -> None:
        io.skills.append(skill)
        core.resolve_skill(skill.skill_id, internal_feedback)

    io.submit_skill = submit_and_resolve  # type: ignore[method-assign]

    result = asyncio.run(
        _run_core_turn_with_policy(
            core,
            turn=UserTurn(
                envelope=Envelope(agent_id="main", robot_id="mock0"),
                text="move closer and inspect",
            ),
            snapshot=RobotSnapshot(robot_id="mock0"),
        )
    )

    assert result.task_finished is False
    assert result.metadata["stop_reason"] == "internal_protocol_response"
    assert result.reply_text is not None
    assert "Execution feedback for skill" not in result.reply_text
    assert "Task continuation:" not in result.reply_text


def test_robot_agent_core_finishes_final_response_after_successful_atomic_skill() -> (
    None
):
    from tests.conftest import FakeProvider

    feedback = (
        "Execution feedback for skill skill_gripper:\n"
        "- outcome: confirmed\n"
        "- subgoal_success: True\n"
        "- task_success: False\n"
        "- summary: gripper set to 0.080rad; robot_state=observed\n"
        "- confidence: 0.75\n"
        "- next_hint: continue with the next useful step\n"
        "- recommended_action: continue\n"
    )
    provider = FakeProvider(
        [
            {
                "tool": "request_capability",
                "args": {
                    "capability": "set_gripper",
                    "objective": "open gripper",
                    "slots": {"action": "open"},
                },
            },
            "夹爪已打开。",
        ]
    )
    io = FakeAgentIO()
    core = RobotAgentCore(
        agent_id="main",
        spec=AgentSpec(
            type="robot_agent", settings={"max_iterations": 2, "skill_timeout_sec": 1.0}
        ),
        io=io,
        provider=provider,
    )

    async def submit_and_resolve(skill: SkillIntent) -> None:
        io.skills.append(skill)
        core.resolve_skill(skill.skill_id, feedback)

    io.submit_skill = submit_and_resolve  # type: ignore[method-assign]

    result = asyncio.run(
        _run_core_turn_with_policy(
            core,
            turn=UserTurn(
                envelope=Envelope(agent_id="main", robot_id="mock0"),
                text="打开夹爪",
            ),
            snapshot=RobotSnapshot(robot_id="mock0"),
        )
    )

    assert result.reply_text == "夹爪已打开。"
    assert result.task_finished is True


def test_robot_agent_core_finishes_visual_answer_after_successful_perception() -> None:
    from tests.conftest import FakeProvider

    feedback = (
        "Execution feedback for skill skill_scene:\n"
        "- outcome: confirmed\n"
        "- subgoal_success: True\n"
        "- task_success: False\n"
        "- summary: scene inspected; robot_state=idle\n"
        "- recommended_action: continue\n"
    )
    provider = FakeProvider(
        [
            {
                "tool": "request_capability",
                "args": {
                    "capability": "inspect_scene",
                    "objective": "describe the scene",
                },
            },
            "前方是一张桌子。",
        ]
    )
    io = FakeAgentIO()
    core = RobotAgentCore(
        agent_id="main",
        spec=AgentSpec(
            type="robot_agent", settings={"max_iterations": 2, "skill_timeout_sec": 1.0}
        ),
        io=io,
        provider=provider,
    )

    async def submit_and_resolve(skill: SkillIntent) -> None:
        io.skills.append(skill)
        core.resolve_skill(skill.skill_id, feedback)

    io.submit_skill = submit_and_resolve  # type: ignore[method-assign]
    result = asyncio.run(
        _run_core_turn_with_policy(
            core,
            turn=UserTurn(
                envelope=Envelope(agent_id="main", robot_id="mock0"),
                text="你看到了什么",
            ),
            snapshot=RobotSnapshot(robot_id="mock0"),
        )
    )

    assert result.reply_text == "前方是一张桌子。"
    assert result.task_finished is True


def test_robot_agent_core_does_not_report_failed_skill_as_completed() -> None:
    from tests.conftest import FakeProvider

    feedback = (
        "Execution feedback for skill skill_gripper:\n"
        "- outcome: failed\n"
        "- subgoal_success: False\n"
        "- task_success: False\n"
        "- summary: skill timed out\n"
        "- failure_reason: skill timed out\n"
        "- recommended_action: inspect_or_recover\n"
    )
    provider = FakeProvider(
        [
            {
                "tool": "request_capability",
                "args": {
                    "capability": "set_gripper",
                    "objective": "关闭夹爪",
                    "slots": {"action": "close"},
                },
            },
            "动作已经完成。",
        ]
    )
    io = FakeAgentIO()
    core = RobotAgentCore(
        agent_id="main",
        spec=AgentSpec(
            type="robot_agent", settings={"max_iterations": 2, "skill_timeout_sec": 1.0}
        ),
        io=io,
        provider=provider,
    )

    async def submit_and_resolve(skill: SkillIntent) -> None:
        io.skills.append(skill)
        core.resolve_skill(skill.skill_id, feedback)

    io.submit_skill = submit_and_resolve  # type: ignore[method-assign]
    result = asyncio.run(
        _run_core_turn_with_policy(
            core,
            turn=UserTurn(
                envelope=Envelope(agent_id="main", robot_id="mock0"),
                text="关闭夹爪",
            ),
            snapshot=RobotSnapshot(robot_id="mock0"),
        )
    )

    assert result.reply_text == "动作执行未成功：skill timed out"
    assert result.task_finished is False


def test_robot_agent_core_does_not_reuse_previous_turn_skill_id() -> None:
    from tests.conftest import FakeProvider

    provider = FakeProvider(
        [
            {
                "tool": "request_capability",
                "args": {
                    "capability": "set_gripper",
                    "objective": "close",
                    "slots": {"action": "close"},
                },
            },
            "The gripper is closed.",
            "I cannot do that.",
        ]
    )
    io = FakeAgentIO()
    core = RobotAgentCore(
        agent_id="main",
        spec=AgentSpec(
            type="robot_agent", settings={"max_iterations": 2, "skill_timeout_sec": 1.0}
        ),
        io=io,
        provider=provider,
    )

    async def submit_and_resolve(skill: SkillIntent) -> None:
        io.skills.append(skill)
        core.resolve_skill(skill.skill_id, "gripper closed")

    io.submit_skill = submit_and_resolve  # type: ignore[method-assign]
    first = UserTurn(
        envelope=Envelope(agent_id="main", robot_id="mock0"), text="close gripper"
    )
    second = UserTurn(
        envelope=Envelope(agent_id="main", robot_id="mock0"), text="move closer"
    )

    first_result = asyncio.run(
        _run_core_turn_with_policy(
            core, turn=first, snapshot=RobotSnapshot(robot_id="mock0")
        )
    )
    second_result = asyncio.run(
        _run_core_turn_with_policy(
            core, turn=second, snapshot=RobotSnapshot(robot_id="mock0")
        )
    )

    assert first_result.metadata["skill_id"] == io.skills[0].skill_id
    assert first_result.task_finished is True
    assert second_result.metadata["skill_id"] is None
    assert second_result.task_finished is True


def test_robot_agent_core_recovery_blocks_new_skill() -> None:
    from tests.conftest import FakeProvider

    io = FakeAgentIO()
    core = RobotAgentCore(
        agent_id="main",
        spec=AgentSpec(type="robot_agent", settings={"max_iterations": 1}),
        io=io,
        provider=FakeProvider("should not be used"),
    )
    turn = UserTurn(
        envelope=Envelope(agent_id="main", robot_id="mock0"), text="open drawer"
    )

    result = asyncio.run(
        _run_core_turn_with_policy(
            core,
            turn=turn,
            snapshot=RobotSnapshot(robot_id="mock0"),
            recovery_context="Recovery context: verify before moving",
        )
    )

    assert result.skill_submitted is False
    assert io.skills == []


def test_robot_turn_policy_limits_tools_during_recovery() -> None:
    policy_engine = RobotTurnPolicy(AgentSpec(type="robot_agent", settings={}))
    payload = AgentTurnInput(
        turn=UserTurn(
            envelope=Envelope(agent_id="main", robot_id="mock0"), text="open drawer"
        ),
        snapshot=RobotSnapshot(robot_id="mock0"),
        recovery_context="Recovery context: verify before moving",
        block_actuation=True,
    )

    policy = policy_engine.build(payload)

    assert policy.allowed_tools is not None
    assert "get_task_context" in policy.allowed_tools
    assert "request_perception" not in policy.allowed_tools
    assert "request_capability" not in policy.allowed_tools


def test_recovery_allowed_tools_contains_all_essential_tools() -> None:
    tools = RobotTurnPolicy.recovery_allowed_tools()

    assert "get_task_context" in tools
    assert "get_robot_status" in tools
    assert "search_memory" in tools
    assert "wait" in tools
    assert len(tools) == 4


def test_recovery_allowed_tools_excludes_perception_and_action() -> None:
    tools = RobotTurnPolicy.recovery_allowed_tools()

    assert "request_perception" not in tools
    assert "request_capability" not in tools
    assert "propose_capability" not in tools


def test_recovery_tools_are_stable_subset_of_all_tools() -> None:
    recovery = RobotTurnPolicy.recovery_allowed_tools()
    all_tools = {
        "get_task_context",
        "get_robot_status",
        "propose_capability",
        "request_capability",
        "request_perception",
        "search_memory",
        "wait",
        "write_memory",
    }

    assert recovery <= all_tools


def test_turn_policy_uses_recovery_tools_when_blocked() -> None:
    policy_engine = RobotTurnPolicy(AgentSpec(type="robot_agent", settings={}))
    payload = AgentTurnInput(
        turn=UserTurn(
            envelope=Envelope(agent_id="main", robot_id="mock0"), text="任何指令"
        ),
        snapshot=RobotSnapshot(robot_id="mock0"),
        recovery_context="资源冲突",
        block_actuation=True,
    )

    policy = policy_engine.build(payload)

    assert policy.allowed_tools == RobotTurnPolicy.recovery_allowed_tools()


def test_robot_turn_policy_does_not_treat_gripper_action_as_status_query() -> None:
    policy_engine = RobotTurnPolicy(AgentSpec(type="robot_agent", settings={}))
    payload = AgentTurnInput(
        turn=UserTurn(
            envelope=Envelope(agent_id="main", robot_id="mock0"), text="关闭夹爪吧"
        ),
        snapshot=RobotSnapshot(robot_id="mock0"),
    )

    policy = policy_engine.build(payload)

    assert policy.allowed_tools is None


def test_robot_turn_policy_treats_arm_pose_command_as_action() -> None:
    policy_engine = RobotTurnPolicy(AgentSpec(type="robot_agent", settings={}))
    payload = AgentTurnInput(
        turn=UserTurn(
            envelope=Envelope(agent_id="main", robot_id="mock0"),
            text="把机械臂回到 home 姿态。",
        ),
        snapshot=RobotSnapshot(robot_id="mock0"),
    )

    policy = policy_engine.build(payload)

    assert policy.allowed_tools is None


def test_robot_turn_policy_treats_arm_joint_adjustment_as_action() -> None:
    policy_engine = RobotTurnPolicy(AgentSpec(type="robot_agent", settings={}))
    payload = AgentTurnInput(
        turn=UserTurn(
            envelope=Envelope(agent_id="main", robot_id="mock0"),
            text="轻微调整一下机械臂关节，保持安全。",
        ),
        snapshot=RobotSnapshot(robot_id="mock0"),
    )

    policy = policy_engine.build(payload)

    assert policy.allowed_tools is None


def test_robot_turn_policy_treats_left_turn_command_as_action() -> None:
    policy_engine = RobotTurnPolicy(AgentSpec(type="robot_agent", settings={}))
    payload = AgentTurnInput(
        turn=UserTurn(
            envelope=Envelope(agent_id="main", robot_id="mock0"),
            text="向左转一点。",
        ),
        snapshot=RobotSnapshot(robot_id="mock0"),
    )

    policy = policy_engine.build(payload)

    assert policy.allowed_tools is None


def test_robot_snapshot_summary_includes_status_metrics() -> None:
    snapshot = RobotSnapshot(
        robot_id="xlerobot",
        status=RobotStatus(
            envelope=Envelope(robot_id="xlerobot"),
            state="idle",
            metrics={
                "battery": {"status": "normal", "percentage": 50.0, "voltage": 10.8},
                "arm": {"state": "ready", "joint_count": 6},
            },
        ),
    )

    summary = snapshot.summary()

    assert "battery={status=normal,percentage=50.0,voltage=10.8}" in summary
    assert "arm={state=ready,joint_count=6}" in summary


def test_robot_agent_core_resolves_snapshot_images_for_runtime() -> None:
    from tests.conftest import FakeProvider

    io = FakeAgentIO()
    media = FakeMediaResolver(images=[np.zeros((4, 4, 3), dtype=np.uint8)])
    core = RobotAgentCore(
        agent_id="main",
        spec=AgentSpec(
            type="robot_agent",
            settings={"max_iterations": 1, "send_images_on_turn": True},
        ),
        io=io,
        media_resolver=media,
        provider=FakeProvider("ok"),
    )
    snapshot = RobotSnapshot(
        robot_id="mock0",
        observation=RobotObservation(
            envelope=Envelope(robot_id="mock0"),
            frame_id=1,
            images=[ImageRef(uri="media://local/images/mock0/test.jpg")],
        ),
    )

    images = core._snapshot_images(snapshot)

    assert len(images) == 1
    assert images[0].shape == (4, 4, 3)


def test_skill_intent_state_machine_observes_result() -> None:
    machine = SkillStateMachine()
    skill = SkillIntent(envelope=Envelope(), skill_id="cmd1", objective="move")

    machine.submit(skill)
    snapshot = machine.observe_result(
        SkillResult(envelope=Envelope(), skill_id="cmd1", status="completed")
    )

    assert snapshot.phase == SkillPhase.COMPLETED
    assert snapshot.needs_feedback is True
    machine.mark_feedback_pending()
    confirmed = machine.mark_feedback_received("ok")
    assert confirmed.phase == SkillPhase.CONFIRMED


def test_robot_agent_core_auto_bootstraps_long_horizon_plan() -> None:
    from tests.conftest import FakeProvider

    provider = FakeProvider("ready")
    io = FakeAgentIO()
    core = RobotAgentCore(
        agent_id="main",
        spec=AgentSpec(
            type="robot_agent", robot_id="xlerobot", settings={"max_iterations": 2}
        ),
        io=io,
        provider=provider,
    )
    turn = UserTurn(
        envelope=Envelope(agent_id="main", robot_id="xlerobot"),
        text=(
            "organize the workspace: find the cup and block, put the cup in the bin, "
            "put the block on the table, remember where everything ended up, and report completion"
        ),
    )
    snapshot = RobotSnapshot(
        robot_id="xlerobot",
        observation=RobotObservation(
            envelope=Envelope(robot_id="xlerobot"),
            frame_id=3,
            images=[ImageRef(uri="memory://frame3")],
        ),
    )

    result = asyncio.run(_run_core_turn_with_policy(core, turn=turn, snapshot=snapshot))

    assert result.reply_text == "ready"
    assert provider.last_messages is not None
    assert "Robot plan:" not in provider.last_messages[1].content
    assert (
        "Robot capability catalog for request_capability.capability"
        in provider.last_messages[1].content
    )


def test_robot_agent_core_mock_xlerobot_uses_single_atomic_deterministic_skill_after_perception() -> (
    None
):
    from tests.conftest import FakeProvider

    io = FakeAgentIO()
    task = (
        "organize the workspace: find the cup and the block, put the cup in the bin, "
        "put the block on the table, remember where each object ended up, and report completion."
    )
    core = RobotAgentCore(
        agent_id="main",
        spec=AgentSpec(
            type="robot_agent",
            robot_id="mock0",
            settings={
                "mode": "autonomous",
                "max_iterations": 20,
                "max_plan_robot_skills": 10,
                "skill_timeout_sec": 1.0,
            },
        ),
        io=io,
        provider=FakeProvider(
            [
                {
                    "tool": "request_capability",
                    "args": {
                        "capability": "inspect_scene",
                        "objective": task,
                        "slots": {"question": task},
                    },
                },
                {
                    "tool": "request_capability",
                    "args": {
                        "capability": "move_base",
                        "objective": task,
                        "slots": {"direction": "forward", "distance_cm": 30},
                    },
                },
                "move_base completed",
            ]
        ),
    )

    async def submit_and_resolve(skill: SkillIntent) -> None:
        io.skills.append(skill)
        assert core.resolve_skill(skill.skill_id, f"{skill.name} completed")

    io.submit_skill = submit_and_resolve  # type: ignore[method-assign]
    turn = UserTurn(envelope=Envelope(agent_id="main", robot_id="mock0"), text=task)

    result = asyncio.run(
        _run_core_turn_with_policy(
            core, turn=turn, snapshot=RobotSnapshot(robot_id="mock0")
        )
    )

    assert result.metadata["stop_reason"] == "text_response"
    assert result.reply_text == "move_base completed"
    assert [skill.name for skill in io.skills] == ["inspect_scene", "move_base"]
    assert io.skills[-1].arguments == {"direction": "forward", "distance_cm": 30}


def test_robot_agent_core_requires_explicit_runtime_provider_without_direct_mode() -> (
    None
):
    with pytest.raises(
        ValueError, match="requires an explicit agent provider configuration"
    ):
        RobotAgentCore(
            agent_id="main",
            spec=AgentSpec(type="robot_agent", settings={"max_iterations": 1}),
            io=FakeAgentIO(),
        )


# ── operator_control_turn ──────────────────────────────────────────
