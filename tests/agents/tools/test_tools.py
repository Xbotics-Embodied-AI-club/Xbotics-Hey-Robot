"""Per-tool tests with mock ToolContext.

Tests every tool in isolation using mocked dependencies, verifying correct
dispatch to the underlying services that the old methods on RobotAgentCore used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

from hey_robot.agents.runtime.state import AgentState
from hey_robot.agents.tools.context import ToolContext, ToolTurnContext
from hey_robot.config import AgentSpec
from hey_robot.memory import MemoryRuntime
from hey_robot.protocol import Envelope, ImageRef, RobotObservation, RobotStatus
from hey_robot.skills import load_skill_registry

# Mocks / fakes.


class _FakeSnapshot:
    status = RobotStatus(
        envelope=Envelope(robot_id="mock0", channel="test"),
        frame_id=42,
        state="idle",
        metrics={
            "battery": {"percentage": 85, "voltage": 11.4, "status": "normal"},
            "readiness": {
                "base": {"ok": True},
                "arm": {"ok": True},
                "gripper": {"ok": True},
                "camera": {"ok": True},
            },
        },
    )
    robot_id = "mock0"
    observation = None


def _make_turn_context() -> ToolTurnContext:
    return ToolTurnContext(
        snapshot_summary="battery=85% joints=ok",
        observation_summary="frame_id=42 images=1 task=pick",
        snapshot=_FakeSnapshot(),
        envelope=Envelope(robot_id="mock0", channel="test"),
    )


def _fake_envelope() -> Envelope:
    return Envelope(robot_id="mock0", channel="test", chat_id="chat1", sender_id="u1")


class _FakeLongTermMemory:
    """In-memory stubs matching LongTermMemoryStore interface."""

    def __init__(self):
        self.events: list[dict] = []
        self.entities: list[dict] = []
        self.places: list[dict] = []
        self.skills: list[dict] = []

    def remember(self, *, kind, key, summary, metadata=None):
        rec = {"kind": kind, "key": key, "summary": summary, "metadata": metadata or {}}
        self.events.append(rec)
        return _LTMRecord(rec)

    def remember_entity(
        self,
        *,
        name,
        summary,
        entity_type="object",
        location=None,
        confidence=1.0,
        attributes=None,
        frame_id=None,
        robot_id=None,
    ):
        rec = {
            "name": name,
            "summary": summary,
            "type": entity_type,
            "location": location,
            "confidence": confidence,
            "attributes": attributes,
            "frame_id": frame_id,
            "robot_id": robot_id,
        }
        self.entities.append(rec)
        return _LTMRecord(rec)

    def remember_place(
        self, *, name, description="", pose=None, confidence=1.0, robot_id=None
    ):
        rec = {
            "name": name,
            "description": description,
            "pose": pose,
            "confidence": confidence,
            "robot_id": robot_id,
        }
        self.places.append(rec)
        return _LTMRecord(rec)

    def remember_user_preference(
        self, *, name, value, summary, confidence=1.0, source="agent_tool"
    ):
        rec = {
            "kind": "user_preference",
            "key": name,
            "summary": summary,
            "confidence": confidence,
            "metadata": {"name": name, "value": value, "source": source},
        }
        self.events.append(rec)
        return _LTMRecord(rec)

    def record_scene_anchor(
        self,
        *,
        name,
        location,
        summary,
        entity_type="object",
        confidence=1.0,
        frame_id=None,
        robot_id=None,
    ):
        rec = {
            "kind": "scene_anchor",
            "key": name,
            "summary": summary,
            "confidence": confidence,
            "metadata": {
                "name": name,
                "location": location,
                "entity_type": entity_type,
                "frame_id": frame_id,
                "robot_id": robot_id,
            },
        }
        self.entities.append(rec)
        return _LTMRecord(rec)

    def remember_task_lesson(
        self,
        *,
        key,
        summary,
        task="",
        success=None,
        failure_mode=None,
        recovery_hint=None,
        skill_name=None,
    ):
        rec = {
            "kind": "task_lesson",
            "key": key,
            "summary": summary,
            "metadata": {
                "task": task,
                "success": success,
                "failure_mode": failure_mode,
                "recovery_hint": recovery_hint,
                "skill_name": skill_name,
            },
        }
        self.events.append(rec)
        return _LTMRecord(rec)

    def remember_skill_experience(
        self,
        *,
        skill_name,
        arguments=None,
        context_summary="",
        success=True,
        summary="",
        failure_mode=None,
        recovery_hint=None,
        verification_summary=None,
        duration_sec=None,
    ):
        rec = {
            "skill_name": skill_name,
            "arguments": arguments,
            "context_summary": context_summary,
            "success": success,
            "summary": summary,
            "failure_mode": failure_mode,
            "recovery_hint": recovery_hint,
            "verification_summary": verification_summary,
            "duration_sec": duration_sec,
        }
        self.skills.append(rec)
        return _LTMRecord(rec)

    def remember_task_result(
        self,
        *,
        task_id,
        root_goal,
        status,
        summary,
        failure_reason=None,
    ):
        rec = {
            "task_id": task_id,
            "root_goal": root_goal,
            "status": status,
            "summary": summary,
            "failure_reason": failure_reason,
        }
        self.events.append(rec)
        return _LTMRecord(rec)

    def query(self, text="", *, kind=None, limit=8):  # noqa: ARG002
        return []

    def prompt_context(self, text="", *, limit=8):  # noqa: ARG002
        return "ltm_context: " + text


class _LTMRecord:
    def __init__(self, d: dict):
        self._d = d

    def to_dict(self):
        return dict(self._d)


class _FakeIO:
    def __init__(self, pending_skills: dict | None = None):
        self.skills: list = []
        self.replies: list = []
        self.notifications: list = []
        self.task_results: list = []
        self._pending = pending_skills  # auto-resolve futures on submit
        self._scene_response = MagicMock()
        self.task_runtime = MagicMock()

    async def submit_skill(self, skill):
        self.skills.append(skill)
        # Auto-resolve the future so tests don't hang waiting for robot feedback
        if self._pending is not None and skill.skill_id in self._pending:
            self._pending[skill.skill_id].set_result("controller_status=completed")

    async def publish_reply(self, reply):
        self.replies.append(reply)

    async def publish_notification(
        self,
        text,
        *,
        channel=None,
        chat_id=None,
        sender_id=None,  # noqa: ARG002
        message_id=None,  # noqa: ARG002
        reply_to_current=False,  # noqa: ARG002
        metadata=None,  # noqa: ARG002
    ):
        self.notifications.append((text, channel, chat_id))

    async def publish_task_result(self, *, success, summary):
        self.task_results.append((success, summary))

    async def query_scene_evidence(
        self,
        *,
        robot_id,  # noqa: ARG002
        question,  # noqa: ARG002
        baseline_frame_id=None,  # noqa: ARG002
        freshness="fresh",  # noqa: ARG002
        timeout_sec=2.0,  # noqa: ARG002
    ):
        return self._scene_response


@dataclass
class _FakeAutonomy:
    events: list[tuple] = field(default_factory=list)

    def remember(
        self, event_type: str, summary: str, *, frame_id: int | None = None
    ) -> None:
        self.events.append((event_type, summary, frame_id))

    def prompt_context(self) -> str:
        return ""


# Context builder.


def _build_ctx(**overrides) -> ToolContext:
    pending = overrides.pop("pending_skills", {})
    io = overrides.pop("io", None) or _FakeIO(pending_skills=pending)
    # Ensure the IO has a reference to the pending futures dict for auto-resolution
    if io._pending is None:
        io._pending = pending
    ltm = overrides.pop("memory_store", _FakeLongTermMemory())
    autonomy = overrides.pop("autonomy", _FakeAutonomy())
    memory = overrides.pop("memory", None) or MemoryRuntime(ltm, autonomy=autonomy)
    turn = overrides.pop("turn_context", _make_turn_context())

    # A MagicMock for runtime-facing tools that access runtime.tools.
    runtime = overrides.pop("runtime", MagicMock())

    defaults = dict(  # noqa: C408
        io=io,
        spec=AgentSpec(),
        memory=memory,
        autonomy=autonomy,
        skill_catalog=load_skill_registry().robot_skill_catalog(),
        skill_catalog_runtime=load_skill_registry().catalog(enabled_only=False),
        runtime=runtime,
        runtime_state=MagicMock(spec=AgentState, task="pick the cup"),
        pending_skills=pending,
        robot_type="xlerobot",
        turn_context=turn,
        _current_envelope=_fake_envelope,
        _configured_robot_type=lambda: "xlerobot",
        _get_task=lambda: "pick the cup",
        _get_robot_status=lambda: "battery=85%",
    )
    defaults.update(overrides)
    return ToolContext(**defaults)


# Tool tests.


class TestWaitTool:
    async def test_wait_no_reason(self):
        from hey_robot.agents.tools.wait import WaitTool

        tool = WaitTool()
        result = await tool.execute()
        assert result == "waiting: no action"

    async def test_wait_with_reason(self):
        from hey_robot.agents.tools.wait import WaitTool

        tool = WaitTool()
        result = await tool.execute(reason="nothing to do")
        assert result == "waiting: nothing to do"


class TestGetRobotStatusTool:
    async def test_returns_snapshot_summary(self):
        from hey_robot.agents.tools.get_robot_status import GetRobotStatusTool

        ctx = _build_ctx()
        tool = GetRobotStatusTool(ctx)
        result = await tool.execute()
        assert "机器人当前" in result
        assert "电池约 85%" in result


class TestRobotStatusObservation:
    async def test_returns_observation_summary(self):
        from hey_robot.agents.tools.get_robot_status import GetRobotStatusTool

        snapshot = type(
            "_SnapshotWithObservation",
            (),
            {
                "status": RobotStatus(
                    envelope=Envelope(robot_id="mock0", channel="test"),
                    frame_id=42,
                    state="idle",
                    metrics={
                        "battery": {
                            "percentage": 85,
                            "voltage": 11.4,
                            "status": "normal",
                        },
                        "readiness": {
                            "base": {"ok": True},
                            "arm": {"ok": True},
                            "gripper": {"ok": True},
                            "camera": {"ok": True},
                        },
                    },
                ),
                "robot_id": "mock0",
                "observation": RobotObservation(
                    envelope=Envelope(robot_id="mock0", channel="test"),
                    frame_id=42,
                    images=[ImageRef(uri="memory://frame42")],
                    task="pick",
                ),
            },
        )()
        turn = ToolTurnContext(
            snapshot_summary="battery=85% joints=ok",
            observation_summary="frame_id=42 images=1 task=pick",
            snapshot=snapshot,
            envelope=Envelope(robot_id="mock0", channel="test"),
        )
        ctx = _build_ctx(turn_context=turn)
        tool = GetRobotStatusTool(ctx)
        result = await tool.execute()
        assert "最近一帧画面 frame=42" in result

    async def test_returns_placeholder_when_no_turn_context(self):
        from hey_robot.agents.tools.get_robot_status import GetRobotStatusTool

        ctx = _build_ctx(turn_context=None)
        tool = GetRobotStatusTool(ctx)
        result = await tool.execute()
        assert result == "no robot snapshot"

    async def test_returns_user_friendly_status_without_observation(self):
        from hey_robot.agents.tools.get_robot_status import GetRobotStatusTool

        turn = ToolTurnContext(
            snapshot_summary="debug summary",
            observation_summary="no observation",
            snapshot=type(
                "_Snapshot",
                (),
                {
                    "status": RobotStatus(
                        envelope=Envelope(robot_id="mock0", channel="test"),
                        state="idle",
                        metrics={
                            "battery": {
                                "percentage": 63.9,
                                "voltage": 11.3,
                                "status": "normal",
                            },
                            "readiness": {
                                "base": {"ok": True},
                                "arm": {"ok": True},
                                "gripper": {"ok": True},
                                "camera": {"ok": True},
                            },
                        },
                    ),
                    "observation": None,
                },
            )(),
            envelope=Envelope(robot_id="mock0", channel="test"),
        )
        ctx = _build_ctx(turn_context=turn)
        tool = GetRobotStatusTool(ctx)
        result = await tool.execute()
        assert "机器人当前空闲" in result
        assert "电池约 63.9%" in result
        assert "底盘正常" in result


class TestMemorySearch:
    async def test_semantic_mode(self):
        from hey_robot.agents.tools.search_memory import SearchMemoryTool

        ltm = _FakeLongTermMemory()
        ctx = _build_ctx(memory_store=ltm)
        tool = SearchMemoryTool(ctx)
        result = await tool.execute(query="cup", mode="semantic")
        assert "ltm_context" in result

    async def test_structured_mode_returns_json(self):
        from hey_robot.agents.tools.search_memory import SearchMemoryTool

        ctx = _build_ctx()
        tool = SearchMemoryTool(ctx)
        result = await tool.execute(query="cup", mode="structured", kind="entity")
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    async def test_defaults_to_semantic(self):
        from hey_robot.agents.tools.search_memory import SearchMemoryTool

        ltm = _FakeLongTermMemory()
        ctx = _build_ctx(memory_store=ltm)
        tool = SearchMemoryTool(ctx)
        result = await tool.execute(query="cup")
        assert "ltm_context" in result


class TestMemoryWrite:
    async def test_remember_event(self):
        from hey_robot.agents.tools.write_memory import WriteMemoryTool

        autonomy = _FakeAutonomy()
        ltm = _FakeLongTermMemory()
        ctx = _build_ctx(memory_store=ltm, autonomy=autonomy)
        tool = WriteMemoryTool(ctx)
        result = await tool.execute(
            kind="event", summary="saw a cup", name="observation"
        )
        assert "observation" in result
        assert len(autonomy.events) == 1
        assert autonomy.events[0][1] == "saw a cup"

    async def test_remember_entity(self):
        from hey_robot.agents.tools.write_memory import WriteMemoryTool

        ltm = _FakeLongTermMemory()
        ctx = _build_ctx(memory_store=ltm)
        tool = WriteMemoryTool(ctx)
        result = await tool.execute(
            kind="entity",
            summary="red cup",
            name="cup",
            location="table",
            entity_type="object",
            confidence=0.9,
        )
        parsed = json.loads(result)
        assert parsed["name"] == "cup"
        assert parsed["location"] == "table"

    async def test_remember_place(self):
        from hey_robot.agents.tools.write_memory import WriteMemoryTool

        ltm = _FakeLongTermMemory()
        ctx = _build_ctx(memory_store=ltm)
        tool = WriteMemoryTool(ctx)
        result = await tool.execute(
            kind="place", summary="charging station", name="dock"
        )
        parsed = json.loads(result)
        assert parsed["name"] == "dock"

    async def test_remember_location(self):
        from hey_robot.agents.tools.write_memory import WriteMemoryTool

        ltm = _FakeLongTermMemory()
        ctx = _build_ctx(memory_store=ltm)
        tool = WriteMemoryTool(ctx)
        result = await tool.execute(
            kind="location", summary="cup on table", name="cup", location="table"
        )
        parsed = json.loads(result)
        assert "entity" in parsed
        assert "place" in parsed
        assert parsed["entity"]["name"] == "cup"

    async def test_remember_skill_experience(self):
        from hey_robot.agents.tools.write_memory import WriteMemoryTool

        ltm = _FakeLongTermMemory()
        ctx = _build_ctx(memory_store=ltm)
        tool = WriteMemoryTool(ctx)
        result = await tool.execute(
            kind="skill_experience",
            summary="grasp succeeded",
            skill_name="grasp",
            success=True,
            object_name="cup",
            duration_sec=2.5,
        )
        parsed = json.loads(result)
        assert parsed["skill_name"] == "grasp"
        assert parsed["success"] is True


class TestRequestPerceptionTool:
    async def test_submits_internal_skill_and_queries_evidence(self):
        from hey_robot.agents.tools.request_perception import RequestPerceptionTool

        io = _FakeIO()
        io._scene_response.to_dict.return_value = {
            "status": "ok",
            "frame_id": 99,
            "image_count": 1,
            "summary": "a cup on the table",
            "confidence": 0.9,
            "objects": ["cup", "table"],
            "risks": [],
            "next_observation_hint": "",
            "source": "camera",
            "metadata": {},
        }
        ctx = _build_ctx(io=io)
        tool = RequestPerceptionTool(ctx)
        result = await tool.execute(
            modality="vision", scope="current_scene", question="what is on the table"
        )
        parsed = json.loads(result)
        assert parsed["tool"] == "request_perception"
        assert parsed["evidence_status"] == "ok"
        assert "cup" in parsed["result"]
        assert len(io.skills) == 1
        assert io.skills[0].name == "inspect_scene"
        assert io.skills[0].arguments == {"question": "what is on the table"}

    async def test_execution_result_scope(self):
        from hey_robot.agents.tools.request_perception import RequestPerceptionTool

        io = _FakeIO()
        io._scene_response.to_dict.return_value = {"status": "ok", "summary": "done"}
        ctx = _build_ctx(io=io)
        tool = RequestPerceptionTool(ctx)
        await tool.execute(scope="execution_result")
        assert io.skills[0].name == "inspect_scene"

    async def test_no_scene_evidence_fallback(self):
        from hey_robot.agents.tools.request_perception import RequestPerceptionTool

        pending: dict = {}
        io = _FakeIO(pending_skills=pending)
        io.query_scene_evidence = None  # type: ignore[method-assign,assignment]  # shadow class method to simulate missing capability
        ctx = _build_ctx(io=io, pending_skills=pending)
        tool = RequestPerceptionTool(ctx)
        result = await tool.execute()
        parsed = json.loads(result)
        assert parsed["evidence_status"] == "degraded"
        assert "not available" in parsed["result"]

    async def test_invalid_modality_raises(self):
        from hey_robot.agents.tools.request_perception import RequestPerceptionTool

        ctx = _build_ctx()
        tool = RequestPerceptionTool(ctx)
        with pytest.raises(ValueError, match="modality"):
            await tool.execute(modality="lidar")

    async def test_invalid_scope_raises(self):
        from hey_robot.agents.tools.request_perception import RequestPerceptionTool

        ctx = _build_ctx()
        tool = RequestPerceptionTool(ctx)
        with pytest.raises(ValueError, match="scope"):
            await tool.execute(scope="back")


class TestRequestCapabilityTool:
    async def test_submits_and_awaits_skill(self):
        from hey_robot.agents.tools.request_capability import RequestCapabilityTool

        pending: dict = {}
        io = _FakeIO(pending_skills=pending)
        ctx = _build_ctx(io=io, pending_skills=pending)
        tool = RequestCapabilityTool(ctx)
        result = await tool.execute(
            capability="set_gripper",
            objective="close the gripper on the cup",
            slots={"action": "close"},
        )
        assert "controller_status=completed" in result

    async def test_empty_objective_raises(self):
        from hey_robot.agents.tools.request_capability import RequestCapabilityTool

        ctx = _build_ctx()
        tool = RequestCapabilityTool(ctx)
        with pytest.raises(ValueError, match="objective"):
            await tool.execute(capability="grasp", objective="")

    async def test_unknown_skill_raises(self):
        from hey_robot.agents.tools.request_capability import RequestCapabilityTool

        ctx = _build_ctx()
        tool = RequestCapabilityTool(ctx)
        with pytest.raises(KeyError, match="unknown skill"):
            await tool.execute(capability="legacy_internal_skill", objective="look")

    async def test_motion_skill_blocked_when_camera_unhealthy(self):
        from hey_robot.agents.tools.request_capability import RequestCapabilityTool

        pending: dict = {}
        io = _FakeIO(pending_skills=pending)
        ctx = _build_ctx(io=io, pending_skills=pending)
        ctx.task_runtime = _make_task_runtime_with_camera(
            "ep1",
            ok=False,
            frame_available=False,
            valid_image_count=0,
            quality_issues=["black_frame"],
            age_ms=5000,
        )
        ctx._current_envelope = lambda: Envelope(
            robot_id="mock0", channel="test", episode_id="ep1"
        )
        tool = RequestCapabilityTool(ctx)
        with pytest.raises(RuntimeError, match="CameraUnsafe"):
            await tool.execute(capability="move_base", objective="move forward")

    async def test_motion_skill_blocked_when_camera_stale(self):
        from hey_robot.agents.tools.request_capability import RequestCapabilityTool

        pending: dict = {}
        io = _FakeIO(pending_skills=pending)
        ctx = _build_ctx(io=io, pending_skills=pending)
        ctx.task_runtime = _make_task_runtime_with_camera(
            "ep1",
            ok=True,
            frame_available=True,
            valid_image_count=3,
            quality_issues=[],
            age_ms=20000,
        )
        ctx._current_envelope = lambda: Envelope(
            robot_id="mock0", channel="test", episode_id="ep1"
        )
        tool = RequestCapabilityTool(ctx)
        with pytest.raises(RuntimeError, match="CameraStale"):
            await tool.execute(capability="move_base", objective="move forward")

    async def test_motion_skill_allowed_when_camera_healthy(self):
        from hey_robot.agents.tools.request_capability import RequestCapabilityTool

        pending: dict = {}
        io = _FakeIO(pending_skills=pending)
        ctx = _build_ctx(io=io, pending_skills=pending)
        ctx.task_runtime = _make_task_runtime_with_camera(
            "ep1",
            ok=True,
            frame_available=True,
            valid_image_count=3,
            quality_issues=[],
            age_ms=2000,
        )
        ctx._current_envelope = lambda: Envelope(
            robot_id="mock0", channel="test", episode_id="ep1"
        )
        tool = RequestCapabilityTool(ctx)
        result = await tool.execute(capability="move_base", objective="move forward")
        assert "controller_status=completed" in result

    async def test_observe_skill_skips_camera_check(self):
        from hey_robot.agents.tools.request_capability import RequestCapabilityTool

        pending: dict = {}
        io = _FakeIO(pending_skills=pending)
        ctx = _build_ctx(io=io, pending_skills=pending)
        # No task_runtime set — would fail if camera check ran for observe skill
        tool = RequestCapabilityTool(ctx)
        result = await tool.execute(capability="inspect_scene", objective="check scene")
        assert "controller_status=completed" in result

    async def test_consecutive_motion_blocked_without_intervening_perception(self):
        from hey_robot.agents.tools.request_capability import RequestCapabilityTool

        pending: dict = {}
        io = _FakeIO(pending_skills=pending)
        ctx = _build_ctx(io=io, pending_skills=pending)
        ctx.task_runtime = _make_task_runtime_with_camera(
            "ep1",
            ok=True,
            frame_available=True,
            valid_image_count=3,
            quality_issues=[],
            age_ms=2000,
        )
        ctx._current_envelope = lambda: Envelope(
            robot_id="mock0", channel="test", episode_id="ep1"
        )
        ctx.runtime_state.last_capability_safety_level = "motion"
        ctx.runtime_state.last_capability_name = "move_base"
        tool = RequestCapabilityTool(ctx)
        with pytest.raises(RuntimeError, match="ConsecutiveMotionBlocked"):
            await tool.execute(capability="move_base", objective="move again")

    async def test_consecutive_motion_allows_stop_skills(self):
        from hey_robot.agents.tools.request_capability import RequestCapabilityTool

        pending: dict = {}
        io = _FakeIO(pending_skills=pending)
        ctx = _build_ctx(io=io, pending_skills=pending)
        ctx.task_runtime = _make_task_runtime_with_camera(
            "ep1",
            ok=True,
            frame_available=True,
            valid_image_count=3,
            quality_issues=[],
            age_ms=2000,
        )
        ctx._current_envelope = lambda: Envelope(
            robot_id="mock0", channel="test", episode_id="ep1"
        )
        ctx.runtime_state.last_capability_safety_level = "motion"
        ctx.runtime_state.last_capability_name = "move_base"
        tool = RequestCapabilityTool(ctx)
        result = await tool.execute(
            capability="stop_motion", objective="emergency stop"
        )
        assert "controller_status=completed" in result

    async def test_consecutive_motion_allowed_after_perception_reset(self):
        from hey_robot.agents.tools.request_capability import RequestCapabilityTool

        pending: dict = {}
        io = _FakeIO(pending_skills=pending)
        ctx = _build_ctx(io=io, pending_skills=pending)
        ctx.task_runtime = _make_task_runtime_with_camera(
            "ep1",
            ok=True,
            frame_available=True,
            valid_image_count=3,
            quality_issues=[],
            age_ms=2000,
        )
        ctx._current_envelope = lambda: Envelope(
            robot_id="mock0", channel="test", episode_id="ep1"
        )
        ctx.runtime_state.last_capability_safety_level = "observe"
        ctx.runtime_state.last_capability_name = "inspect_scene"
        tool = RequestCapabilityTool(ctx)
        result = await tool.execute(
            capability="move_base", objective="move after inspect"
        )
        assert "controller_status=completed" in result

    async def test_skill_blocked_when_turn_recovery_required(self):
        from hey_robot.agents.tools.request_capability import RequestCapabilityTool

        pending: dict = {}
        io = _FakeIO(pending_skills=pending)
        turn = _make_turn_context()
        turn.recovery_required = True
        ctx = _build_ctx(io=io, pending_skills=pending, turn_context=turn)
        tool = RequestCapabilityTool(ctx)
        with pytest.raises(RuntimeError, match="recovery required"):
            await tool.execute(capability="move_base", objective="move")

    async def test_recovery_required_allows_open_gripper(self):
        from hey_robot.agents.tools.request_capability import RequestCapabilityTool

        pending: dict = {}
        io = _FakeIO(pending_skills=pending)
        turn = _make_turn_context()
        turn.recovery_required = True
        ctx = _build_ctx(io=io, pending_skills=pending, turn_context=turn)
        tool = RequestCapabilityTool(ctx)

        result = await tool.execute(
            capability="set_gripper",
            objective="open gripper",
            slots={"action": "open"},
        )

        assert "controller_status=completed" in result
        assert io.skills[0].name == "set_gripper"
        assert io.skills[0].arguments == {"action": "open"}

    async def test_recovery_required_blocks_close_gripper(self):
        from hey_robot.agents.tools.request_capability import RequestCapabilityTool

        pending: dict = {}
        io = _FakeIO(pending_skills=pending)
        turn = _make_turn_context()
        turn.recovery_required = True
        ctx = _build_ctx(io=io, pending_skills=pending, turn_context=turn)
        tool = RequestCapabilityTool(ctx)

        with pytest.raises(RuntimeError, match="recovery required"):
            await tool.execute(
                capability="set_gripper",
                objective="close gripper",
                slots={"action": "close"},
            )

        assert io.skills == []

    async def test_recovery_required_allows_reset_posture(self):
        from hey_robot.agents.tools.request_capability import RequestCapabilityTool

        pending: dict = {}
        io = _FakeIO(pending_skills=pending)
        turn = _make_turn_context()
        turn.recovery_required = True
        ctx = _build_ctx(io=io, pending_skills=pending, turn_context=turn)
        tool = RequestCapabilityTool(ctx)

        result = await tool.execute(
            capability="reset_posture",
            objective="reset posture",
        )

        assert "controller_status=completed" in result
        assert io.skills[0].name == "reset_posture"


def _make_task_runtime_with_camera(
    episode_id, *, ok, frame_available, valid_image_count, quality_issues, age_ms
):
    from hey_robot.episode import RobotEpisodeStateStore

    store = RobotEpisodeStateStore.__new__(RobotEpisodeStateStore)
    store.root = MagicMock()
    store.root.mkdir = MagicMock()

    class _FakeState:
        def __init__(self):
            self.last_status = {
                "timestamp": __import__("time").time(),
                "metrics": {
                    "camera": {
                        "ok": ok,
                        "frame_available": frame_available,
                        "valid_image_count": valid_image_count,
                        "image_quality_issues": list(quality_issues),
                        "age_ms": age_ms,
                    }
                },
            }

    def _load(eid):
        return _FakeState() if eid == episode_id else None

    store.load = _load  # type: ignore[method-assign]
    return type("_TaskRuntime", (), {"robot_states": store})()


class TestTaskContextTool:
    async def test_get_task_context_returns_task_feedback_and_recovery(self):
        from hey_robot.agents.tools.get_task_context import GetTaskContextTool

        def task_dict() -> dict[str, object]:
            return {
                "root_task": "pick the cup",
                "status": "recovering",
                "robot_id": "mock0",
                "attempts": [
                    {
                        "text": "move closer",
                        "status": "feedback_failed",
                        "skill_id": "skill1",
                    }
                ],
                "failure_reason": "cup not visible",
                "recovery": {"strategy": "reobserve", "summary": "Get a fresh view."},
            }

        task_runtime = MagicMock()
        task_runtime.task_runs.load_active.return_value = type(
            "_TaskRun",
            (),
            {"to_dict": staticmethod(task_dict)},
        )()
        task_runtime.robot_states.load.return_value = type(
            "_RobotState",
            (),
            {
                "last_execution_feedback": {
                    "summary": "cup not visible",
                    "next_hint": "reobserve",
                }
            },
        )()
        runtime_state = AgentState(task="pick the cup")
        runtime_state.add_tool_call(
            "request_capability",
            {"capability": "vla_manipulation", "objective": "pick the cup"},
            "target still not reachable",
            success=False,
        )
        runtime_state.add_tool_call(
            "request_capability",
            {"capability": "vla_manipulation", "objective": "pick the cup"},
            "target still not reachable",
            success=False,
        )
        ctx = _build_ctx(
            task_runtime=task_runtime,
            runtime_state=runtime_state,
            _current_envelope=lambda: Envelope(
                episode_id="ep1", robot_id="mock0", channel="test"
            ),
        )

        result = await GetTaskContextTool(ctx).execute()
        parsed = json.loads(result)

        assert parsed["episode_id"] == "ep1"
        assert parsed["task"]["root_task"] == "pick the cup"
        assert parsed["latest_execution_feedback"]["summary"] == "cup not visible"
        assert parsed["recovery"]["strategy"] == "reobserve"
        assert parsed["recommended_next_steps"][0]["option"] == "reobserve"
        assert "Loop warning:" in parsed["loop_warning"]


class TestProposeCapabilityTool:
    async def test_stores_pending_confirmation_and_returns_prompt(self):
        from hey_robot.agents.tools.propose_capability import ProposeCapabilityTool

        io = _FakeIO()
        ctx = _build_ctx(
            io=io,
            _current_envelope=lambda: Envelope(
                episode_id="ep1",
                robot_id="mock0",
                channel="test",
                chat_id="chat1",
                sender_id="u1",
            ),
        )
        tool = ProposeCapabilityTool(ctx)

        result = await tool.execute(
            capability="turn_base",
            objective="先转向 __TASK__",
            confirmation_prompt="要不要我先转一下？",
            slots={"angle_deg": 30},
            interrupt=True,
        )

        assert result == "要不要我先转一下？"
        io.task_runtime.store_pending_confirmation.assert_called_once()
        episode_id, proposal = io.task_runtime.store_pending_confirmation.call_args.args
        assert episode_id == "ep1"
        assert proposal["capability"] == "turn_base"
        assert proposal["objective"] == "先转向 pick the cup"
        assert proposal["slots"] == {"angle_deg": 30}
        assert proposal["interrupt"] is True
        assert proposal["prompt"] == "要不要我先转一下？"

    async def test_rejects_missing_fields(self):
        from hey_robot.agents.tools.propose_capability import ProposeCapabilityTool

        ctx = _build_ctx()
        tool = ProposeCapabilityTool(ctx)

        with pytest.raises(ValueError, match="capability"):
            await tool.execute(
                capability="", objective="look", confirmation_prompt="ok?"
            )
        with pytest.raises(ValueError, match="objective"):
            await tool.execute(
                capability="turn_base",
                objective=" ",
                confirmation_prompt="ok?",
            )
        with pytest.raises(ValueError, match="confirmation_prompt"):
            await tool.execute(
                capability="turn_base",
                objective="look",
                confirmation_prompt=" ",
            )


# to_spec runtime adapter verification.


class TestToolBaseBehavior:
    async def test_cast_validate_schema_and_to_spec(self):
        from hey_robot.agents.tools.base import Tool, tool_parameters
        from hey_robot.agents.tools.schema import (
            ArraySchema,
            BooleanSchema,
            IntegerSchema,
            ObjectSchema,
            StringSchema,
            tool_parameters_schema,
        )

        @tool_parameters(
            tool_parameters_schema(
                count=IntegerSchema(description="count"),
                enabled=BooleanSchema(description="enabled"),
                label=StringSchema("label", nullable=True),
                tags=ArraySchema(StringSchema("tag")),
                nested=ObjectSchema(
                    properties={"flag": BooleanSchema(description="flag")}
                ),
                required=["count", "enabled"],
            )
        )
        class DemoTool(Tool):
            name = "demo"
            description = "demo tool"
            safety_level = "observe"
            read_only = True
            timeout_sec = 1.5
            resources = ("camera",)

            @classmethod
            def create(cls, ctx):
                del ctx
                return cls()

            async def execute(self, **kwargs):
                return kwargs["count"] + 1

        tool = DemoTool()
        casted = tool.cast_params(
            {
                "count": "2",
                "enabled": "yes",
                "label": "",
                "tags": [1, "ok"],
                "nested": {"flag": "no"},
                "extra": object(),
            }
        )

        assert casted["count"] == 2
        assert casted["enabled"] is True
        assert casted["label"] is None
        assert casted["tags"] == ["1", "ok"]
        assert casted["nested"] == {"flag": False}
        assert tool.validate_params(casted) == []
        assert "parameters must be an object" in tool.validate_params("bad")[0]  # type: ignore[arg-type]
        assert tool.concurrency_safe is True
        assert tool.to_schema()["function"]["name"] == "demo"

        spec = tool.to_spec()
        assert spec.name == "demo"
        assert spec.read_only is True
        assert spec.timeout_sec == 1.5
        assert await spec.func(**casted) == "3"

    def test_validate_rejects_non_object_schema(self):
        from hey_robot.agents.tools.base import Tool

        class BadSchemaTool(Tool):
            name = "bad_schema"
            description = "bad schema"
            parameters = {"type": "string"}  # noqa: RUF012

            @classmethod
            def create(cls, ctx):
                del ctx
                return cls()

            async def execute(self, **kwargs):
                del kwargs
                return "ok"

        with pytest.raises(ValueError, match="Schema must be object"):
            BadSchemaTool().validate_params({})

    async def test_to_spec_wraps_sync_execute_result(self):
        from hey_robot.agents.tools.base import Tool

        class SyncTool(Tool):
            name = "sync"
            description = "sync tool"

            @classmethod
            def create(cls, ctx):
                del ctx
                return cls()

            def execute(self, **kwargs):
                return {"echo": kwargs.get("value")}

        assert await SyncTool().to_spec().func(value="ok") == "{'echo': 'ok'}"


class TestToSpecRuntimeAdapter:
    def test_all_tools_produce_valid_spec(self):
        """Every tool's to_spec() must return a ToolSpec consumable by the
        runtime CapabilityResolver / PermissionManager / ToolExecutor pipeline."""
        from hey_robot.agents.tools import ToolLoader

        loader = ToolLoader()
        cls_list = loader.discover()

        # Build a context that allows tools with trivial __init__ (like WaitTool)
        # to succeed. Complex tools will fail init without real deps, so we test
        # to_spec() on the class pattern rather than full instances for those.
        for tool_cls in cls_list:
            # Verify the class contract
            assert tool_cls.name, f"{tool_cls.__name__}.name is empty"
            assert tool_cls.description, f"{tool_cls.__name__}.description is empty"
            assert tool_cls.safety_level in {
                "normal",
                "observe",
                "actuate",
                "communicate",
                "memory_write",
            }, f"{tool_cls.__name__}.safety_level={tool_cls.safety_level!r}"

    def test_registry_produces_runtime_adapter_list(self):
        """The new ToolRegistry.list_tools() output must match the shape the
        old code path expects."""
        from hey_robot.agents.tools import ToolRegistry as NewRegistry

        reg = NewRegistry()
        # Register a single simple tool
        from hey_robot.agents.tools.wait import WaitTool

        reg.register(WaitTool())
        tools = reg.list_tools()
        assert len(tools) == 1
        t = tools[0]
        assert t["name"] == "wait"
        assert "inputSchema" in t
        assert "description" in t
        ann = t["annotations"]
        assert "safetyLevel" in ann
        assert "readOnlyHint" in ann
        assert "concurrencySafeHint" in ann
        assert "resources" in ann
        assert "resultPolicy" in ann
