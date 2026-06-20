from __future__ import annotations

import time

import pytest

from hey_robot.agents.runtime.agent_run import AgentRunReader, AgentRunRecorder
from hey_robot.agents.session import AgentTurnSessions
from hey_robot.agents.skill_state import SkillPhase, SkillStateMachine
from hey_robot.agents.task_safety import evaluate_skill_request, evaluate_user_task
from hey_robot.agents.types import AgentCoreResult
from hey_robot.protocol import Envelope, SkillIntent, SkillResult, UserTurn
from hey_robot.skills.catalog import RobotSkillSpec


def test_agent_turn_sessions_dedupe_trim_lease_and_status() -> None:
    sessions = AgentTurnSessions(max_seen_messages=3, seen_trim_size=2)
    turns = [
        UserTurn(
            envelope=Envelope(
                trace_id=f"tr{i}", message_id=f"msg{i}", robot_id="robot1"
            ),
            text="move",
        )
        for i in range(4)
    ]

    assert sessions.is_duplicate_or_remember(turns[0]) is False
    assert sessions.is_duplicate_or_remember(turns[0]) is True
    for turn in turns[1:]:
        assert sessions.is_duplicate_or_remember(turn) is False
    assert sessions.is_duplicate_or_remember(turns[0]) is False

    assert sessions.active_robot_lease(None, timeout_sec=1.0) is None
    sessions.lease_robot("robot1", "skill1")
    lease = sessions.active_robot_lease("robot1", timeout_sec=1.0)
    assert lease is not None
    assert lease[0] == "skill1"
    sessions.robot_leases["robot1"] = ("skill1", time.time() - 10)
    assert sessions.active_robot_lease("robot1", timeout_sec=0.01) is None

    result = AgentCoreResult(
        reply_text="submitted",
        task_finished=False,
        skill_submitted=True,
        metadata={"skill_id": "skill2"},
    )
    state = sessions.record_turn_result(
        turn=turns[-1], result=result, baseline_frame_id=4
    )
    assert state.skill_id == "skill2"
    assert state.status == "responded"
    updated = sessions.update_turn_status(
        turns[-1].envelope.trace_id, "feedback_pending"
    )
    assert updated is not None
    assert updated.status == "feedback_pending"
    assert sessions.update_turn_status("missing", "done") is None


def test_skill_state_machine_validates_objectives_and_feedback_lifecycle() -> None:
    machine = SkillStateMachine()

    with pytest.raises(ValueError, match="non-empty"):
        machine.submit(
            SkillIntent(envelope=Envelope(), skill_id="skill1", objective=" ")
        )
    with pytest.raises(RuntimeError, match="no skill"):
        machine.mark_feedback_pending()

    snapshot = machine.submit(
        SkillIntent(
            envelope=Envelope(),
            skill_id="skill1",
            objective=" grasp bottle ",
            feedback_mode="operator",
        )
    )
    assert snapshot.phase == SkillPhase.ISSUED
    assert snapshot.objective == "grasp bottle"
    assert snapshot.needs_feedback is False

    ignored = machine.observe_result(
        SkillResult(envelope=Envelope(), skill_id="other", status="completed")
    )
    assert ignored.phase == SkillPhase.ISSUED
    assert machine.observe_result(
        SkillResult(envelope=Envelope(), skill_id="skill1", status="executing")
    ).phase == (SkillPhase.EXECUTING)
    completed = machine.observe_result(
        SkillResult(envelope=Envelope(), skill_id="skill1", status="completed")
    )
    assert completed.needs_feedback is True
    assert machine.mark_feedback_pending().phase == SkillPhase.FEEDBACK_PENDING
    assert (
        machine.mark_feedback_received("grasp verified").phase == SkillPhase.CONFIRMED
    )
    assert machine.snapshot.feedback_summary == "grasp verified"
    assert machine.reset().phase == SkillPhase.IDLE


def test_skill_state_machine_maps_terminal_and_unknown_statuses() -> None:
    machine = SkillStateMachine()

    with pytest.raises(RuntimeError, match="no skill"):
        machine.mark_feedback_received("nothing active")

    machine.submit(
        SkillIntent(
            envelope=Envelope(),
            skill_id="skill1",
            objective="move",
            feedback_mode="none",
        )
    )

    assert machine.observe_result(
        SkillResult(envelope=Envelope(), skill_id="skill1", status="accepted")
    ).phase == (SkillPhase.ACCEPTED)
    failed = machine.observe_result(
        SkillResult(
            envelope=Envelope(),
            skill_id="skill1",
            status="failed",
            error="gripper jammed",
        )
    )
    assert failed.phase == SkillPhase.FAILED
    assert failed.error == "gripper jammed"
    assert failed.is_terminal is True

    assert machine.observe_result(
        SkillResult(envelope=Envelope(), skill_id="skill1", status="interrupted")
    ).phase == (SkillPhase.INTERRUPTED)
    unknown = machine.observe_result(
        SkillResult(envelope=Envelope(), skill_id="skill1", status="unexpected")
    )
    assert unknown.phase == SkillPhase.ACCEPTED
    assert unknown.needs_feedback is False


def test_task_safety_respects_disabled_settings_and_voice_motion_rules() -> None:
    disabled = {"task_safety": {"enabled": False}}
    assert (
        evaluate_user_task("open the door", channel="voice", settings=disabled).allowed
        is True
    )

    voice_move = evaluate_user_task("move forward", channel="voice", settings={})
    assert voice_move.allowed is False
    assert voice_move.rule == "voice_motion_confirmation"

    text_move = evaluate_user_task("move forward", channel="web", settings={})
    assert text_move.allowed is True

    contract = RobotSkillSpec(
        name="move_base",
        description="move the mobile base",
        category="mobile_base",
        safety_level="motion",
    )
    skill_decision = evaluate_skill_request(
        capability="move_base",
        objective="approach the desk",
        contract=contract,
        task="bring bottle",
        channel="voice",
        settings={},
    )
    assert skill_decision.allowed is False
    assert skill_decision.rule == "voice_motion_confirmation"
    confirmed_skill_decision = evaluate_skill_request(
        capability="move_base",
        objective="approach the desk",
        contract=contract,
        task="bring bottle",
        channel="voice",
        settings={},
        confirmed=True,
    )
    assert confirmed_skill_decision.allowed is True

    observe_contract = RobotSkillSpec(
        name="inspect_scene",
        description="inspect with camera",
        category="camera",
        safety_level="observe",
    )
    assert (
        evaluate_skill_request(
            capability="inspect_scene",
            objective="look at the desk",
            contract=observe_contract,
            task="what is on the desk",
            channel="voice",
            settings={},
        ).allowed
        is True
    )


def test_agent_run_reader_handles_missing_and_explicit_run_ids(tmp_path) -> None:
    reader = AgentRunReader(tmp_path)
    assert reader.list_agent_runs() == []
    assert reader.recovery_summary() == {"agent_run_id": None, "has_agent_run": False}
    assert (
        AgentRunReader(tmp_path, agent_run_id="missing").read_jsonl("missing.jsonl")
        == []
    )
    with pytest.raises(ValueError, match="agent_run_id is required"):
        reader.read_jsonl("agent_steps.jsonl")

    recorder = AgentRunRecorder(tmp_path, agent_run_id="run1")
    recorder.record_transcript("user", "bring the bottle", channel="web")
    recorder.record_decision(
        task="bring the bottle",
        robot_state="idle",
        decision={"tool": "request_capability"},
        result={"success": True},
    )

    reader = AgentRunReader(tmp_path)
    assert reader.list_agent_runs() == ["run1"]
    assert reader.read_jsonl("transcript.jsonl", agent_run_id="run1")[0][
        "metadata"
    ] == {"channel": "web"}
    summary = reader.recovery_summary()
    assert summary["agent_run_id"] == "run1"
    assert summary["has_agent_run"] is True
    assert summary["agent_step_count"] == 1
