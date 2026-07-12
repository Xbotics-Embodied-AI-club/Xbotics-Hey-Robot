"""Cover ContextBuilder and edge cases."""

from __future__ import annotations

from hey_robot.cognition.autonomous.context_builder import (
    build_context,
)
from hey_robot.protocol import (
    ActionSnapshot,
    BudgetState,
    DeliberationRequest,
    Envelope,
    GoalSnapshot,
    RobotObservation,
    RobotStatus,
    SkillResult,
    SuccessCriterion,
)


def _base_request(**overrides) -> DeliberationRequest:
    goal = GoalSnapshot(
        "g1",
        1,
        "t1",
        "c1",
        "hash1",
        "test",
        (SuccessCriterion("c1", "evidence_present", "r1", "observed", "scene", 300),),
        "active",
    )
    kwargs = {
        "envelope": Envelope(robot_id="r1"),
        "deliberation_id": "d1",
        "trigger_event_id": "t1",
        "goal": goal,
        "robot_status": None,
        "robot_observation": None,
        "actions": (),
        "evidence": (),
        "latest_skill_result": None,
        "budget_state": BudgetState(0, 0, 0, 100.0),
    }
    kwargs.update(overrides)
    return DeliberationRequest(**kwargs)


def test_build_minimal_context() -> None:
    result = build_context(_base_request(), evaluation_text="INCONCLUSIVE")
    assert result.failure is None
    assert len(result.messages) == 1
    content = result.messages[0].content
    assert "test" in content
    assert "INCONCLUSIVE" in content


def test_build_context_with_status_and_observation() -> None:
    status = RobotStatus(Envelope(robot_id="r1"), state="idle", location_id="room:lab")
    obs = RobotObservation(Envelope(), frame_id=5)
    result = build_context(
        _base_request(robot_status=status, robot_observation=obs),
        evaluation_text="INCONCLUSIVE",
    )
    assert result.failure is None
    content = result.messages[0].content
    assert "room:lab" in content
    assert "frame_id" in content


def test_build_context_with_actions() -> None:
    actions = (
        ActionSnapshot(
            "s1", "d1", "observation", "inspect_scene", "look", {"q": "x"}, "completed"
        ),
        ActionSnapshot(
            "s2", "d2", "skill", "navigate_to", "go", {"t": "lab"}, "completed"
        ),
    )
    result = build_context(
        _base_request(actions=actions),
        evaluation_text="INCONCLUSIVE",
    )
    assert result.failure is None
    content = result.messages[0].content
    assert "inspect_scene" in content
    assert "navigate_to" in content


def test_build_context_with_skill_result() -> None:
    sr = SkillResult(Envelope(), "s1", status="completed", success=True, summary="done")
    result = build_context(
        _base_request(latest_skill_result=sr),
        evaluation_text="INCONCLUSIVE",
    )
    assert result.failure is None
    content = result.messages[0].content
    assert "done" in content


def test_build_context_satisfied_message() -> None:
    result = build_context(
        _base_request(),
        evaluation_text="CONTRACT SATISFIED: all done",
    )
    assert result.failure is None
    assert "SATISFIED" in result.messages[0].content


def test_build_context_rejects_evidence_over_budget() -> None:
    from hey_robot.protocol import EvidenceFact

    evidence = tuple(
        EvidenceFact(
            f"e{i}", "g1", "skill_result", "s1", 1.0, 1, "r1", "observed", "scene"
        )
        for i in range(100)
    )
    result = build_context(
        _base_request(evidence=evidence),
        evaluation_text="INCONCLUSIVE",
        max_evidence_items=8,
    )
    assert result.messages == ()
    assert result.failure is not None
    assert result.failure.code == "CONTEXT_BUDGET_EXCEEDED"
