"""Cover additional AutonomyStore branches."""

from __future__ import annotations

from pathlib import Path

import pytest

from hey_robot.cognition.autonomous.store import AutonomyStore


def _store(tmp_path: Path) -> AutonomyStore:
    return AutonomyStore(tmp_path / "autonomy.sqlite3")


def test_schema_version_mismatch_rejects(tmp_path: Path) -> None:
    import sqlite3 as _sql

    path = tmp_path / "bad.sqlite3"
    db = _sql.connect(str(path))
    db.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    db.execute("INSERT INTO schema_version VALUES (999)")
    db.commit()
    db.close()
    with pytest.raises(RuntimeError, match="schema version mismatch"):
        AutonomyStore(path)


def test_duplicate_goal_creation_rejected(tmp_path: Path) -> None:
    s = _store(tmp_path)
    assert s.create_goal(
        command_id="c1", goal_id="g1", robot_id="r1", snapshot={}, budgets={}
    )
    # Duplicate by goal_id
    assert not s.create_goal(
        command_id="c2", goal_id="g1", robot_id="r2", snapshot={}, budgets={}
    )


def test_duplicate_command_id_replay(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.create_goal(command_id="c1", goal_id="g1", robot_id="r1", snapshot={}, budgets={})
    result = s.command_result("c1")
    assert result is not None
    assert result["goal_id"] == "g1"
    # Non-existent
    assert s.command_result("nonexistent") is None


def test_budget_state_returns_none_for_missing_goal(tmp_path: Path) -> None:
    s = _store(tmp_path)
    assert s.budget_state("nonexistent", 100.0) is None


def test_gate_auto_creates_for_new_robot(tmp_path: Path) -> None:
    s = _store(tmp_path)
    gate = s.gate("new_robot")
    assert gate.robot_id == "new_robot"
    assert gate.state == "ready"
    assert gate.version == 0


def test_active_goal_for_robot_none_when_no_goal(tmp_path: Path) -> None:
    s = _store(tmp_path)
    assert s.active_goal_for_robot("nonexistent") is None


def test_goals_recent_handles_empty(tmp_path: Path) -> None:
    s = _store(tmp_path)
    assert s.goals_recent() == []


def test_actions_for_goal_empty(tmp_path: Path) -> None:
    s = _store(tmp_path)
    assert s.actions_for_goal("nonexistent") == []


def test_schedule_deliberation_rejects_wrong_status(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.create_goal(command_id="c1", goal_id="g1", robot_id="r1", snapshot={}, budgets={})
    # Goal is "pending" — schedule works
    assert s.schedule_deliberation(
        goal_id="g1",
        trigger_event_id="t1",
        deliberation_id="d1",
        request={},
        request_hash="h1",
    )
    # Second with same trigger should fail (unique constraint)
    assert not s.schedule_deliberation(
        goal_id="g1",
        trigger_event_id="t1",
        deliberation_id="d2",
        request={},
        request_hash="h2",
    )
    # After goal moved to non-pending/waiting, should fail
    s._db.execute("UPDATE goals SET status='completed' WHERE goal_id='g1'")
    s._db.commit()
    assert not s.schedule_deliberation(
        goal_id="g1",
        trigger_event_id="t2",
        deliberation_id="d3",
        request={},
        request_hash="h3",
    )


def test_accept_action_rejects_non_active_goal(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.create_goal(command_id="c1", goal_id="g1", robot_id="r1", snapshot={}, budgets={})
    # Goal is pending, not active
    assert not s.accept_action(
        goal_id="g1", deliberation_id="d1", skill_id="s1", payload={}
    )


def test_accept_action_rejects_duplicate_deliberation(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.create_goal(command_id="c1", goal_id="g1", robot_id="r1", snapshot={}, budgets={})
    s._db.execute("UPDATE goals SET status='active' WHERE goal_id='g1'")
    s._db.commit()
    assert s.accept_action(
        goal_id="g1", deliberation_id="d1", skill_id="s1", payload={}
    )
    assert not s.accept_action(
        goal_id="g1", deliberation_id="d1", skill_id="s2", payload={}
    )


def test_accept_action_rejects_duplicate_skill_id(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.create_goal(command_id="c1", goal_id="g1", robot_id="r1", snapshot={}, budgets={})
    s._db.execute("UPDATE goals SET status='active' WHERE goal_id='g1'")
    s._db.commit()
    assert s.accept_action(
        goal_id="g1", deliberation_id="d1", skill_id="s1", payload={}
    )
    assert not s.accept_action(
        goal_id="g1", deliberation_id="d2", skill_id="s1", payload={}
    )


def test_fail_goal_only_on_nonterminal(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.create_goal(command_id="c1", goal_id="g1", robot_id="r1", snapshot={}, budgets={})
    s.fail_goal("g1", "TEST_FAILURE")
    g = s.goal("g1")
    assert g is not None
    assert g["status"] == "failed"
    assert g["termination_reason"] == "TEST_FAILURE"


def test_cancel_goal_succeeds_for_waiting_goal_without_active_action(
    tmp_path: Path,
) -> None:
    s = _store(tmp_path)
    s.create_goal(command_id="c1", goal_id="g1", robot_id="r1", snapshot={}, budgets={})
    s._db.execute("UPDATE goals SET status='waiting' WHERE goal_id='g1'")
    s._db.commit()
    assert s.cancel_goal("g1")
    g = s.goal("g1")
    assert g is not None
    assert g["status"] == "cancelled"


def test_cancel_goal_succeeds_for_active(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.create_goal(command_id="c1", goal_id="g1", robot_id="r1", snapshot={}, budgets={})
    s._db.execute("UPDATE goals SET status='active' WHERE goal_id='g1'")
    s._db.commit()
    assert s.cancel_goal("g1")
    g = s.goal("g1")
    assert g is not None
    assert g["status"] == "cancelled"


def test_begin_stop_rejects_non_waiting_blocked(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.create_goal(command_id="c1", goal_id="g1", robot_id="r1", snapshot={}, budgets={})
    # Goal is pending
    assert not s.begin_stop(
        goal_id="g1",
        robot_id="r1",
        control_id="ctrl1",
        target_skill_id="s1",
        action="interrupt",
        reason="cancel",
        payload={},
    )


def test_begin_stop_duplicate_control_rejected(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.create_goal(command_id="c1", goal_id="g1", robot_id="r1", snapshot={}, budgets={})
    s._db.execute("UPDATE goals SET status='waiting' WHERE goal_id='g1'")
    s._db.commit()
    assert s.begin_stop(
        goal_id="g1",
        robot_id="r1",
        control_id="ctrl1",
        target_skill_id="s1",
        action="interrupt",
        reason="cancel",
        payload={},
    )
    assert not s.begin_stop(
        goal_id="g1",
        robot_id="r1",
        control_id="ctrl1",
        target_skill_id="s2",
        action="interrupt",
        reason="cancel",
        payload={},
    )


def test_cas_control_status_race(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.create_goal(command_id="c1", goal_id="g1", robot_id="r1", snapshot={}, budgets={})
    s._db.execute("UPDATE goals SET status='waiting' WHERE goal_id='g1'")
    s._db.commit()
    s.begin_stop(
        goal_id="g1",
        robot_id="r1",
        control_id="ctrl1",
        target_skill_id="s1",
        action="interrupt",
        reason="cancel",
        payload={},
    )
    assert s.cas_control_status("ctrl1", "persisted", "publishing")
    assert s.cas_control_status("ctrl1", "publishing", "published")
    # Wrong expected status → fail
    assert not s.cas_control_status("ctrl1", "persisted", "completed")


def test_control_returns_none_for_missing(tmp_path: Path) -> None:
    s = _store(tmp_path)
    assert s.control("nonexistent") is None


def test_action_returns_none_for_missing(tmp_path: Path) -> None:
    s = _store(tmp_path)
    assert s.action("nonexistent") is None


def test_active_action_for_goal_none(tmp_path: Path) -> None:
    s = _store(tmp_path)
    assert s.active_action_for_goal("nonexistent") is None


def test_terminal_skill_result_for_missing_action(tmp_path: Path) -> None:
    s = _store(tmp_path)
    result = s.terminal_skill_result(
        skill_id="nonexistent", status="completed", result_hash="h1", evidence=[]
    )
    assert result is None


def test_terminal_skill_result_conflict(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.create_goal(command_id="c1", goal_id="g1", robot_id="r1", snapshot={}, budgets={})
    s._db.execute("UPDATE goals SET status='active' WHERE goal_id='g1'")
    s._db.commit()
    s.accept_action(goal_id="g1", deliberation_id="d1", skill_id="s1", payload={})
    # First terminal result
    r1 = s.terminal_skill_result(
        skill_id="s1", status="completed", result_hash="h1", evidence=[]
    )
    assert r1 == "g1"
    # Second with same hash → replay
    r2 = s.terminal_skill_result(
        skill_id="s1", status="completed", result_hash="h1", evidence=[]
    )
    assert r2 == "g1"
    # Different hash → conflict
    r3 = s.terminal_skill_result(
        skill_id="s1", status="completed", result_hash="h2", evidence=[]
    )
    assert r3 == "conflict"


def test_mark_skill_result_conflict_for_missing(tmp_path: Path) -> None:
    s = _store(tmp_path)
    result = s.mark_skill_result_conflict("nonexistent")
    assert result is None


def test_reconcile_rejects_non_unknown_action(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.create_goal(command_id="c1", goal_id="g1", robot_id="r1", snapshot={}, budgets={})
    s._db.execute("UPDATE goals SET status='active' WHERE goal_id='g1'")
    s._db.commit()
    s.accept_action(goal_id="g1", deliberation_id="d1", skill_id="s1", payload={})
    # Action is "persisted", not "unknown"
    assert not s.reconcile_unknown_action(
        reconcile_id="r1",
        robot_id="r1",
        skill_id="s1",
        operator_id="op",
        status_payload={},
    )


def test_reconcile_rejects_non_uncertain_gate(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.create_goal(command_id="c1", goal_id="g1", robot_id="r1", snapshot={}, budgets={})
    s._db.execute("UPDATE goals SET status='active' WHERE goal_id='g1'")
    s._db.commit()
    s.accept_action(goal_id="g1", deliberation_id="d1", skill_id="s1", payload={})
    s.cas_action_status(skill_id="s1", expected="persisted", status="unknown")
    # Gate is still "ready", not "uncertain"
    assert not s.reconcile_unknown_action(
        reconcile_id="r1",
        robot_id="r1",
        skill_id="s1",
        operator_id="op",
        status_payload={},
    )


def test_append_evidence(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.create_goal(command_id="c1", goal_id="g1", robot_id="r1", snapshot={}, budgets={})
    fact = {
        "evidence_id": "e1",
        "goal_id": "g1",
        "source_kind": "skill_result",
        "source_id": "s1",
    }
    s.append_evidence("g1", [fact])
    evidence = s.evidence_for_goal("g1")
    assert len(evidence) == 1
    assert evidence[0]["evidence_id"] == "e1"


def test_recover_publishing_with_no_actions(tmp_path: Path) -> None:
    s = _store(tmp_path)
    interrupted = s.recover_publishing()
    assert interrupted == []
