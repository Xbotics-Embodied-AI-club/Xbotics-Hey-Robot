from pathlib import Path

from hey_robot.cognition.autonomous.store import AutonomyStore


def test_control_terminal_releases_gate_only_on_confirmed_idle(tmp_path: Path) -> None:
    store = AutonomyStore(tmp_path / "autonomy.sqlite3")
    assert store.create_goal(
        command_id="command", goal_id="goal", robot_id="robot", snapshot={}, budgets={}
    )
    store._db.execute("UPDATE goals SET status='waiting' WHERE goal_id='goal'")
    store._db.commit()
    assert store.begin_stop(
        goal_id="goal",
        robot_id="robot",
        control_id="control",
        target_skill_id="skill",
        action="interrupt",
        reason="cancel",
        payload={},
    )
    assert (
        store.terminal_control(
            control_id="control",
            status="completed",
            result_hash="hash",
            idle_confirmed=True,
        )
        == "goal"
    )
    assert store.gate("robot")["state"] == "ready"
    assert store.goal("goal")["status"] == "cancelled"


def test_publishing_recovery_sets_unknown_gate_and_blocks_goal(tmp_path: Path) -> None:
    store = AutonomyStore(tmp_path / "autonomy.sqlite3")
    assert store.create_goal(
        command_id="command", goal_id="goal", robot_id="robot", snapshot={}, budgets={}
    )
    store._db.execute("UPDATE goals SET status='active' WHERE goal_id='goal'")
    store._db.commit()
    assert store.accept_action(
        goal_id="goal", deliberation_id="d", skill_id="skill", payload={}
    )
    assert store.cas_action_status(
        skill_id="skill", expected="persisted", status="publishing"
    )
    assert store.recover_publishing() == ["skill"]
    assert store.action("skill")["status"] == "unknown"
    assert store.gate("robot")["state"] == "uncertain"
    assert store.goal("goal")["status"] == "blocked"


def test_explicit_reconcile_releases_only_unknown_action_and_uncertain_gate(
    tmp_path: Path,
) -> None:
    store = AutonomyStore(tmp_path / "autonomy.sqlite3")
    assert store.create_goal(
        command_id="command", goal_id="goal", robot_id="robot", snapshot={}, budgets={}
    )
    store._db.execute("UPDATE goals SET status='active' WHERE goal_id='goal'")
    store._db.commit()
    assert store.accept_action(
        goal_id="goal", deliberation_id="d", skill_id="skill", payload={}
    )
    assert store.cas_action_status(
        skill_id="skill", expected="persisted", status="publishing"
    )
    store.recover_publishing()
    assert store.reconcile_unknown_action(
        reconcile_id="r1",
        robot_id="robot",
        skill_id="skill",
        operator_id="operator",
        status_payload={"state": "idle", "skill_id": None},
    )
    assert store.action("skill")["status"] == "reconciled_idle"
    assert store.gate("robot")["state"] == "ready"
