import time
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
    assert store.gate("robot").state == "ready"
    assert store.goal("goal")["status"] == "cancelled"


def test_conflicting_control_result_is_rejected(tmp_path: Path) -> None:
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
            result_hash="first",
            idle_confirmed=True,
        )
        == "goal"
    )
    assert (
        store.terminal_control(
            control_id="control",
            status="completed",
            result_hash="conflict",
            idle_confirmed=True,
        )
        == "conflict"
    )


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
    assert store.gate("robot").state == "uncertain"
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
    assert store.gate("robot").state == "ready"


def test_goal_origin_persists_owner_and_interaction_metadata(tmp_path: Path) -> None:
    store = AutonomyStore(tmp_path / "autonomy.sqlite3")
    assert store.create_goal(
        command_id="cmd-owner",
        goal_id="goal-owner",
        robot_id="robot-1",
        snapshot={"task_id": "task-owner", "objective": "inspect"},
        budgets={},
        owner_principal_id="owner-a",
        origin_interaction_id="interaction-a",
        origin_channel="web",
        created_by="sender-a",
    )

    goal = store.goal("goal-owner")
    assert goal is not None
    assert store.goal_owner("goal-owner") == "owner-a"
    assert goal["origin_interaction_id"] == "interaction-a"
    assert goal["origin_channel"] == "web"
    view = store.goal_view("goal-owner")
    assert view is not None
    assert view["owner_principal_id"] == "owner-a"
    assert view["action_count"] == 0


def test_human_confirmation_wake_condition_reopens_goal_for_scheduling(
    tmp_path: Path,
) -> None:
    store = AutonomyStore(tmp_path / "autonomy.sqlite3")
    assert store.create_goal(
        command_id="cmd-wait",
        goal_id="goal-wait",
        robot_id="robot-1",
        snapshot={},
        budgets={},
    )
    store._db.execute("UPDATE goals SET status='active' WHERE goal_id='goal-wait'")
    store._db.commit()
    assert store.create_wake_condition(
        condition_id="condition-1",
        goal_id="goal-wait",
        kind="human_confirmation",
        expected_payload={"question": "continue?"},
        expires_at=time.time() + 60,
    )
    assert store.goal("goal-wait")["status"] == "waiting_condition"
    assert store.confirm_wake_condition(
        condition_id="condition-1", goal_id="goal-wait", principal_id="owner"
    )
    assert store.goal("goal-wait")["status"] == "waiting"
    assert store.wake_condition("condition-1")["status"] == "satisfied"


def test_robot_status_wake_condition_requires_expected_payload(tmp_path: Path) -> None:
    store = AutonomyStore(tmp_path / "autonomy.sqlite3")
    assert store.create_goal(
        command_id="cmd-status",
        goal_id="goal-status",
        robot_id="robot-1",
        snapshot={},
        budgets={},
    )
    store._db.execute("UPDATE goals SET status='active' WHERE goal_id='goal-status'")
    store._db.commit()
    assert store.create_wake_condition(
        condition_id="condition-status",
        goal_id="goal-status",
        kind="robot_status",
        expected_payload={"state": "idle"},
        expires_at=time.time() + 60,
    )
    assert not store.satisfy_wake_condition(
        condition_id="condition-status",
        goal_id="goal-status",
        kind="robot_status",
        observed_payload={"state": "executing"},
        fulfilled_by="status:robot-1",
    )
    assert store.satisfy_wake_condition(
        condition_id="condition-status",
        goal_id="goal-status",
        kind="robot_status",
        observed_payload={"state": "idle"},
        fulfilled_by="status:robot-1",
    )
    assert store.goal("goal-status")["status"] == "waiting"


def test_expired_wake_condition_enters_review_without_resuming(tmp_path: Path) -> None:
    store = AutonomyStore(tmp_path / "autonomy.sqlite3")
    assert store.create_goal(
        command_id="cmd-expired",
        goal_id="goal-expired",
        robot_id="robot-1",
        snapshot={},
        budgets={},
    )
    store._db.execute("UPDATE goals SET status='active' WHERE goal_id='goal-expired'")
    store._db.commit()
    assert store.create_wake_condition(
        condition_id="condition-expired",
        goal_id="goal-expired",
        kind="human_confirmation",
        expected_payload={},
        expires_at=time.time() - 1,
    )

    assert store.expire_wake_conditions() == ["goal-expired"]
    assert store.wake_condition("condition-expired")["status"] == "expired"
    view = store.goal_view("goal-expired")
    assert view is not None
    assert view["status"] == "needs_review"
    assert view["review"]["reason"] == "WAKE_CONDITION_EXPIRED"


def test_goal_view_includes_wait_gate_budget_and_durable_timeline(
    tmp_path: Path,
) -> None:
    store = AutonomyStore(tmp_path / "autonomy.sqlite3")
    assert store.create_goal(
        command_id="cmd-view",
        goal_id="goal-view",
        robot_id="robot-1",
        snapshot={"objective": "inspect"},
        budgets={"max_skills": 2},
    )
    view = store.goal_view("goal-view")
    assert view is not None
    assert view["gate"]["state"] == "ready"
    assert view["budget"]["limits"]["max_skills"] == 2
    assert view["waiting_condition"] is None
    assert store.goal_timeline("goal-view")[0]["kind"] == "goal.created"


def test_goal_notification_receipt_is_idempotent(tmp_path: Path) -> None:
    store = AutonomyStore(tmp_path / "autonomy.sqlite3")
    assert store.claim_goal_notification(
        goal_id="goal", goal_version=1, status="active", channel="web"
    )
    assert not store.claim_goal_notification(
        goal_id="goal", goal_version=1, status="active", channel="web"
    )


def test_needs_review_is_nonterminal_and_keeps_reason_in_goal_view(
    tmp_path: Path,
) -> None:
    store = AutonomyStore(tmp_path / "autonomy.sqlite3")
    assert store.create_goal(
        command_id="cmd-review",
        goal_id="goal-review",
        robot_id="robot",
        snapshot={},
        budgets={},
    )
    store._db.execute("UPDATE goals SET status='active' WHERE goal_id='goal-review'")
    store._db.commit()
    assert store.mark_needs_review(
        goal_id="goal-review", reason="NO_PROGRESS", details={"repeated_actions": 3}
    )
    view = store.goal_view("goal-review")
    assert view is not None
    assert view["status"] == "needs_review"
    assert view["review"]["reason"] == "NO_PROGRESS"
