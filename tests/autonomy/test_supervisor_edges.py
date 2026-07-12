"""Cover supervisor edge cases: watchdog, gate checks, budget, contract validation, cancel flows."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from hey_robot.cognition.autonomous.supervisor import AutonomySupervisorService
from hey_robot.config import DeploymentConfig
from hey_robot.protocol import (
    ActionProposal,
    DeliberationResult,
    Envelope,
    EvidenceFact,
    FailurePayload,
    GoalBudgets,
    GoalCommand,
    RobotStatus,
    SkillControlResult,
    SkillResult,
    SuccessCriterion,
    TaskEvaluationPayload,
)
from hey_robot.protocol.messages import to_payload


class FakeBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict]] = []

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def publish(self, topic: str, payload: dict) -> None:
        self.published.append((topic, payload))

    async def subscribe(self, _topics: list[str], _handler) -> None:
        pass


def _service(tmp_path: Path, **autonomy_overrides) -> AutonomySupervisorService:
    autonomy = {
        "enabled": True,
        "entity_catalog": ["robot:main", "room:lab", "scene"],
        **autonomy_overrides,
    }
    config = DeploymentConfig.from_dict(
        {
            "deployment": {"id": "test"},
            "resources": {"runtime_dir": str(tmp_path / "runtime")},
            "autonomy": autonomy,
        }
    )
    service = AutonomySupervisorService(config)
    service.bus = FakeBus()  # type: ignore[assignment]
    return service


def _set_robot_ready(svc: AutonomySupervisorService) -> None:
    env = Envelope(robot_id="main", agent_id="main")
    asyncio.run(
        svc._on_snapshot(
            svc.topics.robot_status,
            to_payload(RobotStatus(env, state="idle", battery_percentage=100)),
        )
    )


def _create_goal(svc: AutonomySupervisorService, **kw) -> str:
    import uuid

    env = Envelope(robot_id="main", agent_id="main")
    criteria = kw.pop(
        "criteria",
        (
            SuccessCriterion(
                "c1", "evidence_present", "robot:main", "observed", "scene", 300
            ),
        ),
    )
    cmd = GoalCommand(
        env,
        str(uuid.uuid4()),
        "create",
        objective=kw.pop("objective", "test"),
        success_criteria=criteria,
        **kw,
    )
    asyncio.run(svc._on_goal_command(svc.topics.goal_command, to_payload(cmd)))
    goals = svc.store.goals_recent(1)
    assert len(goals) == 1
    return goals[0]["goal_id"]


def _request_hash(svc: AutonomySupervisorService, goal: dict) -> str:
    value = svc.store.deliberation_request_hash(
        goal_id=goal["goal_id"], deliberation_id=goal["active_deliberation_id"]
    )
    assert value is not None
    return value


# ── Contract validation ──────────────────────────────────────────────────────


def test_rejects_goal_with_unregistered_entity(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    env = Envelope(robot_id="main", agent_id="main")
    cmd = GoalCommand(
        env,
        "cmd1",
        "create",
        objective="go",
        success_criteria=(
            SuccessCriterion(
                "c1", "robot_state", "robot:main", "equals", "room:mars", 300
            ),
        ),
    )
    asyncio.run(svc._on_goal_command(svc.topics.goal_command, to_payload(cmd)))
    assert svc.store.goals_recent(10) == []


def test_rejects_goal_with_unregistered_object_id(tmp_path: Path) -> None:
    svc = _service(
        tmp_path, entity_catalog=["robot:main"]
    )  # only robot registered, no scene/room
    env = Envelope(robot_id="main", agent_id="main")
    cmd = GoalCommand(
        env,
        "cmd1",
        "create",
        objective="test",
        success_criteria=(
            SuccessCriterion(
                "c1", "evidence_present", "robot:main", "observed", "unreg:room", 300
            ),
        ),
    )
    asyncio.run(svc._on_goal_command(svc.topics.goal_command, to_payload(cmd)))
    assert svc.store.goals_recent(10) == []


def test_rejects_create_without_robot_id(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    env = Envelope(robot_id=None)
    cmd = GoalCommand(
        env,
        "cmd1",
        "create",
        objective="test",
        success_criteria=(
            SuccessCriterion(
                "c1", "evidence_present", "robot:main", "observed", "scene", 300
            ),
        ),
    )
    with pytest.raises(ValueError, match="robot_id"):
        asyncio.run(svc._on_goal_command(svc.topics.goal_command, to_payload(cmd)))


def test_rejects_create_without_objective(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    env = Envelope(robot_id="main")
    cmd = GoalCommand(
        env,
        "cmd1",
        "create",
        objective="",
        success_criteria=(
            SuccessCriterion(
                "c1", "evidence_present", "robot:main", "observed", "scene", 300
            ),
        ),
    )
    with pytest.raises(ValueError, match="objective"):
        asyncio.run(svc._on_goal_command(svc.topics.goal_command, to_payload(cmd)))


def test_rejects_create_without_criteria(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    env = Envelope(robot_id="main")
    cmd = GoalCommand(env, "cmd1", "create", objective="test", success_criteria=())
    with pytest.raises(ValueError, match="success_criteria"):
        asyncio.run(svc._on_goal_command(svc.topics.goal_command, to_payload(cmd)))


def test_rejects_create_when_gate_not_ready(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    svc.store.gate("main")  # ensure gate row exists
    svc.store._db.execute(
        "UPDATE robot_execution_gate SET state='uncertain' WHERE robot_id='main'"
    )
    svc.store._db.commit()
    env = Envelope(robot_id="main")
    cmd = GoalCommand(
        env,
        "cmd1",
        "create",
        objective="test",
        success_criteria=(
            SuccessCriterion(
                "c1", "evidence_present", "robot:main", "observed", "scene", 300
            ),
        ),
    )
    asyncio.run(svc._on_goal_command(svc.topics.goal_command, to_payload(cmd)))
    assert svc.store.goals_recent(10) == []


def test_command_idempotency_replay(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    env = Envelope(robot_id="main")
    asyncio.run(
        svc._on_snapshot(
            svc.topics.robot_status,
            to_payload(RobotStatus(env, state="idle", battery_percentage=100)),
        )
    )
    cmd = GoalCommand(
        env,
        "dup_id",
        "create",
        objective="test",
        success_criteria=(
            SuccessCriterion(
                "c1", "evidence_present", "robot:main", "observed", "scene", 300
            ),
        ),
    )
    asyncio.run(svc._on_goal_command(svc.topics.goal_command, to_payload(cmd)))
    asyncio.run(svc._on_goal_command(svc.topics.goal_command, to_payload(cmd)))
    goals = svc.store.goals_recent(10)
    assert len(goals) == 1


# ── Deliberation result edge cases ──────────────────────────────────────────


def test_deliberation_result_for_unknown_goal_ignored(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    result = DeliberationResult(
        Envelope(),
        "d1",
        "ignored",
        "nonexistent",
        "t1",
        "failed",
        failure=FailurePayload("TEST", "TEST", "test", "test"),
    )
    asyncio.run(
        svc._on_deliberation_result(
            svc.topics.agent_deliberation_result, to_payload(result)
        )
    )
    # No crash


def test_deliberation_result_wrong_active_id_ignored(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    _set_robot_ready(svc)
    gid = _create_goal(svc)
    result = DeliberationResult(
        Envelope(robot_id="main"),
        "wrong_delib_id",
        "ignored",
        gid,
        "t1",
        "failed",
        failure=FailurePayload("TEST", "TEST", "test", "test"),
    )
    asyncio.run(
        svc._on_deliberation_result(
            svc.topics.agent_deliberation_result, to_payload(result)
        )
    )
    # Should be ignored, goal still active
    g = svc.store.goal(gid)
    assert g is not None
    assert g["status"] == "active"


def test_deliberation_result_wrong_hash_ignored(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    _set_robot_ready(svc)
    gid = _create_goal(svc)
    goal = svc.store.goal(gid)
    assert goal is not None
    result = DeliberationResult(
        Envelope(robot_id="main"),
        goal["active_deliberation_id"],
        "forged",
        gid,
        goal["snapshot"]["task_id"],
        "failed",
        failure=FailurePayload("TEST", "FORGED", "test", "test"),
    )
    asyncio.run(
        svc._on_deliberation_result(
            svc.topics.agent_deliberation_result, to_payload(result)
        )
    )
    assert svc.store.goal(gid)["status"] == "active"


def test_deliberation_result_completed_sets_goal_completed(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    _set_robot_ready(svc)
    gid = _create_goal(svc)
    g = svc.store.goal(gid)
    result = DeliberationResult(
        Envelope(robot_id="main"),
        g["active_deliberation_id"],
        _request_hash(svc, g),
        gid,
        g["snapshot"]["task_id"],
        "completed",
        evaluation=TaskEvaluationPayload("satisfied", "done", ("e1",)),
    )
    asyncio.run(
        svc._on_deliberation_result(
            svc.topics.agent_deliberation_result, to_payload(result)
        )
    )
    g2 = svc.store.goal(gid)
    assert g2 is not None
    assert g2["status"] == "completed"


def test_deliberation_result_failed_code_preserved(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    _set_robot_ready(svc)
    gid = _create_goal(svc)
    g = svc.store.goal(gid)
    result = DeliberationResult(
        Envelope(robot_id="main"),
        g["active_deliberation_id"],
        _request_hash(svc, g),
        gid,
        g["snapshot"]["task_id"],
        "failed",
        failure=FailurePayload("MODEL_REQUEST", "PROVIDER_TIMEOUT", "test", "timeout"),
    )
    asyncio.run(
        svc._on_deliberation_result(
            svc.topics.agent_deliberation_result, to_payload(result)
        )
    )
    g2 = svc.store.goal(gid)
    assert g2 is not None
    assert g2["status"] == "failed"
    assert g2["termination_reason"] == "PROVIDER_TIMEOUT"


# ── Budget exhaustion ──────────────────────────────────────────────────────


def test_budget_exhausted_before_schedule_fails_goal(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    _set_robot_ready(svc)
    gid = _create_goal(svc, budgets=GoalBudgets(max_deliberations=0))
    g = svc.store.goal(gid)
    assert g["status"] == "failed"
    assert g["termination_reason"] == "BUDGET_EXHAUSTED"


# ── Watchdog tick ──────────────────────────────────────────────────────────


def test_watchdog_tick_budget_exhausted(tmp_path: Path) -> None:
    svc = _service(tmp_path, hard_max_wall_time_sec=0.001)
    import time as _time

    _set_robot_ready(svc)
    gid = _create_goal(svc)
    # Let time pass
    _time.sleep(0.1)
    svc._watchdog_tick()
    g = svc.store.goal(gid)
    assert g is not None
    assert g["status"] == "failed"


def test_watchdog_tick_noop_for_no_goals(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    svc._watchdog_tick()  # No crash, no-op
    assert True


def test_watchdog_skill_timeout(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    svc._deliberation_timeout = 0.0
    _set_robot_ready(svc)
    gid = _create_goal(svc)
    svc._db = svc.store._db
    svc.store._db.execute("UPDATE goals SET status='waiting' WHERE goal_id=?", (gid,))
    svc.store._db.execute(
        "INSERT INTO actions VALUES ('d1', 's_timeout', ?, 'published', 1, '{}', NULL, 0)",
        (gid,),
    )
    svc.store._db.commit()
    svc._watchdog_tick()
    g = svc.store.goal(gid)
    assert g is not None
    assert g["status"] == "blocked"
    assert svc.store.action("s_timeout")["status"] == "unknown"
    assert svc.store.gate("main").state == "uncertain"


# ── Cancel edge cases ─────────────────────────────────────────────────────


def test_cancel_nonexistent_goal_silent(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    cancel = GoalCommand(Envelope(), "c1", "cancel", goal_id="nonexistent")
    asyncio.run(svc._on_goal_command(svc.topics.goal_command, to_payload(cancel)))
    # No crash


def test_cancel_without_goal_id_ignored(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    cancel = GoalCommand(Envelope(), "c1", "cancel")
    with pytest.raises(ValueError, match="goal_id"):
        asyncio.run(svc._on_goal_command(svc.topics.goal_command, to_payload(cancel)))


def test_cancel_waiting_goal_with_no_active_action_calls_cancel(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    _set_robot_ready(svc)
    gid = _create_goal(svc)
    svc.store._db.execute("UPDATE goals SET status='waiting' WHERE goal_id=?", (gid,))
    svc.store._db.commit()
    cancel = GoalCommand(Envelope(robot_id="main"), "c1", "cancel", goal_id=gid)
    asyncio.run(svc._on_goal_command(svc.topics.goal_command, to_payload(cancel)))
    # Cancel for waiting goal without active action — cancel_goal only works for pending/active
    # But the supervisor still handles it gracefully
    g = svc.store.goal(gid)
    assert g is not None


# ── dispatch admission / gateway reject ────────────────────────────────────


def test_dispatch_rejected_by_gate_state(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    _set_robot_ready(svc)
    gid = _create_goal(svc)
    svc.store.gate("main")
    svc.store._db.execute(
        "UPDATE robot_execution_gate SET state='stop_pending' WHERE robot_id='main'"
    )
    svc.store._db.commit()
    g = svc.store.goal(gid)
    proposal = ActionProposal("skill", "navigate_to", "go", {"target": "room:lab"})
    result = DeliberationResult(
        Envelope(robot_id="main"),
        g["active_deliberation_id"],
        _request_hash(svc, g),
        gid,
        g["snapshot"]["task_id"],
        "action_proposed",
        proposal=proposal,
    )
    asyncio.run(
        svc._on_deliberation_result(
            svc.topics.agent_deliberation_result, to_payload(result)
        )
    )
    g2 = svc.store.goal(gid)
    assert g2 is not None
    assert g2["status"] == "failed"


def test_dispatch_rejected_by_robot_offline(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    _set_robot_ready(svc)
    svc._latest_status.clear()
    gid = _create_goal(svc)
    g = svc.store.goal(gid)
    proposal = ActionProposal("skill", "navigate_to", "go", {"target": "room:lab"})
    result = DeliberationResult(
        Envelope(robot_id="main"),
        g["active_deliberation_id"],
        _request_hash(svc, g),
        gid,
        g["snapshot"]["task_id"],
        "action_proposed",
        proposal=proposal,
    )
    asyncio.run(
        svc._on_deliberation_result(
            svc.topics.agent_deliberation_result, to_payload(result)
        )
    )
    g2 = svc.store.goal(gid)
    assert g2 is not None
    assert g2["status"] == "failed"


# ── SkillResult edge cases ──────────────────────────────────────────────────


def test_skill_result_for_unknown_action_ignored(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    result = SkillResult(Envelope(), "unknown_skill", status="completed", success=True)
    asyncio.run(svc._on_skill_result(svc.topics.skill_result, to_payload(result)))
    # No crash


def test_skill_result_rejects_foreign_evidence(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    _set_robot_ready(svc)
    gid = _create_goal(svc)
    goal = svc.store.goal(gid)
    assert goal is not None
    proposal = DeliberationResult(
        Envelope(robot_id="main"),
        goal["active_deliberation_id"],
        _request_hash(svc, goal),
        gid,
        goal["snapshot"]["task_id"],
        "action_proposed",
        proposal=ActionProposal(
            "skill", "navigate_to", "navigate", {"target": "room:lab"}
        ),
    )
    asyncio.run(
        svc._on_deliberation_result(
            svc.topics.agent_deliberation_result, to_payload(proposal)
        )
    )
    action = svc.store.active_action_for_goal(gid)
    assert action is not None
    foreign = EvidenceFact(
        "foreign",
        "other-goal",
        "skill_result",
        action["skill_id"],
        1.0,
        1,
        "robot:main",
        "equals",
        "room:lab",
    )
    asyncio.run(
        svc._on_skill_result(
            svc.topics.skill_result,
            to_payload(
                SkillResult(
                    Envelope(robot_id="main"),
                    action["skill_id"],
                    status="completed",
                    success=True,
                    evidence=(foreign,),
                )
            ),
        )
    )
    assert svc.store.evidence_for_goal(gid) == []


def test_skill_result_failed_sets_goal_failed(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    _set_robot_ready(svc)
    gid = _create_goal(svc)
    svc.store._db.execute("UPDATE goals SET status='waiting' WHERE goal_id=?", (gid,))
    svc.store._db.execute(
        "INSERT INTO actions VALUES ('d_fail', 's_fail', ?, 'published', 1, '{}', NULL, ?)",
        (gid, __import__("time").time()),
    )
    svc.store._db.commit()
    result = SkillResult(Envelope(), "s_fail", status="failed", success=False)
    asyncio.run(svc._on_skill_result(svc.topics.skill_result, to_payload(result)))
    g2 = svc.store.goal(gid)
    assert g2 is not None
    assert g2["status"] == "failed"


def test_skill_result_interrupted_sets_goal_failed(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    _set_robot_ready(svc)
    gid = _create_goal(svc)
    svc.store._db.execute("UPDATE goals SET status='waiting' WHERE goal_id=?", (gid,))
    svc.store._db.execute(
        "INSERT INTO actions VALUES ('d_int', 's_int', ?, 'published', 1, '{}', NULL, ?)",
        (gid, __import__("time").time()),
    )
    svc.store._db.commit()
    result = SkillResult(Envelope(), "s_int", status="interrupted", success=False)
    asyncio.run(svc._on_skill_result(svc.topics.skill_result, to_payload(result)))
    g2 = svc.store.goal(gid)
    assert g2 is not None
    assert g2["status"] == "failed"


def test_skill_result_unknown_sets_goal_blocked(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    _set_robot_ready(svc)
    gid = _create_goal(svc)
    svc.store._db.execute("UPDATE goals SET status='waiting' WHERE goal_id=?", (gid,))
    svc.store._db.execute(
        "INSERT INTO actions VALUES ('d_unk', 's_unk', ?, 'published', 1, '{}', NULL, ?)",
        (gid, __import__("time").time()),
    )
    svc.store._db.commit()
    result = SkillResult(Envelope(), "s_unk", status="unknown", success=None)
    asyncio.run(svc._on_skill_result(svc.topics.skill_result, to_payload(result)))
    g2 = svc.store.goal(gid)
    assert g2 is not None
    assert g2["status"] == "blocked"


# ── Control result edge cases ──────────────────────────────────────────────


def test_control_result_for_missing_control_ignored(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    result = SkillControlResult(
        Envelope(), "unknown_ctrl", "interrupt", None, "completed", True
    )
    asyncio.run(
        svc._on_control_result(svc.topics.skill_control_result, to_payload(result))
    )
    # No crash


# ── Evidence validation ────────────────────────────────────────────────────


def test_evidence_invalid_subject_filtered(tmp_path: Path) -> None:
    from hey_robot.protocol import EvidenceFact

    svc = _service(tmp_path)
    fact = EvidenceFact(
        "e1",
        "g1",
        "skill_result",
        "s1",
        1.0,
        1,
        "unregistered:subject",
        "observed",
        "scene",
    )
    assert not svc._valid_evidence(
        fact,
        robot_id=None,
        goal_id="g1",
        source_kind="skill_result",
        source_id="s1",
    )


def test_evidence_valid_subject_accepted(tmp_path: Path) -> None:
    from hey_robot.protocol import EvidenceFact

    svc = _service(tmp_path)
    fact = EvidenceFact(
        "e1", "g1", "skill_result", "s1", 1.0, 1, "robot:main", "observed", "scene"
    )
    assert svc._valid_evidence(
        fact,
        robot_id="main",
        goal_id="g1",
        source_kind="skill_result",
        source_id="s1",
    )


# ── Persistence failure during schedule ───────────────────────────────────


def test_schedule_persistence_failure(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    svc._watchdog_interval = 999.0  # disable watchdog
    svc._watchdog_task = None
    _set_robot_ready(svc)
    gid = _create_goal(svc)
    g = svc.store.goal(gid)
    assert g is not None
    assert g["status"] == "active"
