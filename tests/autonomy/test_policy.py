from hey_robot.cognition.autonomous.policy import check_budget, dispatch_admission
from hey_robot.config import DeploymentConfig
from hey_robot.protocol import BudgetState, Envelope, GoalBudgets, RobotStatus


def test_budget_rejects_missing_battery_for_physical_admission() -> None:
    decision = check_budget(
        budgets=GoalBudgets(),
        hard_wall_time_sec=3600,
        hard_deliberations=40,
        hard_skills=24,
        minimum_battery=20,
        state=BudgetState(0, 0, 0, None),
    )
    assert decision.code == "STATE_UNKNOWN"


def test_dispatch_rejects_uncertain_gate_and_offline_status() -> None:
    assert (
        dispatch_admission(gate_state="uncertain", status=None).code
        == "ROBOT_EXECUTION_UNCERTAIN"
    )
    assert (
        dispatch_admission(
            gate_state="ready",
            status=RobotStatus(Envelope(robot_id="main"), state="offline"),
        ).code
        == "ROBOT_OFFLINE"
    )


def test_config_rejects_legacy_agent_autonomy_settings() -> None:
    try:
        DeploymentConfig.from_dict(
            {"agents": {"main": {"settings": {"autonomy": {"enabled": True}}}}}
        )
    except ValueError as exc:
        assert "removed autonomy fields" in str(exc)
    else:
        raise AssertionError("legacy autonomy settings were accepted")
