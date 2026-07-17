"""自治监督器使用的纯准入判断和预算判断。"""

from __future__ import annotations

from dataclasses import dataclass

from hey_robot.protocol import BudgetState, GoalBudgets, RobotExecutionGate, RobotStatus


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    code: str | None = None


def check_budget(
    *,
    budgets: GoalBudgets,
    hard_wall_time_sec: float,
    hard_deliberations: int,
    hard_skills: int,
    minimum_battery: float,
    state: BudgetState,
) -> PolicyDecision:
    if state.elapsed_wall_time_sec >= min(
        budgets.max_wall_time_sec, hard_wall_time_sec
    ):
        return PolicyDecision(False, "BUDGET_EXHAUSTED")
    if state.deliberations_used >= min(budgets.max_deliberations, hard_deliberations):
        return PolicyDecision(False, "BUDGET_EXHAUSTED")
    if state.skills_used >= min(budgets.max_skills, hard_skills):
        return PolicyDecision(False, "BUDGET_EXHAUSTED")
    if state.battery_percentage is None:
        return PolicyDecision(False, "STATE_UNKNOWN")
    if state.battery_percentage < max(budgets.min_battery_percentage, minimum_battery):
        return PolicyDecision(False, "BUDGET_EXHAUSTED")
    return PolicyDecision(True)


def dispatch_admission(
    *, gate: RobotExecutionGate, status: RobotStatus | None
) -> PolicyDecision:
    if gate.state != "ready":
        return PolicyDecision(False, "ROBOT_EXECUTION_UNCERTAIN")
    if status is None or status.state in {"offline", "unknown"}:
        return PolicyDecision(False, "ROBOT_OFFLINE")
    return PolicyDecision(True)
