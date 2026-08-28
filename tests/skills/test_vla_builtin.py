from __future__ import annotations

import pytest

from hey_robot.skills.builtins import (
    robocasa_session,
    vla as vla_builtin,
)
from hey_robot.skills.builtins.vla import _attempt_diagnostics
from hey_robot.skills.context import SkillContext
from hey_robot.skills.models import SkillResult


@pytest.fixture(autouse=True)
def _clear_session_state() -> None:
    robocasa_session.clear_session_state()


def test_local_option_diagnostics_are_normalized_for_agent_decisions() -> None:
    diagnostics = {
        "grasp_detected": False,
        "grasp_obj": None,
        "peak_lift": 0.02,
        "base_drift": 0.1,
    }

    assert _attempt_diagnostics({"vla_history": [{"diagnostics": diagnostics}]}) == {
        "after": diagnostics
    }


async def test_repeated_miss_does_not_block_an_environment_root_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def uses_local(_ctx: SkillContext) -> bool:
        return True

    async def effective_steps(_ctx: SkillContext, _prompt: str, requested: int) -> int:
        return requested

    monkeypatch.setattr(vla_builtin, "_uses_local_foundation_option", uses_local)
    monkeypatch.setattr(vla_builtin, "_effective_max_steps", effective_steps)
    robocasa_session.set_reposition_required("robocasa", "episode", True)
    calls: list[str] = []

    async def execute(
        _ctx: SkillContext, primitive: str, _arguments: dict
    ) -> SkillResult:
        calls.append(primitive)
        return SkillResult(
            True,
            "policy option completed",
            "completed",
            data={
                "execution_success": True,
                "termination_reason": "budget",
                "option": {"status": "budget"},
            },
        )

    async def task_progress(_ctx: SkillContext) -> dict[str, object]:
        return {"washed_time": 0.0}

    monkeypatch.setattr(vla_builtin, "execute_robot_action", execute)
    monkeypatch.setattr(vla_builtin, "_task_progress", task_progress)

    result = await vla_builtin.manipulate(
        SkillContext("run", "episode", "robocasa"),
        {"task_prompt": "Stir the vegetables.", "max_steps": 17},
    )

    assert result.success is True
    assert calls == ["run_policy_option"]
    assert result.data["termination_reason"] == "budget"
    assert result.data["decision_state"]["required_next_action"] is None
    assert result.data["decision_state"]["retry_prompt_must_match_exactly"] is True
