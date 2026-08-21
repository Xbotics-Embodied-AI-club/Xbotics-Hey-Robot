from __future__ import annotations

import json

from hey_robot.skills.vla.execution_memory import (
    record_attempt,
    retrieve_successful_attempts,
)


def test_execution_memory_records_symbolic_attempt(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HEY_ROBOT_EXECUTION_MEMORY_DIR", str(tmp_path))
    path = record_attempt(
        task="RinseSinkBasin",
        run_id="trial-1",
        prompt="rinse the sink basin",
        result={
            "termination_reason": "environment_done",
            "subgoal_succeeded": True,
            "steps_used": 12,
            "attempt_diagnostics": {"after": {"grasp_contact": True}},
        },
    )
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["primitive"] == "vla_act"
    assert row["task"] == "RinseSinkBasin"
    assert row["attempt_diagnostics"]["after"]["grasp_contact"] is True
    assert retrieve_successful_attempts("RinseSinkBasin") == [row]
