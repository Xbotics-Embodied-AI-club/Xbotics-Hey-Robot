from __future__ import annotations

from evaluation.habitat3.worker.preflight import report
from evaluation.habitat3.worker.profiles import get_profile


def test_preflight_reports_each_missing_required_asset(tmp_path) -> None:
    result = report(tmp_path, "habitat3_social_spot_human_oracle")

    assert result["ok"] is False
    assert len(result["missing"]) == 6
    assert {item["name"] for item in result["checks"]} == {
        "HSSD scene dataset config",
        "HSSD stages",
        "HSSD uncluttered scene instances",
        "Habitat rearrange dataset",
        "Spot URDF",
        "Humanoid data",
    }


def test_symbolic_profile_is_allowlisted_and_uses_pddl_task() -> None:
    profile = get_profile("habitat3_spot_human_rearrange_symbolic")

    assert profile.task == "RearrangePddlTask-v0"
    assert profile.executor_mode == "privileged_symbolic"
    assert "habitat_symbolic_pick" in profile.skills
    assert "habitat_follow_human" not in profile.skills
