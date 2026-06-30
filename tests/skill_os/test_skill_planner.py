from __future__ import annotations

from hey_robot.skill_os.skill_planner import SkillPlanner


def test_skill_planner_emits_plain_human_follow_without_default_duration() -> None:
    action = SkillPlanner().plan("follow me")

    assert action is not None
    assert action.name == "human_follow"
    assert action.arguments == {}
