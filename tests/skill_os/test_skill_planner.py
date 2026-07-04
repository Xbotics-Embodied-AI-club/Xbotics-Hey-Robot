from __future__ import annotations

from hey_robot.skill_os.skill_planner import SkillPlanner


def test_skill_planner_emits_plain_human_follow_without_default_duration() -> None:
    action = SkillPlanner().plan("follow me")

    assert action is not None
    assert action.name == "human_follow"
    assert action.arguments == {}


def test_skill_planner_maps_common_classic_skills() -> None:
    planner = SkillPlanner()

    cases = [
        ("urgent stop now", "stop_motion", {"emergency": True}),
        ("stop", "stop_motion", {}),
        ("look around", "look_around", {"question": "look around"}),
        ("align marker", "detect_marker", {}),
        ("look at person", None, None),
        ("remember this scene", None, None),
        ("recall memory", None, None),
        ("what do you see", "inspect_scene", {"question": "what do you see"}),
        ("reset posture", "reset_posture", {}),
        ("move to pose home", "set_arm_pose", {"pose_name": "home"}),
        ("open gripper", "set_gripper", {"action": "open"}),
        ("close gripper", "set_gripper", {"action": "close"}),
        ("gripper 42 percent", "set_gripper", {"opening_pct": 42.0}),
        (
            "move backward 2m",
            "move_base",
            {"direction": "backward", "distance_cm": 80.0},
        ),
        (
            "move forward 12cm",
            "move_base",
            {"direction": "forward", "distance_cm": 12.0},
        ),
        ("turn left 45", "turn_base", {"direction": "left", "angle_deg": 45.0}),
        ("turn right 180", "turn_base", {"direction": "right", "angle_deg": 120.0}),
    ]

    for text, name, arguments in cases:
        action = planner.plan(text)
        if name is None:
            assert action is None
            continue
        assert action is not None
        assert action.name == name
        assert action.arguments == arguments
