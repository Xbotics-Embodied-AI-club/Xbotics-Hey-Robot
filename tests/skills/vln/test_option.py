from hey_robot.skills.vln.option import planner_to_action


def test_pixel_goal_conversion_uses_reported_image_width() -> None:
    centered = planner_to_action(
        {"mode": "pixel_goal", "pixel_goal": [100, 160], "image_width": 320}
    )
    right = planner_to_action(
        {"mode": "pixel_goal", "pixel_goal": [100, 300], "image_width": 320}
    )

    assert centered["name"] == "move_base"
    assert right["name"] == "turn_base"
    assert right["arguments"]["direction"] == "right"


def test_discrete_actions_keep_internnav_motion_semantics() -> None:
    forward = planner_to_action(
        {"action_code": 1, "heading_deg": 0.0, "forward_distance_cm": 25.0}
    )
    left = planner_to_action({"action_code": 2, "heading_deg": -15.0})
    right = planner_to_action({"action_code": 3, "heading_deg": 15.0})

    assert forward["arguments"] == {"direction": "forward", "distance_cm": 25.0}
    assert left["arguments"] == {"direction": "left", "angle_deg": 15.0}
    assert right["arguments"] == {"direction": "right", "angle_deg": 15.0}
