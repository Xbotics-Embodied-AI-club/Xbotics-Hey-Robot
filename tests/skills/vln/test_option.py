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
