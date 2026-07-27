from hey_robot.skills.vln.option import planner_to_actions


def test_base_action_chunk_consumes_velocity_actions() -> None:
    commands = planner_to_actions(
        {
            "control_mode": "base_action_chunk",
            "control_chunk": {
                "kind": "base_velocity_chunk",
                "stop": False,
                "actions": [
                    {
                        "kind": "base_velocity_step",
                        "vx": 0.0,
                        "vy": 0.0,
                        "wz": -0.3,
                        "duration_ms": 250,
                        "source": "discrete_right",
                    }
                ],
            },
        }
    )

    command = commands[0]
    assert command["name"] == "base_velocity_step"
    assert command["arguments"] == {
        "vx": 0.0,
        "vy": 0.0,
        "wz": -0.3,
        "duration_ms": 250,
    }
    assert command["reason"] == "discrete_right"


def test_base_action_chunk_appends_deferred_stop() -> None:
    commands = planner_to_actions(
        {
            "control_mode": "base_action_chunk",
            "control_chunk": {
                "kind": "base_velocity_chunk",
                "stop": False,
                "stop_after_actions": True,
                "actions": [
                    {
                        "kind": "base_velocity_step",
                        "vx": 0.25,
                        "vy": 0.0,
                        "wz": 0.0,
                        "duration_ms": 1000,
                        "source": "system1_forward",
                    }
                ],
            },
        }
    )

    assert [command["name"] for command in commands] == [
        "base_velocity_step",
        "stop_motion",
    ]
