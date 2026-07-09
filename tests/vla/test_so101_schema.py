from __future__ import annotations

import numpy as np

from hey_robot.vla.so101_schema import (
    ACTION_SPACE,
    SO101_STATE_SCHEMA,
    action_chunk_first_vector,
    action_vector_to_chunk,
    action_vector_to_targets,
    state_from_arm_status,
)


def test_state_from_arm_status_prefers_explicit_vla_state() -> None:
    raw = {
        "vla_state": [1, 2, 3, 4, 5, 0.25],
        "arm_status": {
            "joint_states": {
                "shoulder_pan": 9.0,
                "shoulder_lift": 9.0,
                "elbow_flex": 9.0,
                "wrist_flex": 9.0,
                "wrist_roll": 9.0,
            },
            "gripper_opening_pct": 90.0,
        },
    }

    assert state_from_arm_status(raw) == [1.0, 2.0, 3.0, 4.0, 5.0, 0.25]


def test_state_from_arm_status_builds_single_arm_vector_with_gripper_unit() -> None:
    raw = {
        "arm_status": {
            "joint_states": {
                "shoulder_pan": 0.1,
                "shoulder_lift": 0.2,
                "elbow_flex": 0.3,
                "wrist_flex": 0.4,
                "wrist_roll": 0.5,
            },
            "gripper_opening_pct": 75.0,
        }
    }

    assert state_from_arm_status(raw) == [0.1, 0.2, 0.3, 0.4, 0.5, 0.75]


def test_action_vector_to_chunk_round_trips_first_action() -> None:
    action = [0.1111119, 0.2, 0.3, 0.4, 0.5, 1.25]

    chunk = action_vector_to_chunk(action, done=True)

    assert chunk["kind"] == "action_chunk"
    assert chunk["action_space"] == ACTION_SPACE
    assert chunk["done"] is True
    assert chunk["actions"][0]["joints"]["shoulder_pan"] == 0.111112
    assert chunk["actions"][0]["gripper"] == 1.0
    assert action_chunk_first_vector(chunk) == [0.111112, 0.2, 0.3, 0.4, 0.5, 1.0]


def test_action_vector_to_targets_scales_gripper_for_driver_range() -> None:
    class Adapter:
        gripper_open_value = 1.7

    class Driver:
        adapter = Adapter()

    targets = action_vector_to_targets([0, 1, 2, 3, 4, 0.5], Driver())

    assert np.allclose(targets, np.asarray([0, 1, 2, 3, 4, 0.85]))
    assert SO101_STATE_SCHEMA == "so101_single_arm_rad_gripper01"
