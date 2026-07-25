from __future__ import annotations

import numpy as np

from hey_robot.robot_backends.xlerobot.hardware.vla_codec import (
    SO101_STATE_SCHEMA,
    action_vector_to_targets,
)


def test_action_vector_to_targets_scales_gripper_for_driver_range() -> None:
    class Adapter:
        gripper_open_value = 1.7

    class Driver:
        adapter = Adapter()

    targets = action_vector_to_targets([0, 1, 2, 3, 4, 0.5], Driver())

    assert np.allclose(targets, np.asarray([0, 1, 2, 3, 4, 0.85]))
    assert SO101_STATE_SCHEMA == "so101_single_arm_rad_gripper01"
