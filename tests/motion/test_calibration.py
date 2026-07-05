from __future__ import annotations

import numpy as np

from hey_robot.motion.calibration import camera_to_base, load_transform


def test_simulation_calibration_defaults_to_identity() -> None:
    transform = load_transform(None)
    point = np.asarray([0.2, -0.1, 0.05])
    assert np.array_equal(camera_to_base(point, transform), point)
