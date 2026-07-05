from __future__ import annotations

import numpy as np
import pytest

from hey_robot.motion.density_cluster import density_cluster_mean


def test_density_cluster_ignores_distant_outlier() -> None:
    samples = np.asarray(
        [[0.1, 0.2, 0.3], [0.101, 0.199, 0.3], [0.099, 0.201, 0.301], [1, 1, 1]]
    )
    result = density_cluster_mean(samples, 0.015)
    assert result == pytest.approx([0.1, 0.2, 0.300333333], abs=1e-6)
