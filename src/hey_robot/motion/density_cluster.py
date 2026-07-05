from __future__ import annotations

import numpy as np


def density_cluster_mean(samples: np.ndarray, threshold: float) -> np.ndarray:
    points = np.asarray(samples, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("samples must be a non-empty Nx3 array")
    if not np.all(np.isfinite(points)):
        raise ValueError("samples must be finite")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if len(points) < 3:
        return np.asarray(np.median(points, axis=0), dtype=np.float64)
    best_index = 0
    best_count = 0
    for index, point in enumerate(points):
        count = int(np.sum(np.linalg.norm(points - point, axis=1) < threshold))
        if count > best_count:
            best_index = index
            best_count = count
    distances = np.linalg.norm(points - points[best_index], axis=1)
    cluster = points[distances < threshold]
    return np.asarray(np.mean(cluster, axis=0), dtype=np.float64)
