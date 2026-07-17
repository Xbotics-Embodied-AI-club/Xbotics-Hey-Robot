# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
# 为 Xbotics Hey Robot 修改。
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def load_transform(path: str | Path | None = None) -> np.ndarray:
    """加载相机到机座的齐次变换；仿真中缺省返回单位矩阵。"""
    if path is None:
        return np.eye(4, dtype=np.float64)
    resolved = Path(path).expanduser()
    if not resolved.exists():
        return np.eye(4, dtype=np.float64)
    with resolved.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    transform = np.asarray(payload["transform_matrix"], dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("calibration transform must be a finite 4x4 matrix")
    return transform


def camera_to_base(position: np.ndarray, transform: np.ndarray) -> np.ndarray:
    point = np.asarray(position, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    if point.shape != (3,) or matrix.shape != (4, 4):
        raise ValueError("camera_to_base expects a 3-vector and a 4x4 transform")
    homogeneous = np.concatenate((point, np.ones(1, dtype=np.float64)))
    return (matrix @ homogeneous)[:3]
