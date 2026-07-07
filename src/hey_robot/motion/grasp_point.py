# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
# Ported to Xbotics Hey Robot: perception-driven grasp point localisation.

"""Grasp point localisation — bbox → 3D grasp point via table-plane intersection.

Adapted from vector-os-nano ``pick.py._sample_from_perception()``, replacing
their "depth camera → point cloud" step with "bbox bottom-centre → ray-plane
intersection". The sampling + density-clustering framework is identical; only
the 3D projection primitive differs.

Simulation note:
  When running inside MuJoCo the detector/oracle bridges are provided by the
  simulation driver (``sim_locate_object``, ``sim_get_object_state``).  This
  module is DETECTOR-AGNOSTIC — it consumes a bbox tuple and a camera geometry
  snapshot, regardless of whether the bbox came from a VLM, an oracle, or
  classical vision.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from hey_robot.motion.density_cluster import density_cluster_mean
from hey_robot.motion.table_plane import ray_plane_intersection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables (mirroring vector-os-nano pick.py constants)
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_COUNT: int = 20
DEFAULT_SAMPLE_INTERVAL: float = 0.05  # seconds
DEFAULT_CLUSTER_THRESHOLD: float = 0.015  # metres (1.5 cm)

# Workspace limits for SO101 (from vector-os-nano pick.py)
WORKSPACE_MIN_DIST: float = 0.05  # 5 cm
WORKSPACE_MAX_DIST: float = 0.35  # 35 cm


# ---------------------------------------------------------------------------
# Single-shot
# ---------------------------------------------------------------------------


def grasp_point_from_bbox(
    bbox: tuple[float, float, float, float],
    camera_matrix: np.ndarray,
    cam_position: np.ndarray,
    cam_rotation: np.ndarray,
    plane: tuple[float, float, float, float],
) -> np.ndarray | None:
    """Estimate the 3D world grasp point for one detection bbox.

    Uses the bottom-centre of *bbox* as the pixel that contacts the table
    surface, then ray-plane intersects to recover 3D.

    Args:
        bbox: (x1, y1, x2, y2) in image pixels.
        camera_matrix: 3×3 intrinsics.
        cam_position: (3,) world camera position.
        cam_rotation: (3,3) world-from-camera rotation.
        plane: (a, b, c, d) table-plane coefficients.

    Returns:
        (3,) world grasp point, or None if intersection is degenerate.
    """
    u = (float(bbox[0]) + float(bbox[2])) / 2.0
    v = float(bbox[3])  # bottom edge — expected to touch the table
    return ray_plane_intersection(
        (u, v),
        camera_matrix,
        cam_position,
        cam_rotation,
        plane,
    )


# ---------------------------------------------------------------------------
# Multi-sample density-cluster pipeline
# ---------------------------------------------------------------------------


def sample_grasp_point(
    detect_fn: Callable[[], tuple[float, float, float, float] | None],
    camera_matrix: np.ndarray,
    cam_position: np.ndarray,
    cam_rotation: np.ndarray,
    plane: tuple[float, float, float, float],
    *,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
    cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD,
) -> np.ndarray | None:
    """Sample N bboxes, project each to 3D, density-cluster to a robust point.

    *detect_fn* is called *sample_count* times — it encapsulates the detector
    (or oracle) and returns a single bbox (x1,y1,x2,y2) or None on miss.

    Returns the density-cluster mean (3,) world point, or None when fewer than
    3 valid samples are collected.
    """
    samples: list[np.ndarray] = []

    for _ in range(sample_count):
        bbox = detect_fn()
        if bbox is not None:
            point = grasp_point_from_bbox(
                bbox,
                camera_matrix,
                cam_position,
                cam_rotation,
                plane,
            )
            if point is not None:
                samples.append(point)
        if sample_interval > 0:
            time.sleep(sample_interval)

    if len(samples) < 3:
        logger.warning(
            "sample_grasp_point: only %d valid samples (need ≥3)", len(samples)
        )
        return None

    arr = np.array(samples, dtype=np.float64)
    if len(arr) < 3:
        return np.median(arr, axis=0)  # type: ignore[no-any-return]
    return density_cluster_mean(arr, cluster_threshold)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Workspace check
# ---------------------------------------------------------------------------


def check_workspace(
    point: np.ndarray,
    *,
    min_dist: float = WORKSPACE_MIN_DIST,
    max_dist: float = WORKSPACE_MAX_DIST,
) -> bool:
    """True when *point* is within the SO101 arm's XY reachable band."""
    dist_xy = float(np.linalg.norm(point[:2]))
    return min_dist <= dist_xy <= max_dist


# ---------------------------------------------------------------------------
# Camera geometry snapshot helper
# ---------------------------------------------------------------------------


class CameraGeometry:
    """Immutable snapshot of a camera's pose + intrinsics for a single frame."""

    def __init__(
        self,
        camera_matrix: np.ndarray,
        cam_position: np.ndarray,
        cam_rotation: np.ndarray,
    ) -> None:
        self.K = np.asarray(camera_matrix, dtype=np.float64)
        self.pos = np.asarray(cam_position, dtype=np.float64)
        self.R = np.asarray(cam_rotation, dtype=np.float64)

    @classmethod
    def from_mujoco(
        cls,
        model: Any,
        data: Any,
        camera_name: str,
        render_width: int = 640,
        render_height: int = 480,
    ) -> CameraGeometry:
        """Build from a MuJoCo model + data for the named camera."""
        import mujoco

        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            raise ValueError(f"MuJoCo camera not found: {camera_name}")

        fovy_deg = float(model.cam_fovy[cam_id])
        intrinsic = np.array(
            [
                [0.0, 0, render_width / 2.0],
                [0, 0.0, render_height / 2.0],
                [0, 0, 1.0],
            ],
            dtype=np.float64,
        )
        fy = (render_height / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
        intrinsic[0, 0] = fy
        intrinsic[1, 1] = fy

        pos = data.cam_xpos[cam_id].copy()
        rot_mujoco = data.cam_xmat[cam_id].reshape(3, 3).copy()

        # Convert MuJoCo camera frame → OpenCV convention.
        # MuJoCo: +X right, +Y up, +Z back (view along -Z)
        # OpenCV:  +X right, +Y down, +Z forward
        convert = np.diag([1.0, -1.0, -1.0])
        rot_opencv = rot_mujoco @ convert

        return cls(intrinsic, pos, rot_opencv)


def grasp_point_from_bbox_with_camera(
    bbox: tuple[float, float, float, float],
    camera: CameraGeometry,
    plane: tuple[float, float, float, float],
) -> np.ndarray | None:
    """Convenience wrapper: bbox + CameraGeometry + plane → world 3D point."""
    return grasp_point_from_bbox(bbox, camera.K, camera.pos, camera.R, plane)
