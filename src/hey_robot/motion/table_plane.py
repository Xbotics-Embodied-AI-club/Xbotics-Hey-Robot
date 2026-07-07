# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
# Ported to Xbotics Hey Robot: table-plane grasp depth estimator.

"""Table-plane ray intersection — single-RGB depth without a depth camera.

The core idea: if an object sits on a calibrated table plane, a single RGB
detection (bbox bottom-centre pixel) + the known plane equation + the known
camera pose gives the object's 3D position — no depth sensor needed.

The plane is fit once by touching 3+ non-collinear table points with the
end-effector and recording their FK positions. After that ray_plane_intersection
estimates 3D for any detected bbox whose bottom edge touches the table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Plane fitting
# ---------------------------------------------------------------------------


def fit_plane(points: np.ndarray) -> tuple[float, float, float, float]:
    """Fit plane ax + by + cz + d = 0 from N×3 points (N >= 3, non-collinear).

    Returns (a, b, c, d) where (a, b, c) is the unit normal.

    Raises ValueError when fewer than 3 points or points are collinear.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 3:
        raise ValueError("Need at least 3 non-collinear 3D points")
    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid)
    normal = vh[2]
    normal /= np.linalg.norm(normal)
    a, b, c = float(normal[0]), float(normal[1]), float(normal[2])
    d = -float(np.dot(normal, centroid))
    return (a, b, c, d)


# ---------------------------------------------------------------------------
# Ray-plane intersection
# ---------------------------------------------------------------------------


def ray_plane_intersection(
    pixel: tuple[float, float],
    camera_matrix: np.ndarray,
    cam_position: np.ndarray,
    cam_rotation: np.ndarray,
    plane: tuple[float, float, float, float],
) -> np.ndarray | None:
    """Compute the 3D world point where a camera ray hits the table plane.

    Args:
        pixel: (u, v) image coordinate — typically the bottom centre of a bbox
               whose lower edge contacts the table surface.
        camera_matrix: 3×3 intrinsics [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
        cam_position: (3,) world-frame camera optical centre.
        cam_rotation: (3,3) world-from-camera rotation matrix.
        plane: (a, b, c, d) plane coefficients, ax + by + cz + d = 0.

    Returns:
        (3,) world-frame intersection point, or None when the ray is parallel
        to the plane or the intersection is behind the camera.
    """
    u, v = float(pixel[0]), float(pixel[1])
    intrinsic = np.asarray(camera_matrix, dtype=np.float64)
    pos = np.asarray(cam_position, dtype=np.float64)
    rotation = np.asarray(cam_rotation, dtype=np.float64)
    a, b, c, d = (float(plane[0]), float(plane[1]), float(plane[2]), float(plane[3]))

    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])

    # Camera-frame ray (OpenCV convention: z forward, x right, y down).
    ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)

    # World-frame ray.
    ray_world = rotation @ ray_cam

    # Ray-plane intersection: n·(pos + t*ray) + d = 0 → t = -(n·pos + d) / (n·ray).
    denom = a * ray_world[0] + b * ray_world[1] + c * ray_world[2]
    if abs(denom) < 1e-9:
        return None  # ray parallel to plane
    t = -(a * pos[0] + b * pos[1] + c * pos[2] + d) / denom
    if t <= 0:
        return None  # plane behind camera

    return pos + t * ray_world  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Camera intrinsics helpers
# ---------------------------------------------------------------------------


def intrinsics_from_fovy(fovy_deg: float, width: int, height: int) -> np.ndarray:
    """Build 3×3 intrinsics from vertical FOV and image dimensions (square pixels)."""
    fy = (height / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    fx = fy  # square pixels in simulation
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Persistent calibration
# ---------------------------------------------------------------------------


class TablePlaneCalibration:
    """Persistent table-plane calibration, saved to / loaded from YAML."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else None
        self._plane: tuple[float, float, float, float] | None = None
        if self._path and self._path.exists():
            self.load(self._path)

    @property
    def plane(self) -> tuple[float, float, float, float] | None:
        return self._plane

    def calibrate(self, points: list[tuple[float, float, float]]) -> None:
        """Fit plane from 3+ FK touch points and persist."""
        arr = np.array(points, dtype=np.float64)
        self._plane = fit_plane(arr)
        if self._path:
            self.save(self._path)

    def save(self, path: str | Path) -> None:
        if self._plane is None:
            raise RuntimeError("No calibration to save — run calibrate() first")
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(
                {"plane": list(self._plane), "note": "table-plane calibration"},
                fh,
            )

    def load(self, path: str | Path) -> None:
        resolved = Path(path).expanduser()
        if not resolved.exists():
            raise FileNotFoundError(f"Calibration file not found: {resolved}")
        payload = yaml.safe_load(str(resolved)) or {}
        coeffs = payload.get("plane")
        if not isinstance(coeffs, (list, tuple)) or len(coeffs) != 4:
            raise ValueError("Invalid calibration file: missing or malformed 'plane'")
        self._plane = (
            float(coeffs[0]),
            float(coeffs[1]),
            float(coeffs[2]),
            float(coeffs[3]),
        )
