# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
# Ported to Xbotics Hey Robot: simulation validation of the table-plane
# ray-intersection grasp algorithm.

"""Validate grasp_point against MuJoCo ground truth.

Loads the home scene, reads camera parameters, uses the oracle
to get the true 3D position of the wand, back-projects that position to
2D (simulating a perfect detector), then recovers 3D via ray-plane intersection
and measures the reconstruction error.

A virtual "table" plane is placed at the wand's z-height so the algorithm
has a meaningful plane to intersect.

Usage:
    cd D:/agent_robot/Xbotics-Hey-Robot
    python tests/test_grasp_algorithm_sim.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

# When running from tests/, Python adds tests/ to sys.path[0].
# tests/logging/__init__.py shadows the stdlib logging module, which breaks any
# module that does ``import logging`` (e.g. grasp_point.py).  Pop the tests
# directory from sys.path before anything else caches the wrong module.
_this_dir = str(Path(__file__).resolve().parent)
for _i, _p in enumerate(sys.path):
    if str(Path(_p).resolve()) == _this_dir:
        sys.path.pop(_i)
        break

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

# ---------------------------------------------------------------------------
# Validation parameters
# ---------------------------------------------------------------------------

SCENE_PATH = PROJECT_ROOT / "assets" / "scenes" / "home_scene.xml"

PASS_THRESHOLD_M = 0.025  # 2.5 cm — generous for first-pass table-plane recovery
RENDER_WIDTH = 640
RENDER_HEIGHT = 480

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def setup_mujoco():
    """Configure MuJoCo GL, import, load model, return (model, data)."""
    import os
    import platform

    if platform.system() == "Windows":
        os.environ.setdefault("MUJOCO_GL", "wgl")
    elif platform.system() == "Linux" and not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # settle
    for _ in range(50):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def get_camera_geometry(model, data, cam_name):
    """Return (intrinsic, pos, rotation) for the named MuJoCo camera."""
    import mujoco

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        raise ValueError(f"Camera not found: {cam_name}")

    fovy_deg = float(model.cam_fovy[cam_id])
    fy = (RENDER_HEIGHT / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    fx = fy
    intrinsic = np.array(
        [[fx, 0, RENDER_WIDTH / 2.0], [0, fy, RENDER_HEIGHT / 2.0], [0, 0, 1.0]],
        dtype=np.float64,
    )
    pos = data.cam_xpos[cam_id].copy()
    rotation = data.cam_xmat[cam_id].reshape(3, 3).copy()
    return intrinsic, pos, rotation


def project_3d_to_2d(world_point, intrinsic, cam_pos, cam_rot):
    """Project a 3D world point to image coordinates.

    Returns (u, v) or None if point is behind camera.
    """
    point_cam = cam_rot.T @ (world_point - cam_pos)
    if point_cam[2] <= 0:
        return None
    u = intrinsic[0, 0] * point_cam[0] / point_cam[2] + intrinsic[0, 2]
    v = intrinsic[1, 1] * point_cam[1] / point_cam[2] + intrinsic[1, 2]
    return (float(u), float(v))


def get_wand_position(model, data):
    """Return ground-truth 3D position of the wand body."""
    import mujoco

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wand")
    return data.xpos[body_id].copy()


def get_wand_grasp_position(model, data):
    """Return ground-truth 3D position of the wand_grasp site."""
    import mujoco

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "wand_grasp")
    return data.site_xpos[site_id].copy()


def synthetic_camera_look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = np.array([0.0, 0.0, 1.0]),
    fovy_deg: float = 90.0,
    width: int = 640,
    height: int = 480,
):
    """Build (K, pos, R) for a synthetic camera looking at *target* from *eye*.

    Camera frame (right-handed):
      +z = eye → target (forward)
      +x = up × z (right)
      +y = z × x (down)
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    rotation = np.column_stack([x, y, z])

    fy = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    fx = fy
    intrinsic = np.array(
        [[fx, 0, width / 2.0], [0, fy, height / 2.0], [0, 0, 1.0]], dtype=np.float64
    )

    return intrinsic, eye.copy(), rotation


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------


def _emit(msg: str = "") -> None:
    """Write a line to stdout."""
    sys.stdout.write(msg + "\n")


def validate(plane_z: float | None = None):
    """Run the full validation pipeline.

    Args:
        plane_z: Override z for the virtual table plane.  When None, the
                 plane is placed at the wand body z (simulating the wand lying
                 ON a table at that height).  For a real tabletop scenario,
                 pass the calibrated table z.
    """
    from hey_robot.motion.table_plane import ray_plane_intersection

    _emit("=" * 60)
    _emit("GRASP ALGORITHM — SIMULATION VALIDATION")
    _emit("=" * 60)

    # --- 1. Load scene -------------------------------------------------------
    _emit("\n[1/5] Loading MuJoCo scene ...")
    t0 = time.time()
    model, data = setup_mujoco()
    _emit(
        f"  Loaded in {time.time() - t0:.2f}s  nbody={model.nbody}  "
        f"timestep={model.opt.timestep:.4f}"
    )

    # --- 2. Camera geometry --------------------------------------------------
    _emit("\n[2/5] Camera geometry ...")
    cam_intrinsic, cam_pos, cam_rotation = get_camera_geometry(model, data, "front")
    _emit("  front camera (FYI):")
    _emit(f"    fovy   = {model.cam_fovy[0]:.1f} deg")
    _emit(f"    pos    = [{cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}]")
    _emit(
        f"    view   = [{cam_rotation[0, 2]:.3f}, {cam_rotation[1, 2]:.3f}, {cam_rotation[2, 2]:.3f}]"
    )
    _wrist_intrinsic, wrist_pos, _wrist_rotation = get_camera_geometry(
        model, data, "right_wrist"
    )
    _emit("  right_wrist camera (FYI):")
    _emit(f"    fovy   = {model.cam_fovy[1]:.1f} deg")
    _emit(f"    pos    = [{wrist_pos[0]:.3f}, {wrist_pos[1]:.3f}, {wrist_pos[2]:.3f}]")

    # --- 3. Ground truth -----------------------------------------------------
    _emit("\n[3/5] Reading ground truth positions ...")
    wand_body = get_wand_position(model, data)
    wand_grasp = get_wand_grasp_position(model, data)
    _emit(
        f"  wand body      = [{wand_body[0]:.4f}, {wand_body[1]:.4f}, "
        f"{wand_body[2]:.4f}]"
    )
    _emit(
        f"  wand_grasp site = [{wand_grasp[0]:.4f}, {wand_grasp[1]:.4f}, "
        f"{wand_grasp[2]:.4f}]"
    )

    # --- 3b. Synthetic camera -----------------------------------------------
    # The real front/right_wrist cameras don't face the wand in this scene.
    # Construct a synthetic overhead camera looking at the wand midpoint to
    # validate the ray-plane intersection math (camera-agnostic geometry).
    wand_mid = (wand_body + wand_grasp) / 2.0
    eye = wand_mid + np.array([0.0, -0.35, 0.5], dtype=np.float64)
    cam_intrinsic, cam_pos, cam_rotation = synthetic_camera_look_at(
        eye,
        wand_mid,
        fovy_deg=75.0,
    )
    _emit("\n  synth camera (used for validation):")
    _emit(f"    eye    = [{cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}]")
    _emit(f"    target = [{wand_mid[0]:.3f}, {wand_mid[1]:.3f}, {wand_mid[2]:.3f}]")
    _emit("    fovy   = 75.0 deg")

    # --- 4. Virtual table plane ----------------------------------------------
    effective_z = plane_z if plane_z is not None else float(wand_body[2])
    # Horizontal plane at effective_z: 0*x + 0*y + 1*z - effective_z = 0
    plane = (0.0, 0.0, 1.0, -effective_z)
    _emit(f"\n[4/5] Virtual table plane: z = {effective_z:.4f} m")

    # --- 5. Simulate detection + recover 3D ----------------------------------
    # The ray-plane intersection recovers the 3D point where a camera ray hits
    # the table plane.  It is ONLY correct for points that lie ON the plane
    # (e.g. the base of an object touching the table).  Points above the plane
    # (like wand_grasp at z+80mm) will show an error because their ray
    # intersects the plane at a different XY — this is geometrically expected
    # and does NOT indicate an algorithm bug.
    #
    # We test three on-plane targets:
    #   1. wand_body — the wand's contact point with the dock (~table)
    #   2. synth_plane_pt — a second on-plane point at a different XY
    #   3. wand_body_off_axis — wand body seen from a more oblique camera
    _emit("\n[5/5] Simulating detection & recovering 3D ...")

    _on_plane_gt = np.array([wand_body[0], wand_body[1], effective_z], dtype=np.float64)

    all_errors_mm = []

    # --- 5a. Primary camera, wand_body (on-plane) ---
    pixel = project_3d_to_2d(wand_body, cam_intrinsic, cam_pos, cam_rotation)
    if pixel is not None:
        u, v = pixel
        bbox = (u - 30, v - 30, u + 30, v)
        est = ray_plane_intersection(
            ((bbox[0] + bbox[2]) / 2.0, bbox[3]),
            cam_intrinsic,
            cam_pos,
            cam_rotation,
            plane,
        )
        if est is not None:
            err = float(np.linalg.norm(est - wand_body)) * 1000.0
            all_errors_mm.append(err)
            _emit(
                f"  wand_body      err={err:5.1f} mm  "
                f"gt=[{wand_body[0]:.4f}, {wand_body[1]:.4f}, {wand_body[2]:.4f}]  "
                f"est=[{est[0]:.4f}, {est[1]:.4f}, {est[2]:.4f}]  "
                f"{'PASS' if err <= PASS_THRESHOLD_M * 1000 else 'FAIL'}"
            )

    # --- 5b. Primary camera, second on-plane point ---
    synth_pt = np.array(
        [wand_body[0] + 0.04, wand_body[1] + 0.03, effective_z], dtype=np.float64
    )
    pixel2 = project_3d_to_2d(synth_pt, cam_intrinsic, cam_pos, cam_rotation)
    if pixel2 is not None:
        u, v = pixel2
        bbox = (u - 30, v - 30, u + 30, v)
        est = ray_plane_intersection(
            ((bbox[0] + bbox[2]) / 2.0, bbox[3]),
            cam_intrinsic,
            cam_pos,
            cam_rotation,
            plane,
        )
        if est is not None:
            err = float(np.linalg.norm(est - synth_pt)) * 1000.0
            all_errors_mm.append(err)
            _emit(
                f"  synth_plane_pt err={err:5.1f} mm  "
                f"gt=[{synth_pt[0]:.4f}, {synth_pt[1]:.4f}, {synth_pt[2]:.4f}]  "
                f"est=[{est[0]:.4f}, {est[1]:.4f}, {est[2]:.4f}]  "
                f"{'PASS' if err <= PASS_THRESHOLD_M * 1000 else 'FAIL'}"
            )

    # --- 5c. Off-axis camera (more oblique angle → larger perspective effect) ---
    # Place camera to the side, closer to the plane, to make the test harder.
    off_eye = wand_body + np.array([-0.2, -0.15, 0.35], dtype=np.float64)
    off_intrinsic, off_pos, off_rotation = synthetic_camera_look_at(
        off_eye,
        wand_body,
        fovy_deg=70.0,
    )
    pixel3 = project_3d_to_2d(wand_body, off_intrinsic, off_pos, off_rotation)
    if pixel3 is not None:
        u, v = pixel3
        bbox = (u - 30, v - 30, u + 30, v)
        est = ray_plane_intersection(
            ((bbox[0] + bbox[2]) / 2.0, bbox[3]),
            off_intrinsic,
            off_pos,
            off_rotation,
            plane,
        )
        if est is not None:
            err = float(np.linalg.norm(est - wand_body)) * 1000.0
            all_errors_mm.append(err)
            _emit(
                f"  off_axis_cam   err={err:5.1f} mm  "
                f"gt=[{wand_body[0]:.4f}, {wand_body[1]:.4f}, {wand_body[2]:.4f}]  "
                f"est=[{est[0]:.4f}, {est[1]:.4f}, {est[2]:.4f}]  "
                f"{'PASS' if err <= PASS_THRESHOLD_M * 1000 else 'FAIL'}"
            )

    # --- 5d. wand_grasp (ABOVE plane — error expected, shown for reference) ---
    pixel_g = project_3d_to_2d(wand_grasp, cam_intrinsic, cam_pos, cam_rotation)
    if pixel_g is not None:
        u, v = pixel_g
        bbox = (u - 30, v - 30, u + 30, v)
        est = ray_plane_intersection(
            ((bbox[0] + bbox[2]) / 2.0, bbox[3]),
            cam_intrinsic,
            cam_pos,
            cam_rotation,
            plane,
        )
        if est is not None:
            err = float(np.linalg.norm(est - wand_grasp)) * 1000.0
            # Also show error vs. its on-plane projection (the "correct" answer)
            proj_gt = np.array(
                [wand_grasp[0], wand_grasp[1], effective_z], dtype=np.float64
            )
            err_vs_proj = float(np.linalg.norm(est - proj_gt)) * 1000.0
            _emit(
                f"  wand_grasp↑    err={err:5.1f} mm  (vs on-plane proj: {err_vs_proj:.1f} mm) "
                f"gt=[{wand_grasp[0]:.4f}, {wand_grasp[1]:.4f}, {wand_grasp[2]:.4f}]  "
                f"est=[{est[0]:.4f}, {est[1]:.4f}, {est[2]:.4f}]  "
                f"[above plane — not scored]"
            )

    # --- 6. Report -----------------------------------------------------------
    _emit("\n" + "=" * 60)
    if not all_errors_mm:
        _emit("RESULT: FAIL — no valid test points")
        return False

    mean_err = np.mean(all_errors_mm)
    max_err = np.max(all_errors_mm)
    passed = all(e <= PASS_THRESHOLD_M * 1000 for e in all_errors_mm)
    _emit(f"  Mean error: {mean_err:.1f} mm")
    _emit(f"  Max  error: {max_err:.1f} mm")
    _emit(f"  Threshold:  {PASS_THRESHOLD_M * 1000:.0f} mm")
    _emit(f"  RESULT:     {'PASS' if passed else 'FAIL'}")
    _emit("=" * 60)

    # --- 7. Also test density-cluster sampling --------------------------------
    _emit("\n[DENSITY CLUSTER TEST]")
    from hey_robot.motion.grasp_point import sample_grasp_point

    def simulated_detector():
        """Simulate detector noise on wand_body (on-plane contact point)."""
        pixel = project_3d_to_2d(wand_body, cam_intrinsic, cam_pos, cam_rotation)
        if pixel is None:
            return None
        noise_u = np.random.normal(0, 3)  # 3 px std
        noise_v = np.random.normal(0, 3)
        u = pixel[0] + noise_u
        v = pixel[1] + noise_v
        return (u - 30, v - 30, u + 30, v)  # 60×60 bbox

    cluster_point = sample_grasp_point(
        simulated_detector,
        cam_intrinsic,
        cam_pos,
        cam_rotation,
        plane,
        sample_count=20,
        sample_interval=0.0,  # no sleep in sim
    )
    if cluster_point is not None:
        cluster_err = float(np.linalg.norm(cluster_point - wand_body)) * 1000
        _emit(
            f"  Density-cluster result: [{cluster_point[0]:.4f}, "
            f"{cluster_point[1]:.4f}, {cluster_point[2]:.4f}]"
        )
        _emit(f"  Error vs wand_body: {cluster_err:.1f} mm")
        _emit(f"  {'PASS' if cluster_err <= PASS_THRESHOLD_M * 1000 else 'FAIL'}")
    else:
        _emit("  Density-cluster sampling failed (not enough valid samples)")

    return passed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    try:
        success = validate()
    except ImportError as exc:
        _emit(f"\nFATAL: missing dependency — {exc}")
        _emit("Install mujoco:  pip install mujoco")
        sys.exit(1)
    except FileNotFoundError as exc:
        _emit(f"\nFATAL: file not found — {exc}")
        sys.exit(1)
    except Exception as exc:
        _emit(f"\nFATAL: {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    sys.exit(0 if success else 1)
