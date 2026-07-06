from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

SCENE_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "robots"
    / "xlerobot"
    / "dock_scene.xml"
)


def test_scene_contains_dock_and_wand():
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    for name in ("wand_dock", "cat_wand"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert body_id >= 0, f"missing body: {name}"


def test_single_arm_scene_has_arm_joints():
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    for name in (
        "Rotation",
        "Pitch",
        "Elbow",
        "Wrist_Pitch",
        "Wrist_Roll",
        "Jaw",
    ):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert joint_id >= 0, f"missing joint: {name}"


def test_scene_has_cameras():
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    for name in ("front", "right_wrist"):
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        assert cam_id >= 0, f"missing camera: {name}"


def test_single_arm_scene_has_left_actuators():
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    for name in (
        "Rotation_L",
        "Pitch_L",
        "Elbow_L",
        "Wrist_Pitch_L",
        "Wrist_Roll_L",
        "Jaw_L",
    ):
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert act_id >= 0, f"missing actuator: {name}"


def test_scene_has_weld_constraint():
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    assert model.neq >= 1, "missing weld equality constraint"


def test_wand_stays_in_dock_for_three_seconds():
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    wand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cat_wand")
    dt = float(model.opt.timestep)
    for _ in range(int(3.0 / dt)):
        mujoco.mj_step(model, data)
    pos = data.xpos[wand_id]
    assert np.isfinite(pos).all(), "wand position is not finite"
    assert 0.03 < pos[2] < 1.0, f"wand z={pos[2]:.3f} out of expected range"
