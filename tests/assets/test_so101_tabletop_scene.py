from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

SCENE = Path("assets/robots/so101_tabletop/scene.xml").resolve()


def test_scene_dimensions_and_named_contract() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    assert (model.nq, model.nv, model.nu, model.neq) == (48, 42, 6, 6)
    assert (model.nbody, model.ncam) == (15, 3)

    for name in (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "jaw_visual_joint",
    ):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0
    for name in (
        "act_shoulder_pan",
        "act_shoulder_lift",
        "act_elbow_flex",
        "act_wrist_flex",
        "act_wrist_roll",
        "act_jaw_visual",
    ):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) >= 0
    for name in ("overhead", "front", "side"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0
    for name in ("banana", "mug", "bottle", "screwdriver", "duck", "lego"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site") >= 0


def test_scene_is_repository_local_and_manifested() -> None:
    text = SCENE.read_text(encoding="utf-8")
    assert 'meshdir="meshes/"' in text
    assert "/home/" not in text
    assert len(list((SCENE.parent / "meshes").glob("*.stl"))) == 19
    assert (SCENE.parent / "SOURCE.md").is_file()
    assert (SCENE.parent / "SHA256SUMS").is_file()
    assert (SCENE.parent / "LICENSE-APACHE-2.0").is_file()
    assert (SCENE.parent / "NOTICE").is_file()


def test_scene_stays_finite_for_ten_simulated_seconds() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    for _ in range(int(10.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.qvel))
    for name in ("banana", "mug", "bottle", "screwdriver", "duck", "lego"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert 0.0 < float(data.xpos[body_id, 2]) < 0.2
