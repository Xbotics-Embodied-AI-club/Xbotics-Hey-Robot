from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from hey_robot.config.model import RobotSpec
from hey_robot.robot_backends.simulation.xlerobot_sim_driver import XLeRobotSimDriver
from hey_robot.robot_runtime.manager import create_driver_context

SCENE_PATH = (
    Path(__file__).resolve().parents[2] / "assets" / "scenes" / "home_scene.xml"
)


def _body_position(model, data, name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert body_id >= 0, f"missing body: {name}"
    return np.array(data.xpos[body_id], dtype=float)


def _geom_position(model, data, name: str) -> np.ndarray:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert geom_id >= 0, f"missing geom: {name}"
    return np.array(data.geom_xpos[geom_id], dtype=float)


def _site_position(model, data, name: str) -> np.ndarray:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    assert site_id >= 0, f"missing site: {name}"
    return np.array(data.site_xpos[site_id], dtype=float)


def _geom_bounds(model, data, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = model.geom_dataid[geom_id]
        start = model.mesh_vertadr[mesh_id]
        count = model.mesh_vertnum[mesh_id]
        local_vertices = model.mesh_vert[start : start + count]
    else:
        size = model.geom_size[geom_id]
        local_vertices = np.array(
            [
                [x, y, z]
                for x in (-size[0], size[0])
                for y in (-size[1], size[1])
                for z in (-size[2], size[2])
            ],
            dtype=float,
        )
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    vertices = data.geom_xpos[geom_id] + local_vertices @ rotation.T
    return vertices.min(axis=0), vertices.max(axis=0)


def _body_bounds(model, data, name: str) -> tuple[np.ndarray, np.ndarray]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert body_id >= 0, f"missing body: {name}"
    bounds = []
    start = model.body_geomadr[body_id]
    for geom_id in range(start, start + model.body_geomnum[body_id]):
        bounds.extend(_geom_bounds(model, data, geom_id))
    vertices = np.array(bounds)
    return vertices.min(axis=0), vertices.max(axis=0)


def _named_geom_bounds(model, data, name: str) -> tuple[np.ndarray, np.ndarray]:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert geom_id >= 0, f"missing geom: {name}"
    return _geom_bounds(model, data, geom_id)


def _bounds_center(bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    minimum, maximum = bounds
    return (minimum + maximum) / 2.0


def _assert_xy_bounds_inside(
    bounds: tuple[np.ndarray, np.ndarray],
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> None:
    minimum, maximum = bounds
    assert minimum[0] >= x_range[0] - 0.01
    assert maximum[0] <= x_range[1] + 0.01
    assert minimum[1] >= y_range[0] - 0.01
    assert maximum[1] <= y_range[1] + 0.01


def test_living_room_assets_stay_inside_west_wall() -> None:
    driver = _home_scene_driver()

    for geom_name in ("rug_living_carpet", "rug_living_darker"):
        minimum, _ = _named_geom_bounds(driver.model, driver.data, geom_name)
        assert minimum[0] >= 0.0

    for body_name in ("tv_console", "tv_screen"):
        minimum, _ = _body_bounds(driver.model, driver.data, body_name)
        assert minimum[0] >= 0.0


def test_living_room_sofa_is_centered_on_rug() -> None:
    driver = _home_scene_driver()

    rug_center = _bounds_center(
        _named_geom_bounds(driver.model, driver.data, "rug_living_carpet")
    )
    sofa_center = _bounds_center(_body_bounds(driver.model, driver.data, "sofa_main"))

    assert sofa_center[:2] == pytest.approx(rug_center[:2], abs=0.02)


def test_home_scene_rugs_are_inside_their_rooms() -> None:
    driver = _home_scene_driver()

    room_bounds = {
        "rug_living_carpet": ((0.0, 6.0), (0.0, 5.0)),
        "rug_living_darker": ((0.0, 6.0), (0.0, 5.0)),
        "rug_hall_carpet": ((6.0, 14.0), (0.0, 10.0)),
        "rug_hall_darker": ((6.0, 14.0), (0.0, 10.0)),
        "rug_study_carpet": ((14.0, 20.0), (5.0, 10.0)),
        "rug_study_darker": ((14.0, 20.0), (5.0, 10.0)),
        "rug_master_carpet": ((0.0, 7.0), (10.0, 14.0)),
        "rug_master_darker": ((0.0, 7.0), (10.0, 14.0)),
    }
    for geom_name, (x_range, y_range) in room_bounds.items():
        _assert_xy_bounds_inside(
            _named_geom_bounds(driver.model, driver.data, geom_name),
            x_range=x_range,
            y_range=y_range,
        )

    hall_center = _bounds_center(
        _named_geom_bounds(driver.model, driver.data, "rug_hall_carpet")
    )
    assert hall_center[:2] == pytest.approx([10.0, 5.0], abs=0.02)


def test_dining_chairs_face_the_table() -> None:
    driver = _home_scene_driver()
    table_center = _bounds_center(
        _body_bounds(driver.model, driver.data, "dining_table")
    )

    for chair_name in ("dchair1", "dchair2", "dchair3", "dchair4"):
        chair_id = mujoco.mj_name2id(driver.model, mujoco.mjtObj.mjOBJ_BODY, chair_name)
        assert chair_id >= 0, f"missing body: {chair_name}"
        chair_pos = np.array(driver.data.xpos[chair_id], dtype=float)
        direction_to_table = table_center[:2] - chair_pos[:2]
        direction_to_table /= np.linalg.norm(direction_to_table)
        chair_forward = driver.data.xmat[chair_id].reshape(3, 3)[:2, 1]
        chair_forward /= np.linalg.norm(chair_forward)
        assert float(np.dot(chair_forward, direction_to_table)) > 0.95


def test_home_scene_visible_assets_do_not_sink_below_floor() -> None:
    driver = _home_scene_driver()
    failures: list[str] = []

    for geom_id in range(driver.model.ngeom):
        geom_name = mujoco.mj_id2name(driver.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name in {"floor", "floor_raycast"}:
            continue
        if driver.model.geom_rgba[geom_id][3] == 0.0:
            continue
        minimum, _ = _geom_bounds(driver.model, driver.data, geom_id)
        if minimum[2] < -0.01:
            body_id = driver.model.geom_bodyid[geom_id]
            body_name = mujoco.mj_id2name(
                driver.model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            failures.append(
                f"{body_name or '<world>'}/{geom_name or geom_id}: {minimum[2]:.3f}"
            )

    assert failures == []


def test_home_scene_mounts_wand_dock_to_robot_body() -> None:
    driver = _home_scene_driver()

    dock0 = _body_position(driver.model, driver.data, "wand_dock")
    wand0 = _body_position(driver.model, driver.data, "wand")

    driver._step_velocity(50, 0.0, 0.2, 0.0)

    dock1 = _body_position(driver.model, driver.data, "wand_dock")
    wand1 = _body_position(driver.model, driver.data, "wand")
    dock_delta = dock1 - dock0
    wand_delta = wand1 - wand0

    assert dock_delta[0] > 0.01
    assert wand_delta[0] > 0.01
    assert wand_delta == pytest.approx(dock_delta, abs=3e-3)
    assert wand1 - dock1 == pytest.approx([0.0, 0.0, 0.105], abs=3e-3)
    assert _geom_position(driver.model, driver.data, "wand_ball")[0] > (
        _site_position(driver.model, driver.data, "wand_grasp")[0] + 0.15
    )


def test_home_scene_can_pick_wand_from_dock_with_oracle_path() -> None:
    driver = _home_scene_driver()
    assert driver._dock_arm_side == "right"
    assert driver._dock_arm.joint_names == (
        "Rotation_2",
        "Pitch_2",
        "Elbow_2",
        "Wrist_Pitch_2",
        "Wrist_Roll_2",
    )

    locate = _primitive(driver, "sim_locate_object", {"query": "wand"})
    grasp_point = locate.data["samples"][0]
    grasp_axis = locate.data["grasp_axis"]
    assert grasp_point == pytest.approx([0.135, 0.0, 0.8377], abs=1e-3)
    assert grasp_axis == pytest.approx([0.9404, 0.0, 0.3401], abs=1e-3)

    pre_grasp_point = [grasp_point[0], grasp_point[1], grasp_point[2] + 0.05]
    pre_grasp = _primitive(
        driver, "arm_solve_position_ik", {"target_xyz": pre_grasp_point}
    ).data["joint_positions"]
    grasp = _primitive(
        driver,
        "arm_solve_position_ik",
        {
            "target_xyz": grasp_point,
            "target_axis": grasp_axis,
            "current_joints": pre_grasp,
        },
    ).data["joint_positions"]

    _primitive(driver, "set_gripper", {"action": "open"})
    _primitive(driver, "move_arm_joints", {"joints": _joint_payload(pre_grasp)})
    _primitive(driver, "move_arm_joints", {"joints": _joint_payload(grasp)})
    close = _primitive(driver, "set_gripper", {"action": "close"})
    closed_position = _body_position(driver.model, driver.data, "wand")
    _primitive(driver, "move_arm_joints", {"joints": _joint_payload(pre_grasp)})
    lifted_position = _body_position(driver.model, driver.data, "wand")

    assert close.data["held_object"] == "wand"
    assert close.data["welds"]["wand"] is True
    assert lifted_position[2] > closed_position[2] + 0.03


def _home_scene_driver() -> XLeRobotSimDriver:
    spec = RobotSpec(
        type="xlerobot_sim",
        family="xlerobot",
        environment="sim",
        driver="mujoco",
        embodiment_profile="xlerobot_sim",
        settings={"mjcf_path": str(SCENE_PATH)},
    )
    driver = XLeRobotSimDriver(create_driver_context("sim_robot", spec, "test"))
    driver.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    driver.data = mujoco.MjData(driver.model)
    mujoco.mj_forward(driver.model, driver.data)
    driver._initialize_dock_manipulation()
    return driver


def _primitive(driver: XLeRobotSimDriver, name: str, arguments: dict):
    from hey_robot.protocol import RobotSkillAction

    result = driver._execute_dock_primitive(
        RobotSkillAction(name=name, arguments=arguments)
    )
    assert result is not None
    assert result.success
    if "operation_success" in result.data:
        assert result.data["operation_success"] is True
    return result


def _joint_payload(joints: list[float]) -> dict[str, float]:
    return dict(
        zip(
            ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"),
            joints,
            strict=True,
        )
    )
