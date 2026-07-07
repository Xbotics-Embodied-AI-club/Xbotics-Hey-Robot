from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from hey_robot.config.model import RobotSpec
from hey_robot.robot_runtime.base import RobotDriverContext
from hey_robot.robot_runtime.embodiments import get_embodiment_profile
from hey_robot.robot_runtime.simulation.xlerobot_sim_driver import XLeRobotSimDriver

SCENE_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "scenes"
    / "cat_play_home_scene.xml"
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


def test_cat_play_home_scene_mounts_wand_dock_to_robot_body() -> None:
    driver = _cat_play_driver()

    dock0 = _body_position(driver.model, driver.data, "wand_dock")
    wand0 = _body_position(driver.model, driver.data, "cat_wand")

    driver._step_velocity(50, 0.0, 0.2, 0.0)

    dock1 = _body_position(driver.model, driver.data, "wand_dock")
    wand1 = _body_position(driver.model, driver.data, "cat_wand")
    dock_delta = dock1 - dock0
    wand_delta = wand1 - wand0

    assert dock_delta[0] > 0.01
    assert wand_delta[0] > 0.01
    assert wand_delta == pytest.approx(dock_delta, abs=3e-3)
    assert wand1 - dock1 == pytest.approx([0.0, 0.0, 0.105], abs=3e-3)
    assert _geom_position(driver.model, driver.data, "wand_ball")[0] > (
        _site_position(driver.model, driver.data, "wand_grasp")[0] + 0.15
    )


def test_cat_play_home_scene_can_pick_wand_from_dock_with_oracle_path() -> None:
    driver = _cat_play_driver()
    assert driver._dock_arm_side == "right"
    assert driver._dock_arm.joint_names == (
        "Rotation_2",
        "Pitch_2",
        "Elbow_2",
        "Wrist_Pitch_2",
        "Wrist_Roll_2",
    )

    locate = _primitive(driver, "sim_locate_object", {"query": "cat_wand"})
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
    closed_position = _body_position(driver.model, driver.data, "cat_wand")
    _primitive(driver, "move_arm_joints", {"joints": _joint_payload(pre_grasp)})
    lifted_position = _body_position(driver.model, driver.data, "cat_wand")

    assert close.data["held_object"] == "cat_wand"
    assert close.data["welds"]["cat_wand"] is True
    assert lifted_position[2] > closed_position[2] + 0.03


def _cat_play_driver() -> XLeRobotSimDriver:
    spec = RobotSpec(
        type="xlerobot_sim",
        family="xlerobot",
        environment="sim",
        driver="mujoco",
        embodiment_profile="xlerobot_sim",
        settings={"mjcf_path": str(SCENE_PATH)},
    )
    driver = XLeRobotSimDriver(
        RobotDriverContext(
            deployment_id="test",
            robot_id="sim_robot",
            spec=spec,
            embodiment=get_embodiment_profile(spec),
        )
    )
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
