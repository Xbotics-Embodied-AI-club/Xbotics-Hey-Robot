from __future__ import annotations

import pytest

from hey_robot.config import RobotSpec
from hey_robot.protocol import RobotSkillAction, SkillIntent
from hey_robot.robot_runtime import get_embodiment_profile
from hey_robot.robot_runtime.base import RobotDriverContext
from hey_robot.robot_runtime.simulation.so101_mobile import So101MobileSimDriver
from hey_robot.robot_runtime.simulation.xlerobot_sim_driver import XLeRobotSimDriver
from hey_robot.skill_os.builtins.dock_manipulation import PickWandSkill, PlaceWandSkill
from hey_robot.skill_os.context import SkillContext


def _spec():
    return RobotSpec(
        type="so101_mobile_sim",
        family="so101_mobile",
        environment="sim",
        driver="mujoco",
        embodiment_profile="so101_mobile_sim",
        settings={"mjcf_path": "assets/robots/xlerobot/dock_scene.xml"},
    )


def _context():
    return RobotDriverContext(
        robot_id="test",
        spec=_spec(),
        deployment_id="test",
        embodiment=None,
        skill_catalog=None,
    )


class DirectRobotPort:
    """Adapter that wraps driver.apply_action for SkillContext."""

    def __init__(self, driver):
        self.driver = driver

    async def run(self, name, arguments):
        action = RobotSkillAction(name=name, arguments=arguments).to_robot_action(
            SkillIntent(name, name)
        )
        await self.driver.apply_action(action)
        return self.driver.last_skill_result.to_dict()


def _dual_arm_driver() -> XLeRobotSimDriver:
    import mujoco

    scene = "assets/robots/xlerobot/cat_play_home_scene.xml"
    spec = RobotSpec(
        type="xlerobot_sim",
        family="xlerobot",
        environment="sim",
        driver="mujoco",
        embodiment_profile="xlerobot_sim",
        settings={"mjcf_path": scene},
    )
    context = RobotDriverContext(
        robot_id="dual_test",
        spec=spec,
        deployment_id="test",
        embodiment=get_embodiment_profile(spec),
        skill_catalog=None,
    )
    driver = XLeRobotSimDriver(context)
    driver.model = mujoco.MjModel.from_xml_path(scene)
    driver.data = mujoco.MjData(driver.model)
    for actuator_id, position in driver.adapter.arm_rest_positions().items():
        driver.data.ctrl[actuator_id] = position
        driver._set_actuator_joint_position(actuator_id, position)
    mujoco.mj_forward(driver.model, driver.data)
    driver._initialize_dock_manipulation()
    driver.state = "idle"
    return driver


@pytest.mark.asyncio
async def test_pick_wand_from_dock():
    driver = So101MobileSimDriver(_context())
    await driver.start()
    try:
        ctx = SkillContext(robot=DirectRobotPort(driver))
        result = await PickWandSkill().execute(
            ctx, {"object_label": "wand", "max_retries": 2}
        )
        assert result.success, f"pick failed: {result.failure_mode} - {result.error}"
        assert driver.gripper.held_object == "cat_wand"
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_pick_and_place_cycle():
    driver = So101MobileSimDriver(_context())
    await driver.start()
    try:
        ctx = SkillContext(robot=DirectRobotPort(driver))

        pick_result = await PickWandSkill().execute(
            ctx, {"object_label": "wand", "max_retries": 2}
        )
        assert pick_result.success, f"pick failed: {pick_result.failure_mode}"
        assert driver.gripper.held_object == "cat_wand"

        place_result = await PlaceWandSkill().execute(ctx, {})
        assert place_result.success, f"place failed: {place_result.failure_mode}"
        assert driver.gripper.held_object is None
        assert driver.gripper.weld_states().get("cat_wand", True) is False
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_dual_arm_xlerobot_left_arm_pick_and_place_cycle():
    driver = _dual_arm_driver()
    ctx = SkillContext(robot=DirectRobotPort(driver))

    pick_result = await PickWandSkill().execute(ctx, {})
    assert pick_result.success, pick_result.error
    assert driver._dock_gripper.held_object == "cat_wand"

    place_result = await PlaceWandSkill().execute(ctx, {})
    assert place_result.success, place_result.error
    assert driver._dock_gripper.held_object is None


@pytest.mark.asyncio
async def test_pick_missing_object_fails():
    driver = So101MobileSimDriver(_context())
    await driver.start()
    try:
        ctx = SkillContext(robot=DirectRobotPort(driver))
        result = await PickWandSkill().execute(
            ctx, {"object_label": "laptop", "max_retries": 1}
        )
        assert result.success is False
        assert result.failure_mode == "object_not_found"
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_place_without_held_object_fails():
    driver = So101MobileSimDriver(_context())
    await driver.start()
    try:
        ctx = SkillContext(robot=DirectRobotPort(driver))
        result = await PlaceWandSkill().execute(ctx, {})
        assert result.success is False
        assert result.failure_mode == "gripper_empty"
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_perceive_grasp_point_primitive():
    """perceive_grasp_point returns a 3D point via ray-plane intersection."""
    driver = So101MobileSimDriver(_context())
    await driver.start()
    try:
        result = driver._perceive_grasp_point(
            {
                "query": "wand",
                "camera": "front",
                "sample_count": 20,
                "sample_interval": 0.0,
            }
        )
        assert result.success is True
        point = result.data.get("point_3d")
        assert point is not None
        assert len(point) == 3
        assert result.data.get("source") == "ray_plane_intersection"

        import mujoco
        import numpy as np

        body_id = mujoco.mj_name2id(
            driver.session.model, mujoco.mjtObj.mjOBJ_BODY, "cat_wand"
        )
        gt = driver.session.data.xpos[body_id]
        error_mm = float(np.linalg.norm(np.array(point) - gt)) * 1000
        assert error_mm <= 25, f"perception error {error_mm:.1f}mm > 25mm threshold"
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_pick_wand_perception_mode():
    """PickWandSkill in perception mode successfully picks and holds the wand."""
    driver = So101MobileSimDriver(_context())
    await driver.start()
    try:
        ctx = SkillContext(robot=DirectRobotPort(driver))
        result = await PickWandSkill().execute(
            ctx,
            {
                "object_label": "wand",
                "max_retries": 2,
                "mode": "perception",
                "camera": "front",
            },
        )
        assert result.success, (
            f"perception pick failed: {result.failure_mode} - {result.error}"
        )
        assert result.data.get("source") == "ray_plane_intersection"
        assert driver.gripper.held_object == "cat_wand"
    finally:
        await driver.close()
