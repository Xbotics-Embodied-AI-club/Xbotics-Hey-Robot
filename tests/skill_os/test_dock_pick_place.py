from __future__ import annotations

import pytest

from hey_robot.config import RobotSpec
from hey_robot.protocol import RobotSkillAction, SkillIntent
from hey_robot.robot_runtime.base import RobotDriverContext
from hey_robot.robot_runtime.simulation.so101_mobile import So101MobileSimDriver
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
