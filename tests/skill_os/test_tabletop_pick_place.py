from __future__ import annotations

from typing import Any

import pytest

from hey_robot.config import RobotSpec
from hey_robot.protocol import Envelope, RobotSkillAction, SkillIntent
from hey_robot.robot_runtime.base import RobotDriverContext
from hey_robot.robot_runtime.embodiments import get_embodiment_profile
from hey_robot.robot_runtime.simulation.so101_tabletop import (
    So101TabletopSimDriver,
)
from hey_robot.skill_os.builtins.tabletop_manipulation import PickSkill, PlaceSkill
from hey_robot.skill_os.context import SkillContext


def _driver() -> So101TabletopSimDriver:
    spec = RobotSpec(
        type="so101_sim",
        family="so101",
        environment="sim",
        driver="mujoco",
        embodiment_profile="so101_tabletop_sim",
        settings={"mjcf_path": "assets/robots/so101_tabletop/scene.xml"},
    )
    return So101TabletopSimDriver(
        RobotDriverContext(
            robot_id="tabletop",
            spec=spec,
            deployment_id="test",
            embodiment=get_embodiment_profile(spec),
        )
    )


class DirectRobotPort:
    def __init__(self, driver: So101TabletopSimDriver) -> None:
        self.driver = driver

    async def run(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        action = RobotSkillAction(name, arguments or {}).to_robot_action(
            SkillIntent(envelope=Envelope(robot_id="tabletop"), name=name)
        )
        await self.driver.apply_action(action)
        assert self.driver.last_skill_result is not None
        return self.driver.last_skill_result.to_dict()


@pytest.mark.asyncio
async def test_pick_hold_and_place_front_right() -> None:
    driver = _driver()
    await driver.start()
    context = SkillContext(robot=DirectRobotPort(driver))
    try:
        picked = await PickSkill().execute(
            context,
            {"object_label": "banana", "mode": "hold", "max_retries": 1},
        )
        assert picked.success, picked
        assert picked.data["lift_m"] > 0.05
        assert driver.gripper.held_object == "banana"

        placed = await PlaceSkill().execute(context, {"location": "front_right"})
        assert placed.success, placed
        assert placed.data["xy_error_m"] < 0.06
        assert driver.gripper.held_object is None
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_pick_missing_object_fails_closed() -> None:
    driver = _driver()
    await driver.start()
    try:
        result = await PickSkill().execute(
            SkillContext(robot=DirectRobotPort(driver)),
            {"object_label": "laptop", "mode": "hold", "max_retries": 1},
        )
        assert result.success is False
        assert result.failure_mode == "object_not_found"
        assert driver.gripper.held_object is None
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_place_rejects_empty_gripper() -> None:
    driver = _driver()
    await driver.start()
    try:
        result = await PlaceSkill().execute(
            SkillContext(robot=DirectRobotPort(driver)), {"location": "front"}
        )
        assert result.success is False
        assert result.failure_mode == "gripper_empty"
    finally:
        await driver.close()
