from __future__ import annotations

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


@pytest.mark.asyncio
async def test_pick_then_place_releases_target_at_requested_location() -> None:
    spec = RobotSpec(
        type="so101_sim",
        family="so101",
        environment="sim",
        driver="mujoco",
        embodiment_profile="so101_tabletop_sim",
        settings={"mjcf_path": "assets/robots/so101_tabletop/scene.xml"},
    )
    driver = So101TabletopSimDriver(
        RobotDriverContext(
            robot_id="tabletop",
            spec=spec,
            deployment_id="test",
            embodiment=get_embodiment_profile(spec),
        )
    )

    class Port:
        async def run(self, name, arguments=None):
            action = RobotSkillAction(name, arguments or {}).to_robot_action(
                SkillIntent(envelope=Envelope(robot_id="tabletop"), name=name)
            )
            await driver.apply_action(action)
            assert driver.last_skill_result is not None
            return driver.last_skill_result.to_dict()

    await driver.start()
    context = SkillContext(robot=Port())
    try:
        picked = await PickSkill().execute(
            context,
            {"object_label": "banana", "mode": "hold", "max_retries": 1},
        )
        assert picked.success, picked
        placed = await PlaceSkill().execute(context, {"location": "front_right"})
        assert placed.success, placed
        assert driver.gripper.held_object is None
        assert driver.gripper.weld_states()["banana"] is False
        x, y, _z = driver.arm.get_object_positions()["banana"]
        assert x == pytest.approx(0.30, abs=0.06)
        assert y == pytest.approx(-0.12, abs=0.06)
    finally:
        await driver.close()
