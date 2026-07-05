from __future__ import annotations

import pytest

from hey_robot.config import RobotSpec
from hey_robot.protocol import Envelope, RobotSkillAction, SkillIntent
from hey_robot.robot_runtime.base import RobotDriverContext
from hey_robot.robot_runtime.embodiments import get_embodiment_profile
from hey_robot.robot_runtime.simulation.so101_tabletop import (
    So101TabletopSimDriver,
)
from hey_robot.skill_os.builtins.tabletop_manipulation import PickSkill
from hey_robot.skill_os.context import SkillContext


@pytest.mark.asyncio
async def test_banana_is_welded_and_lifted_more_than_five_centimeters() -> None:
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
    try:
        initial_z = driver.arm.get_object_positions()["banana"][2]
        result = await PickSkill().execute(
            SkillContext(robot=Port()),
            {"object_label": "banana", "mode": "hold", "max_retries": 1},
        )
        final_z = driver.arm.get_object_positions()["banana"][2]
        assert result.success, result
        assert driver.gripper.weld_states()["banana"] is True
        assert driver.gripper.held_object == "banana"
        assert final_z - initial_z > 0.05
    finally:
        await driver.close()
