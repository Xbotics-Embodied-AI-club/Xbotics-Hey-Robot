from __future__ import annotations

from pathlib import Path

import pytest

from hey_robot.config import DeploymentConfig, RobotSpec
from hey_robot.protocol import Envelope, RobotSkillAction, SkillIntent
from hey_robot.robot_runtime.base import RobotDriverContext
from hey_robot.robot_runtime.embodiments import get_embodiment_profile
from hey_robot.robot_runtime.manager import RobotManager
from hey_robot.robot_runtime.simulation.so101_tabletop import (
    So101TabletopSimDriver,
)


def _context() -> RobotDriverContext:
    spec = RobotSpec(
        type="so101_sim",
        family="so101",
        environment="sim",
        driver="mujoco",
        embodiment_profile="so101_tabletop_sim",
        settings={"mjcf_path": "assets/robots/so101_tabletop/scene.xml"},
    )
    return RobotDriverContext(
        robot_id="tabletop",
        spec=spec,
        deployment_id="test",
        embodiment=get_embodiment_profile(spec),
    )


def _action(name: str, arguments: dict | None = None):
    return RobotSkillAction(name, arguments or {}).to_robot_action(
        SkillIntent(envelope=Envelope(robot_id="tabletop"), name=name)
    )


def test_manager_routes_so101_mujoco_to_tabletop_driver() -> None:
    config = DeploymentConfig.from_yaml(Path("configs/so101.tabletop.sim.yaml"))
    driver = RobotManager(config).require("tabletop_arm")
    assert isinstance(driver, So101TabletopSimDriver)


@pytest.mark.asyncio
async def test_driver_lifecycle_capabilities_observation_and_ik() -> None:
    driver = So101TabletopSimDriver(_context())
    await driver.start()
    try:
        capabilities = await driver.capabilities()
        assert capabilities.cameras == ["overhead", "front", "side"]
        assert capabilities.metadata["perception_source"] == "mujoco_ground_truth"
        observation = await driver.observe()
        assert len(observation.assets) == 3
        assert observation.metadata["oracle"]["simulation_only"] is True

        status = await driver.apply_action(
            _action("arm_solve_position_ik", {"target_xyz": [0.25, 0, 0.1]})
        )
        assert status.success is True
        assert driver.last_skill_result is not None
        assert driver.last_skill_result.data["operation_success"] is True
        assert len(driver.last_skill_result.data["joint_positions"]) == 5
    finally:
        await driver.close()
