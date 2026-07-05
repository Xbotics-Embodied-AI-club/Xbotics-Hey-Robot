from __future__ import annotations

import pytest

from hey_robot.config import RobotSpec
from hey_robot.protocol import RobotSkillAction, SkillIntent
from hey_robot.robot_runtime.base import RobotDriverContext
from hey_robot.robot_runtime.simulation.so101_mobile import So101MobileSimDriver


def _spec(**overrides):
    defaults = {
        "type": "so101_mobile_sim",
        "family": "so101_mobile",
        "environment": "sim",
        "driver": "mujoco",
        "embodiment_profile": "so101_mobile_sim",
        "settings": {"mjcf_path": "assets/robots/xlerobot/dock_scene.xml"},
    }
    defaults.update(overrides)
    return RobotSpec(**defaults)


def _context(spec=None):
    if spec is None:
        spec = _spec()
    return RobotDriverContext(
        robot_id="test",
        spec=spec,
        deployment_id="test",
        embodiment=None,
        skill_catalog=None,
    )


def _action(name, arguments):
    return RobotSkillAction(name=name, arguments=arguments).to_robot_action(
        SkillIntent(name, name)
    )


@pytest.mark.asyncio
async def test_driver_lifecycle_and_capabilities():
    driver = So101MobileSimDriver(_context())
    await driver.start()
    try:
        caps = await driver.capabilities()
        assert caps.driver_type == "so101_mobile_sim"
        assert "front" in caps.cameras
        assert "right_wrist" in caps.cameras
        assert caps.metadata["perception_source"] == "mujoco_ground_truth"
        assert caps.metadata["grasp_model"] == "conditional_weld"

        obs = await driver.observe()
        assert len(obs.assets) == 2
        assert obs.metadata["oracle"]["simulation_only"] is True

        health = await driver.health()
        assert health.online is True
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_driver_ik_reachable():
    driver = So101MobileSimDriver(_context())
    await driver.start()
    try:
        status = await driver.apply_action(
            _action("arm_solve_position_ik", {"target_xyz": [0.35, 0.05, 0.60]})
        )
        assert status.success is True
        result = driver.last_skill_result.to_dict()
        assert result["operation_success"] is True
        assert len(result["joint_positions"]) == 5
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_driver_ik_unreachable():
    driver = So101MobileSimDriver(_context())
    await driver.start()
    try:
        status = await driver.apply_action(
            _action("arm_solve_position_ik", {"target_xyz": [5.0, 5.0, 5.0]})
        )
        assert status.success is True  # action applied, no runtime error
        result = driver.last_skill_result.to_dict()
        assert result["operation_success"] is False
        assert result["failure_mode"] == "ik_unreachable"
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_driver_oracle_locate_wand():
    driver = So101MobileSimDriver(_context())
    await driver.start()
    try:
        status = await driver.apply_action(
            _action("sim_locate_object", {"query": "wand", "sample_count": 1})
        )
        assert status.success is True
        result = driver.last_skill_result.to_dict()
        assert result["object_name"] == "cat_wand"
        assert len(result["samples"]) == 1
        assert len(result["samples"][0]) == 3
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_driver_oracle_caption():
    driver = So101MobileSimDriver(_context())
    await driver.start()
    try:
        status = await driver.apply_action(_action("inspect_scene", {}))
        assert status.success is True
        result = driver.last_skill_result.to_dict()
        assert "cat wand" in result["oracle"]["caption"].lower()
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_driver_set_gripper_cycle():
    driver = So101MobileSimDriver(_context())
    await driver.start()
    try:
        await driver.apply_action(_action("set_gripper", {"action": "open"}))
        assert driver.gripper.held_object is None

        await driver.apply_action(_action("set_gripper", {"action": "close"}))
        assert driver.gripper.held_object is None
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_driver_reset_posture():
    driver = So101MobileSimDriver(_context())
    await driver.start()
    try:
        await driver.apply_action(_action("reset_posture", {}))
        assert driver.state == "idle"
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_driver_stop_motion():
    driver = So101MobileSimDriver(_context())
    await driver.start()
    try:
        await driver.apply_action(_action("stop_motion", {}))
        assert driver.state == "idle"
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_driver_reset():
    driver = So101MobileSimDriver(_context())
    await driver.start()
    try:
        await driver.reset()
        assert driver.state == "idle"
        assert driver.last_error is None
    finally:
        await driver.close()
