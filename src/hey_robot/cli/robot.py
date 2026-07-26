from __future__ import annotations

import argparse
import asyncio

from hey_robot.app.runtime_components import build_local_runtime_components
from hey_robot.config import DeploymentConfig
from hey_robot.robot_transport import RobotService
from hey_robot.skills import BusSkillServer, robot_action_specs_from_config


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Hey Robot robot driver service")
    parser.add_argument("--config", required=True, help="Deployment YAML path")
    args = parser.parse_args()

    config = DeploymentConfig.from_yaml(args.config)
    service = RobotService(
        config,
        action_specs=robot_action_specs_from_config(config),
    )
    components = build_local_runtime_components(config, robot_service=service)
    skill_server = BusSkillServer(config, components.skill_client)
    robot_task = asyncio.create_task(service.start(), name="robot-service")
    skill_task = asyncio.create_task(skill_server.start(), name="skill-server")
    try:
        await asyncio.gather(robot_task, skill_task)
    finally:
        robot_task.cancel()
        skill_task.cancel()
        await asyncio.gather(robot_task, skill_task, return_exceptions=True)
        await skill_server.close()
        await service.stop()


def main() -> None:
    asyncio.run(async_main())
