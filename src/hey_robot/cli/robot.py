from __future__ import annotations

import argparse
import asyncio

from hey_robot.config import DeploymentConfig
from hey_robot.robot_transport import RobotService
from hey_robot.skills import robot_action_specs_from_config


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Hey Robot robot driver service")
    parser.add_argument("--config", required=True, help="Deployment YAML path")
    args = parser.parse_args()

    config = DeploymentConfig.from_yaml(args.config)
    service = RobotService(
        config,
        action_specs=robot_action_specs_from_config(config),
    )
    try:
        await service.start()
    finally:
        await service.stop()


def main() -> None:
    asyncio.run(async_main())
