from __future__ import annotations

import argparse
import asyncio

from hey_robot.config import DeploymentConfig
from hey_robot.robot_runtime import RobotService
from hey_robot.skill_os.registry import registry_from_config


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Hey Robot robot driver service")
    parser.add_argument("--config", required=True, help="Deployment YAML path")
    args = parser.parse_args()

    config = DeploymentConfig.from_yaml(args.config)
    service = RobotService(
        config,
        skill_catalog=registry_from_config(config).robot_skill_catalog(),
    )
    try:
        await service.start()
    finally:
        await service.stop()


def main() -> None:
    asyncio.run(async_main())
