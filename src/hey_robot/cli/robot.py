from __future__ import annotations

import argparse
import asyncio

from hey_robot.config import DeploymentConfig
from hey_robot.robots import RobotService


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Hey Robot robot driver service")
    parser.add_argument("--config", required=True, help="Deployment YAML path")
    args = parser.parse_args()

    service = RobotService(DeploymentConfig.from_yaml(args.config))
    try:
        await service.start()
    finally:
        await service.stop()


def main() -> None:
    asyncio.run(async_main())
