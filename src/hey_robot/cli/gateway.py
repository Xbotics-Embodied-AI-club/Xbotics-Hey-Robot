from __future__ import annotations

import argparse
import asyncio

from hey_robot.config import DeploymentConfig
from hey_robot.gateway import GatewayService


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Hey Robot interaction gateway")
    parser.add_argument("--config", required=True, help="Deployment YAML path")
    parser.add_argument("--episode-dir", default=None, help="Episode store directory")
    args = parser.parse_args()

    config = DeploymentConfig.from_yaml(args.config)
    service = GatewayService(config, episode_dir=args.episode_dir)
    try:
        await service.start()
    finally:
        await service.stop()


def main() -> None:
    asyncio.run(async_main())
