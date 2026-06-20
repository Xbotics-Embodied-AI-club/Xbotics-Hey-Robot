from __future__ import annotations

import argparse
import asyncio

from hey_robot.agents import TaskSupervisorService
from hey_robot.config import DeploymentConfig


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Hey Robot task supervisor service")
    parser.add_argument("--config", required=True, help="Deployment YAML path")
    parser.add_argument("--episode-dir", default=None, help="Episode store directory")
    args = parser.parse_args()

    config = DeploymentConfig.from_yaml(args.config)
    service = TaskSupervisorService(config, episode_dir=args.episode_dir)
    try:
        await service.start()
    finally:
        await service.stop()


def main() -> None:
    asyncio.run(async_main())
