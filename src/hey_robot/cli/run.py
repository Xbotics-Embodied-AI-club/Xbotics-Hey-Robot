from __future__ import annotations

import argparse
import asyncio
import logging

from hey_robot.app import DeploymentRunner
from hey_robot.config import DeploymentConfig

logger = logging.getLogger(__name__)


async def async_main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a full local Hey Robot deployment"
    )
    parser.add_argument("--config", required=True, help="Deployment YAML path")
    parser.add_argument("--episode-dir", default=None, help="Episode store directory")
    args = parser.parse_args()

    runner = DeploymentRunner(
        DeploymentConfig.from_yaml(args.config), episode_dir=args.episode_dir
    )
    try:
        await runner.run()
    except KeyboardInterrupt:
        logger.info("用户中断，正在关闭。")


def main() -> None:
    with asyncio.Runner() as runner:
        runner.run(async_main())
