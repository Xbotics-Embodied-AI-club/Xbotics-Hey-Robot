from __future__ import annotations

import argparse
import asyncio

from hey_robot.agents import RobotAgentService
from hey_robot.config import DeploymentConfig


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Hey Robot agent service")
    parser.add_argument("--config", required=True, help="Deployment YAML path")
    parser.add_argument(
        "--agent-id", default=None, help="Agent id from deployment config"
    )
    parser.add_argument("--episode-dir", default=None, help="Episode store directory")
    args = parser.parse_args()

    config = DeploymentConfig.from_yaml(args.config)
    agent_id = args.agent_id or config.default_agent_id()
    service = RobotAgentService(config, agent_id=agent_id, episode_dir=args.episode_dir)
    try:
        await service.start()
    finally:
        await service.stop()


def main() -> None:
    asyncio.run(async_main())
