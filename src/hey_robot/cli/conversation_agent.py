from __future__ import annotations

import argparse
import asyncio

from hey_robot.app.conversation import build_conversation_agent
from hey_robot.config import DeploymentConfig


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Hey Robot conversation agent")
    parser.add_argument("--config", required=True, help="Deployment YAML path")
    parser.add_argument("--agent-id", default=None)
    args = parser.parse_args()
    config = DeploymentConfig.from_yaml(args.config)
    service = build_conversation_agent(
        config, agent_id=args.agent_id or config.default_agent_id()
    )
    try:
        await service.start()
    finally:
        await service.stop()


def main() -> None:
    asyncio.run(async_main())
