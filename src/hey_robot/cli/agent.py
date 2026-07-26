from __future__ import annotations

import argparse
import asyncio

from hey_robot.cognition import AutonomousAgentService
from hey_robot.config import DeploymentConfig
from hey_robot.skills import BusSkillClient


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Hey Robot agent service")
    parser.add_argument("--config", required=True, help="Deployment YAML path")
    parser.add_argument(
        "--agent-id", default=None, help="Agent id from deployment config"
    )
    args = parser.parse_args()

    config = DeploymentConfig.from_yaml(args.config)
    agent_id = args.agent_id or config.default_agent_id()
    skills = BusSkillClient(config)
    service = AutonomousAgentService(
        config,
        agent_id=agent_id,
        skill_client=skills,
    )
    try:
        await service.start()
    finally:
        await service.stop()
        await skills.close()


def main() -> None:
    asyncio.run(async_main())
