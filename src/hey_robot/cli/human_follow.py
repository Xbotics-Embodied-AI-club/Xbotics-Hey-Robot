from __future__ import annotations

import argparse
import asyncio

from hey_robot.config import DeploymentConfig
from hey_robot.human_follow import HumanFollowService


async def _run(config_path: str) -> None:
    service = HumanFollowService(DeploymentConfig.from_yaml(config_path))
    try:
        await service.start()
    finally:
        await service.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hey Robot human follow service")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    asyncio.run(_run(args.config))


if __name__ == "__main__":
    main()
