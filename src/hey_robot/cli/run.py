from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from contextlib import suppress

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

    config = DeploymentConfig.from_yaml(args.config)
    runner = DeploymentRunner(
        config,
        episode_dir=args.episode_dir,
        config_path=args.config,
    )
    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    for handled_signal in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(handled_signal, shutdown.set)
    run_task = asyncio.create_task(runner.run())
    shutdown_task = asyncio.create_task(shutdown.wait())
    try:
        done, _ = await asyncio.wait(
            {run_task, shutdown_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if run_task in done:
            await run_task
    finally:
        shutdown_task.cancel()
        run_task.cancel()
        stop_task = asyncio.create_task(runner.stop())
        with suppress(asyncio.CancelledError):
            await asyncio.shield(stop_task)
        if not stop_task.done():
            with suppress(asyncio.CancelledError):
                await stop_task
        if not run_task.done():
            with suppress(asyncio.CancelledError):
                await run_task


def main() -> None:
    try:
        with asyncio.Runner() as runner:
            runner.run(async_main())
    except KeyboardInterrupt:
        logger.info("用户中断，部署已关闭。")
