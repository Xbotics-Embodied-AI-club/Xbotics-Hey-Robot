from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress

from hey_robot.config import DeploymentConfig
from hey_robot.foundation.transport.grpc import build_model_service


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Hey Robot gRPC model service")
    parser.add_argument("--config", required=True, help="Deployment YAML path")
    parser.add_argument(
        "--service-id", required=True, help="model_services entry to run"
    )
    parser.add_argument("--host", default=None, help="gRPC bind host")
    parser.add_argument("--port", type=int, default=None, help="gRPC bind port")
    args = parser.parse_args()

    config = DeploymentConfig.from_yaml(args.config)
    spec = config.model_services.get(args.service_id)
    if spec is None:
        raise SystemExit(f"unknown model service: {args.service_id}")

    try:
        service = build_model_service(
            config, service_id=args.service_id, host=args.host, port=args.port
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        # asyncio.run cancels the main task on SIGINT. Treat that as an
        # operator-requested shutdown, not a model-service failure.
        with suppress(asyncio.CancelledError):
            await service.start()
    finally:
        with suppress(asyncio.CancelledError):
            await service.stop()


def main() -> None:
    with suppress(KeyboardInterrupt):
        asyncio.run(async_main())


if __name__ == "__main__":
    main()
