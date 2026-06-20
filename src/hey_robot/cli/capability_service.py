from __future__ import annotations

import argparse
import asyncio

from hey_robot.capability.transport.grpc import VLACapabilityService
from hey_robot.config import DeploymentConfig


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Hey Robot gRPC capability service")
    parser.add_argument("--config", required=True, help="Deployment YAML path")
    parser.add_argument(
        "--service-id", required=True, help="capability_services entry to run"
    )
    parser.add_argument("--host", default=None, help="gRPC bind host")
    parser.add_argument("--port", type=int, default=None, help="gRPC bind port")
    args = parser.parse_args()

    config = DeploymentConfig.from_yaml(args.config)
    spec = config.capability_services.get(args.service_id)
    if spec is None:
        raise SystemExit(f"unknown capability service: {args.service_id}")
    if spec.type != "vla_service":
        raise SystemExit(
            f"unsupported capability service type for standalone runner: {spec.type}"
        )

    service = VLACapabilityService(
        config, service_id=args.service_id, host=args.host, port=args.port
    )
    try:
        await service.start()
    finally:
        await service.stop()


def main() -> None:
    asyncio.run(async_main())
