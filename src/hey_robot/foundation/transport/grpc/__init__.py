"""gRPC model-service transport with lazy exports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hey_robot.foundation.backends.lerobot.executor import LeRobotPolicyExecutor
    from hey_robot.foundation.transport.grpc.server import (
        ModelServiceServicer,
        ModelServiceState,
        RobotPolicyService,
        VLNPlannerService,
        build_model_service,
    )

__all__ = [
    "LeRobotPolicyExecutor",
    "ModelServiceServicer",
    "ModelServiceState",
    "RobotPolicyService",
    "VLNPlannerService",
    "build_model_service",
]


def __getattr__(name: str) -> Any:
    if name == "LeRobotPolicyExecutor":
        from hey_robot.foundation.backends.lerobot import executor

        return getattr(executor, name)
    if name in {
        "ModelServiceServicer",
        "ModelServiceState",
        "RobotPolicyService",
        "VLNPlannerService",
        "build_model_service",
    }:
        from hey_robot.foundation.transport.grpc import server

        return getattr(server, name)
    raise AttributeError(name)
