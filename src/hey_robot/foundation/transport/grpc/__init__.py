"""gRPC model-service transport with lazy exports."""

from typing import Any

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
