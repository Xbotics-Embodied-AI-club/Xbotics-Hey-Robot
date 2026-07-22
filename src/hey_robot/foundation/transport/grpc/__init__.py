"""gRPC model-service transport with lazy exports."""

from typing import Any

__all__ = [
    "DEFAULT_ARM_CALIBRATION_DIR",
    "LeRobotVLAExecutor",
    "LeRobotVLAPolicyExecutor",
    "ModelServiceServicer",
    "ModelServiceState",
    "VLAPolicyService",
    "VLNPlannerService",
    "build_model_service",
]


def __getattr__(name: str) -> Any:
    if name in {
        "DEFAULT_ARM_CALIBRATION_DIR",
        "LeRobotVLAExecutor",
        "LeRobotVLAPolicyExecutor",
    }:
        from hey_robot.foundation.backends.vla.lerobot import executor

        return getattr(executor, name)
    if name in {
        "ModelServiceServicer",
        "ModelServiceState",
        "VLAPolicyService",
        "VLNPlannerService",
        "build_model_service",
    }:
        from hey_robot.foundation.transport.grpc import server

        return getattr(server, name)
    raise AttributeError(name)
