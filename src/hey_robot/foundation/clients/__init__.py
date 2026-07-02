"""Foundation service clients."""

from hey_robot.foundation.clients.manager import ModelServiceRegistry
from hey_robot.foundation.clients.mock import MockModelServiceClient
from hey_robot.foundation.clients.models import (
    ModelServiceClient,
    ServiceHealth,
    ServiceInvocationRequest,
    ServiceInvocationResult,
)

__all__ = [
    "MockModelServiceClient",
    "ModelServiceClient",
    "ModelServiceRegistry",
    "ServiceHealth",
    "ServiceInvocationRequest",
    "ServiceInvocationResult",
]
