from __future__ import annotations

from hey_robot.config import DeploymentConfig, ModelServiceSpec
from hey_robot.foundation.clients.mock import MockModelServiceClient
from hey_robot.foundation.clients.models import ModelServiceClient


class ModelServiceRegistry:
    def __init__(self, config: DeploymentConfig) -> None:
        self.config = config
        self.clients: dict[str, ModelServiceClient] = {
            service_id: self._build_client(service_id, spec)
            for service_id, spec in config.model_services.items()
            if spec.enabled
        }

    def service_for(
        self,
        skill_name: str,
        robot_id: str | None,
    ) -> tuple[str, ModelServiceSpec, ModelServiceClient] | None:
        for service_id, spec in self.config.model_services.items():
            if not spec.enabled:
                continue
            if robot_id and spec.robot_id and spec.robot_id != robot_id:
                continue
            if skill_name in spec.provides:
                client = self.clients.get(service_id)
                if client is not None:
                    return service_id, spec, client
        return None

    def _build_client(
        self, service_id: str, spec: ModelServiceSpec
    ) -> ModelServiceClient:
        if spec.type in {"mock", "mock_vla_policy"}:
            return MockModelServiceClient(service_id, spec)
        from hey_robot.foundation.transport.grpc.client import GrpcModelServiceClient

        return GrpcModelServiceClient(service_id, spec)
