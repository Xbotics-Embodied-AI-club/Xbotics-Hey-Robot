from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hey_robot.capability.vla.io_adapter import VLAIOAdapter

from hey_robot.capability.runtime.mock import MockCapabilityClient
from hey_robot.capability.runtime.models import CapabilityClient
from hey_robot.config import CapabilityServiceSpec, DeploymentConfig


class CapabilityRuntime:
    def __init__(
        self,
        config: DeploymentConfig,
        *,
        io_adapter: VLAIOAdapter | None = None,
    ) -> None:
        self.config = config
        self._io_adapter = io_adapter
        self.clients: dict[str, CapabilityClient] = {
            service_id: self._build_client(service_id, spec)
            for service_id, spec in config.capability_services.items()
            if spec.enabled
        }

    def set_vla_io_adapter(self, io: VLAIOAdapter) -> None:
        """Inject a VLA I/O adapter into all in-process VLA capability clients."""
        self._io_adapter = io
        for client in self.clients.values():
            setter = getattr(client, "set_io", None)
            if callable(setter):
                setter(io)

    def service_for(
        self,
        skill_name: str,
        robot_id: str | None,
    ) -> tuple[str, CapabilityServiceSpec, CapabilityClient] | None:
        for service_id, spec in self.config.capability_services.items():
            if not spec.enabled:
                continue
            if robot_id and spec.robot_id and spec.robot_id != robot_id:
                continue
            if skill_name in spec.skill_names:
                client = self.clients.get(service_id)
                if client is not None:
                    return service_id, spec, client
        return None

    def _build_client(
        self, service_id: str, spec: CapabilityServiceSpec
    ) -> CapabilityClient:
        if spec.type in {"mock", "mock_vla_service"}:
            return MockCapabilityClient(service_id, spec)
        if spec.type == "vla_service":
            from hey_robot.capability.vla.capability_client import (
                VLACapabilityClient,
            )

            return VLACapabilityClient(service_id, spec, io=self._io_adapter)
        from hey_robot.capability.transport.grpc.client import GrpcCapabilityClient

        return GrpcCapabilityClient(service_id, spec)
