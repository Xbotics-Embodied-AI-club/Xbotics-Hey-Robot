from __future__ import annotations

from typing import Any

import grpc
from google.protobuf.struct_pb2 import Struct

from hey_robot.capability.contract.v1 import capability_pb2, capability_pb2_grpc
from hey_robot.capability.runtime.models import (
    CapabilityExecutionRequest,
    CapabilityExecutionResult,
    CapabilityHealth,
)
from hey_robot.config import CapabilityServiceSpec


class GrpcCapabilityClient:
    def __init__(self, service_id: str, spec: CapabilityServiceSpec) -> None:
        if not spec.target:
            raise ValueError(f"capability service {service_id} missing gRPC target")
        self.service_id = service_id
        self.spec = spec
        self.target = str(spec.target).strip()
        self._channel = grpc.aio.insecure_channel(self.target)
        self._stub = capability_pb2_grpc.CapabilityServiceStub(self._channel)

    async def health(self) -> CapabilityHealth:
        timeout = float(self.spec.settings.get("health_timeout_sec", 2.0))
        try:
            response = await self._stub.GetHealth(
                capability_pb2.GetHealthRequest(service_id=self.service_id),
                timeout=timeout,
            )
        except grpc.aio.AioRpcError as exc:
            return CapabilityHealth(
                name=self.service_id,
                online=False,
                loaded=False,
                busy=False,
                robot_id=self.spec.robot_id,
                error=f"{exc.code().name}: {exc.details()}",
                error_code=exc.code().name,
            )
        return CapabilityHealth(
            name=response.name or self.service_id,
            online=bool(response.online),
            loaded=bool(response.loaded),
            busy=bool(response.busy),
            robot_id=response.robot_id or self.spec.robot_id,
            error=response.error_message or None,
            metrics=_struct_to_dict(response.metrics),
            current_skill_id=response.current_skill_id or None,
            error_code=response.error_code or None,
            version=response.version or None,
        )

    async def execute(
        self, request: CapabilityExecutionRequest
    ) -> CapabilityExecutionResult:
        payload = capability_pb2.ExecuteCapabilityRequest(
            service_id=request.service_id,
            trace_id=request.intent.envelope.trace_id,
            episode_id=request.intent.envelope.episode_id or "",
            skill_id=request.intent.skill_id,
            skill_name=request.intent.name or request.contract.name,
            robot_id=request.intent.envelope.robot_id or self.spec.robot_id,
            objective=request.intent.objective,
            arguments=_dict_to_struct(dict(request.intent.arguments)),
            timeout_sec=float(request.timeout_sec),
            metadata=_dict_to_struct(dict(request.intent.metadata)),
        )
        try:
            response = await self._stub.ExecuteCapability(
                payload, timeout=request.timeout_sec + 5.0
            )
        except grpc.aio.AioRpcError as exc:
            return CapabilityExecutionResult(
                success=False,
                status="failed",
                summary=exc.details() or "capability execution failed",
                failure_mode="capability_unavailable",
                error=exc.details() or None,
                error_code=exc.code().name,
            )
        return CapabilityExecutionResult(
            success=bool(response.success),
            status=response.status or ("completed" if response.success else "failed"),
            summary=response.summary or ("completed" if response.success else "failed"),
            failure_mode=response.failure_mode or None,
            error=response.error_message or None,
            metrics=_struct_to_dict(response.metrics),
            error_code=response.error_code or None,
        )

    async def cancel(self, skill_id: str) -> None:
        await self._stub.CancelCapability(
            capability_pb2.CancelCapabilityRequest(
                service_id=self.service_id, skill_id=skill_id
            ),
            timeout=2.0,
        )


def _dict_to_struct(value: dict[str, Any]) -> Struct:
    message = Struct()
    message.update(value)
    return message


def _struct_to_dict(value: Struct) -> dict[str, Any]:
    return dict(value) if value is not None else {}
