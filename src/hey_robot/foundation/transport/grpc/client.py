from __future__ import annotations

import os
from typing import Any

import grpc
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct

from hey_robot.config import ModelServiceSpec
from hey_robot.foundation.clients.models import (
    ServiceHealth,
    ServiceInvocationRequest,
    ServiceInvocationResult,
)
from hey_robot.foundation.contract.v1 import model_service_pb2, model_service_pb2_grpc


class GrpcModelServiceClient:
    def __init__(
        self,
        service_id: str,
        spec: ModelServiceSpec,
        *,
        auth_token: str | None = None,
    ) -> None:
        if not spec.target:
            raise ValueError(f"model service {service_id} missing gRPC target")
        self.service_id = service_id
        self.spec = spec
        self.target = str(spec.target).strip()
        # gRPC target 不接受 grpc:// scheme。
        self.target = self.target.removeprefix("grpc://")
        token_env = str(spec.settings.get("auth_token_env") or "").strip()
        self._token = auth_token or (os.environ.get(token_env) if token_env else None)
        self._channel = grpc.aio.insecure_channel(self.target)
        self._stub = model_service_pb2_grpc.ModelServiceStub(self._channel)

    async def health(self) -> ServiceHealth:
        timeout = float(self.spec.settings.get("health_timeout_sec", 2.0))
        try:
            response = await self._stub.GetHealth(
                model_service_pb2.GetHealthRequest(service_id=self.service_id),
                timeout=timeout,
                metadata=self._metadata(),
            )
        except grpc.aio.AioRpcError as exc:
            return ServiceHealth(
                name=self.service_id,
                online=False,
                loaded=False,
                busy=False,
                robot_id=self.spec.robot_id,
                error=f"{exc.code().name}: {exc.details()}",
                error_code=exc.code().name,
            )
        return ServiceHealth(
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
        self, request: ServiceInvocationRequest
    ) -> ServiceInvocationResult:
        arguments = (
            request.arguments
            if request.arguments is not None
            else request.intent.arguments
        )
        payload = model_service_pb2.ExecuteSkillRequest(
            service_id=request.service_id,
            trace_id=request.intent.envelope.trace_id,
            episode_id=request.intent.envelope.episode_id or "",
            skill_id=request.intent.skill_id,
            skill_name=request.intent.name or request.contract.name,
            robot_id=request.intent.envelope.robot_id or self.spec.robot_id,
            objective=request.intent.objective,
            arguments=_dict_to_struct(dict(arguments)),
            timeout_sec=float(request.timeout_sec),
            metadata=_dict_to_struct({}),
        )
        try:
            response = await self._stub.ExecuteSkill(
                payload,
                timeout=request.timeout_sec + 5.0,
                metadata=self._metadata(),
            )
        except grpc.aio.AioRpcError as exc:
            return ServiceInvocationResult(
                success=False,
                status="failed",
                summary=exc.details() or "model service invocation failed",
                failure_mode="model_service_unavailable",
                error=exc.details() or None,
                error_code=exc.code().name,
            )
        return ServiceInvocationResult(
            success=bool(response.success),
            status=response.status or ("completed" if response.success else "failed"),
            summary=response.summary or ("completed" if response.success else "failed"),
            failure_mode=response.failure_mode or None,
            error=response.error_message or None,
            metrics=_struct_to_dict(response.metrics),
            error_code=response.error_code or None,
        )

    async def cancel(self, skill_id: str) -> None:
        await self._stub.CancelSkill(
            model_service_pb2.CancelSkillRequest(
                service_id=self.service_id, skill_id=skill_id
            ),
            timeout=2.0,
            metadata=self._metadata(),
        )

    async def close(self) -> None:
        await self._channel.close()

    def _metadata(self) -> tuple[tuple[str, str], ...]:
        return (("authorization", f"Bearer {self._token}"),) if self._token else ()


def _dict_to_struct(value: dict[str, Any]) -> Struct:
    message = Struct()
    message.update(value)
    return message


def _struct_to_dict(value: Struct) -> dict[str, Any]:
    """递归地将 protobuf Struct 转换为普通 Python 值。"""
    if value is None:
        return {}
    return MessageToDict(value, preserving_proto_field_name=True)  # type: ignore[no-any-return]
