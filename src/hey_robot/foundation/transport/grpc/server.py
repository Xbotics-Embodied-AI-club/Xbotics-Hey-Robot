from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import grpc
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct

from hey_robot.foundation.contract.v1 import model_service_pb2, model_service_pb2_grpc
from hey_robot.logging import HeyRobotLogger

if TYPE_CHECKING:
    from hey_robot.config import DeploymentConfig, ModelServiceSpec

logger = HeyRobotLogger(name="model_service")


@dataclass
class ModelServiceState:
    service_id: str
    spec: ModelServiceSpec
    busy: bool = False
    current_skill_id: str | None = None
    last_error: str | None = None
    last_result: dict[str, Any] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


class ModelServiceExecutor(Protocol):
    def health(self) -> dict[str, Any]: ...

    def execute(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    def cancel(self) -> None: ...


class ModelServiceServicer(model_service_pb2_grpc.ModelServiceServicer):
    def __init__(
        self,
        state: ModelServiceState,
        executor: ModelServiceExecutor,
        *,
        bearer_token: str | None = None,
    ) -> None:
        self.state = state
        self.executor = executor
        self.bearer_token = bearer_token

    async def GetHealth(self, request, context):
        del request
        await self._authorize(context)
        payload = self.executor.health()
        metrics = {
            **dict(payload.get("metrics", {}) or {}),
            **dict(self.state.metrics),
            "last_result": self.state.last_result,
        }
        return model_service_pb2.GetHealthResponse(
            service_id=self.state.service_id,
            name=str(payload.get("name") or self.state.service_id),
            robot_id=str(payload.get("robot_id") or self.state.spec.robot_id),
            online=bool(payload.get("online", True)),
            loaded=bool(payload.get("loaded", True)),
            busy=bool(self.state.busy),
            current_skill_id=self.state.current_skill_id or "",
            error_message=str(self.state.last_error or payload.get("error") or ""),
            metrics=_dict_to_struct(metrics),
            version="grpc-v1",
        )

    async def ExecuteSkill(self, request, context):
        await self._authorize(context)
        if self.state.busy:
            return model_service_pb2.ExecuteSkillResponse(
                success=False,
                status="failed",
                summary=f"model service {self.state.service_id} is busy",
                failure_mode="model_service_busy",
                error_code="MODEL_SERVICE_BUSY",
                error_message=f"model service {self.state.service_id} is busy",
            )
        self.state.busy = True
        self.state.current_skill_id = request.skill_id or None
        self.state.last_error = None
        payload = {
            "service_id": request.service_id,
            "trace_id": request.trace_id,
            "episode_id": request.episode_id,
            "skill_id": request.skill_id,
            "skill_name": request.skill_name,
            "robot_id": request.robot_id,
            "objective": request.objective,
            "arguments": _struct_to_dict(request.arguments),
            "timeout_sec": request.timeout_sec,
            "metadata": _struct_to_dict(request.metadata),
        }
        try:
            result = await asyncio.to_thread(self.executor.execute, payload)
        except Exception as exc:
            result = {
                "success": False,
                "status": "failed",
                "failure_mode": "execution_failed",
                "summary": f"model service invocation failed: {type(exc).__name__}: {exc}",
                "error": str(exc),
                "error_code": "EXECUTION_FAILED",
            }
        finally:
            self.state.busy = False
            self.state.current_skill_id = None
        self.state.last_result = result
        self.state.last_error = (
            (str(result.get("error") or "") or None)
            if not result.get("success")
            else None
        )
        return model_service_pb2.ExecuteSkillResponse(
            success=bool(result.get("success", False)),
            status=str(
                result.get("status")
                or ("completed" if result.get("success") else "failed")
            ),
            summary=str(result.get("summary") or ""),
            failure_mode=str(result.get("failure_mode") or ""),
            error_code=str(result.get("error_code") or ""),
            error_message=str(result.get("error") or ""),
            metrics=_dict_to_struct(dict(result.get("metrics", {}) or {})),
        )

    async def CancelSkill(self, request, context):
        del request
        await self._authorize(context)
        self.executor.cancel()
        return model_service_pb2.CancelSkillResponse(
            accepted=True, summary="cancel requested"
        )

    async def _authorize(self, context: Any) -> None:
        if not self.bearer_token:
            return
        metadata = dict(context.invocation_metadata())
        token = metadata.get("authorization", "").removeprefix("Bearer ")
        if token == self.bearer_token:
            return
        await context.abort(
            grpc.StatusCode.PERMISSION_DENIED,
            "ModelService data-plane credential is required",
        )


class VLAPolicyService:
    def __init__(
        self,
        config: DeploymentConfig,
        *,
        service_id: str,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        from hey_robot.foundation.backends.vla.lerobot import (
            LeRobotVLAExecutor,
            LeRobotVLAPolicyExecutor,
        )

        self.config = config
        self.service_id = service_id
        self.spec = config.model_services[service_id]
        self.host = host or str(self.spec.settings.get("host", "127.0.0.1"))
        self.port = port or int(self.spec.settings.get("port", 9090))
        self.state = ModelServiceState(service_id, self.spec)
        backend_mode = str(
            self.spec.settings.get("backend_mode")
            or self.spec.settings.get("mode")
            or self.spec.settings.get("backend")
            or "action_chunk_policy"
        )
        executor: ModelServiceExecutor
        if backend_mode in {"lerobot_control_loop", "legacy_lerobot_control_loop"}:
            executor = LeRobotVLAExecutor(service_id, self.spec)
        else:
            executor = LeRobotVLAPolicyExecutor(service_id, self.spec)
        self.executor = executor
        self._server: grpc.aio.Server | None = None

    async def start(self) -> None:
        self._server = grpc.aio.server()
        model_service_pb2_grpc.add_ModelServiceServicer_to_server(
            ModelServiceServicer(self.state, self.executor),
            self._server,
        )
        bind_target = f"{self.host}:{self.port}"
        self._server.add_insecure_port(bind_target)
        logger.info(
            f"VLA policy service [{self.service_id}] listening on grpc://{bind_target}"
        )
        await self._server.start()
        await self._server.wait_for_termination()

    async def stop(self) -> None:
        self.executor.cancel()
        if self._server is not None:
            await self._server.stop(grace=0.5)


class VLNPlannerService:
    def __init__(
        self,
        config: DeploymentConfig,
        *,
        service_id: str,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        from hey_robot.foundation.backends.vln.internvla_n1_system2 import (
            InternVLAN1System2Executor,
        )

        self.config = config
        self.service_id = service_id
        self.spec = config.model_services[service_id]
        self.host = host or str(self.spec.settings.get("host", "127.0.0.1"))
        self.port = port or int(self.spec.settings.get("port", 9091))
        self.state = ModelServiceState(service_id, self.spec)
        self.executor = InternVLAN1System2Executor(service_id, self.spec)
        self._server: grpc.aio.Server | None = None

    async def start(self) -> None:
        self._server = grpc.aio.server()
        model_service_pb2_grpc.add_ModelServiceServicer_to_server(
            ModelServiceServicer(self.state, self.executor),
            self._server,
        )
        bind_target = f"{self.host}:{self.port}"
        self._server.add_insecure_port(bind_target)
        logger.info(
            f"VLN planner service [{self.service_id}] listening on grpc://{bind_target}"
        )
        await self._server.start()
        await self._server.wait_for_termination()

    async def stop(self) -> None:
        self.executor.cancel()
        if self._server is not None:
            await self._server.stop(grace=0.5)


def build_model_service(
    config: DeploymentConfig,
    *,
    service_id: str,
    host: str | None = None,
    port: int | None = None,
) -> VLAPolicyService | VLNPlannerService:
    spec = config.model_services[service_id]
    if spec.type == "vla_policy":
        return VLAPolicyService(config, service_id=service_id, host=host, port=port)
    if spec.type == "vln_planner":
        return VLNPlannerService(config, service_id=service_id, host=host, port=port)
    raise ValueError(f"unsupported model service type: {spec.type}")


def _dict_to_struct(value: dict[str, Any]) -> Struct:
    message = Struct()
    message.update(value)
    return message


def _struct_to_dict(value: Struct) -> dict[str, Any]:
    """递归地将 protobuf Struct 转换为普通 Python 值。"""
    if value is None:
        return {}
    return MessageToDict(value, preserving_proto_field_name=True)  # type: ignore[no-any-return]
