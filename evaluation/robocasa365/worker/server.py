from __future__ import annotations

import argparse
import asyncio
import os
from contextlib import suppress
from typing import Any

import grpc
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct

try:  # Generated from proto/ inside the standalone container image.
    from hey_robot.model_service.v1 import model_service_pb2, model_service_pb2_grpc
except ModuleNotFoundError:  # Makes local protocol tests possible without the image.
    from hey_robot.foundation.contract.v1 import (  # type: ignore[no-redef]
        model_service_pb2,
        model_service_pb2_grpc,
    )

_runtime_grpc_module: Any = None
try:
    from episode_manager import EpisodeManager
except ModuleNotFoundError:
    from evaluation.robocasa365.worker.episode_manager import EpisodeManager

try:
    from vla_option_executor import (  # type: ignore[import-not-found]
        OptionExecutionError,
        OptionResult,
        VLAOptionExecutor,
        request_from_payload as option_request_from_payload,
    )
except ModuleNotFoundError:
    from evaluation.robocasa365.worker.vla_option_executor import (
        OptionExecutionError,
        OptionResult,
        VLAOptionExecutor,
        request_from_payload as option_request_from_payload,
    )

try:
    from runtime_server import RoboCasaRuntimeService
except ModuleNotFoundError:
    from evaluation.robocasa365.worker.runtime_server import RoboCasaRuntimeService

with suppress(ModuleNotFoundError):
    from hey_robot.robocasa_runtime.v1 import (  # type: ignore[no-redef]
        robocasa_runtime_pb2_grpc as _runtime_grpc_module,
    )


class RoboCasaModelService(model_service_pb2_grpc.ModelServiceServicer):
    """Standalone ModelService v1 implementation for the RoboCasa worker."""

    def __init__(
        self,
        *,
        executor: VLAOptionExecutor | None = None,
        episode_manager: EpisodeManager | None = None,
    ) -> None:
        if episode_manager is None:
            try:
                from contract import ALLOWED_TASKS
            except ModuleNotFoundError:
                from evaluation.robocasa365.worker.contract import ALLOWED_TASKS
            episode_manager = EpisodeManager(allowed_tasks=ALLOWED_TASKS)
        self.episode_manager = episode_manager
        self.executor = executor or VLAOptionExecutor(manager=self.episode_manager)
        self._last_result: dict[str, Any] | None = None
        self._last_error: str | None = None

    async def GetHealth(self, request, context):  # noqa: N802
        del request, context
        option_health = self.executor.health()
        metrics = dict(option_health.get("metrics", {}) or {})
        metrics["last_result"] = self._last_result
        return model_service_pb2.GetHealthResponse(
            service_id="robocasa365",
            name="robocasa365",
            robot_id="robocasa365",
            online=bool(option_health.get("online", True)),
            loaded=bool(option_health.get("loaded", False)),
            busy=False,
            current_skill_id="",
            error_message=str(self._last_error or option_health.get("error") or ""),
            metrics=_dict_to_struct(metrics),
            version="robocasa365-bridge-v1",
        )

    async def ExecuteSkill(self, request, context):  # noqa: N802
        del context
        if request.skill_name == "robocasa_option":
            return await self._execute_option(request)
        return _failure_response(
            "invalid_task",
            "UNSUPPORTED_SKILL",
            f"worker only supports robocasa_option, got {request.skill_name!r}",
        )

    async def _execute_option(self, request) -> model_service_pb2.ExecuteSkillResponse:
        if not self.episode_manager.active:
            return _failure_response(
                "trial_unavailable",
                "TRIAL_UNAVAILABLE",
                "begin a RoboCasa trial before executing an option",
            )
        payload = {
            "skill_id": request.skill_id,
            "objective": request.objective,
            "arguments": _struct_to_dict(request.arguments),
            "timeout_sec": request.timeout_sec,
        }
        try:
            option_request = option_request_from_payload(payload)
            result = await asyncio.to_thread(self.executor.run, option_request)
        except OptionExecutionError as exc:
            result = OptionResult(
                success=False,
                status="failed",
                summary=f"RoboCasa option request rejected: {exc}",
                failure_mode=exc.failure_mode,
                error=str(exc),
            )
        except Exception as exc:
            result = OptionResult(
                success=False,
                status="failed",
                summary=f"RoboCasa option worker failed: {type(exc).__name__}: {exc}",
                failure_mode="execution_failed",
                error=str(exc),
            )
        payload_result = result.to_dict()
        self.episode_manager.record_event(
            "model_service_option",
            {
                "skill_id": request.skill_id,
                "success": result.success,
                "status": result.status,
                "failure_mode": result.failure_mode,
                "metrics": dict(result.metrics or {}),
            },
        )
        self._last_result = payload_result
        self._last_error = result.error if not result.success else None
        return model_service_pb2.ExecuteSkillResponse(
            success=result.success,
            status=result.status,
            summary=result.summary,
            failure_mode=result.failure_mode or "",
            error_code=(result.failure_mode or "").upper(),
            error_message=result.error or "",
            metrics=_dict_to_struct(dict(result.metrics or {})),
        )

    async def CancelSkill(self, request, context):  # noqa: N802
        del context
        accepted = self.executor.cancel(request.skill_id)
        return model_service_pb2.CancelSkillResponse(
            accepted=accepted,
            summary="option cancellation accepted"
            if accepted
            else "no matching option is active",
            error_code="" if accepted else "SKILL_NOT_ACTIVE",
        )


async def serve(
    host: str,
    port: int,
    *,
    evaluator_token: str | None = None,
    data_token: str | None = None,
) -> None:
    server = grpc.aio.server()
    resource_lock = asyncio.Lock()
    from evaluation.robocasa365.worker.contract import ALLOWED_TASKS

    episode_manager = EpisodeManager(allowed_tasks=ALLOWED_TASKS)
    model_service = RoboCasaModelService(episode_manager=episode_manager)
    model_service_pb2_grpc.add_ModelServiceServicer_to_server(
        model_service,
        server,
    )
    if _runtime_grpc_module is not None:
        _runtime_grpc_module.add_RoboCasaRuntimeServicer_to_server(
            RoboCasaRuntimeService(
                resource_lock=resource_lock,
                manager=episode_manager,
                evaluator_token=evaluator_token,
                data_token=data_token,
                prepare_trial=model_service.executor.prepare,
            ),
            server,
        )
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    await server.wait_for_termination()


def main() -> None:
    parser = argparse.ArgumentParser(description="RoboCasa365 ModelService v1 worker")
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104
    parser.add_argument("--port", type=int, default=9092)
    parser.add_argument(
        "--evaluator-token", default=os.environ.get("ROBOCASA_EVALUATOR_TOKEN")
    )
    parser.add_argument("--data-token", default=os.environ.get("ROBOCASA_DATA_TOKEN"))
    parser.add_argument("--insecure-local", action="store_true")
    args = parser.parse_args()
    if bool(args.evaluator_token) != bool(args.data_token):
        parser.error("set both evaluator and data tokens")
    if not args.insecure_local and not args.evaluator_token:
        parser.error("set evaluator/data tokens or explicitly pass --insecure-local")
    asyncio.run(
        serve(
            args.host,
            args.port,
            evaluator_token=args.evaluator_token or None,
            data_token=args.data_token or None,
        )
    )


def _dict_to_struct(value: dict[str, Any]) -> Struct:
    message = Struct()
    message.update(value)
    return message


def _struct_to_dict(value: Struct) -> dict[str, Any]:
    if value is None:
        return {}
    return MessageToDict(value, preserving_proto_field_name=True)  # type: ignore[no-any-return]


def _failure_response(
    failure_mode: str, error_code: str, message: str
) -> model_service_pb2.ExecuteSkillResponse:
    return model_service_pb2.ExecuteSkillResponse(
        success=False,
        status="failed",
        summary=message,
        failure_mode=failure_mode,
        error_code=error_code,
        error_message=message,
    )


if __name__ == "__main__":
    main()
