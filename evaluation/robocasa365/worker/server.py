from __future__ import annotations

import argparse
import asyncio
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

try:
    from rollout import (  # type: ignore[import-not-found]
        RoboCasaRolloutRunner,
        RolloutError,
        RolloutResult,
        request_from_payload,
    )
except ModuleNotFoundError:
    from evaluation.robocasa365.worker.rollout import (
        RoboCasaRolloutRunner,
        RolloutError,
        RolloutResult,
        request_from_payload,
    )

try:
    from option_runner import (  # type: ignore[import-not-found]
        OptionError,
        OptionResult,
        RoboCasaOptionRunner,
        request_from_payload as option_request_from_payload,
    )
except ModuleNotFoundError:
    from evaluation.robocasa365.worker.option_runner import (
        OptionError,
        OptionResult,
        RoboCasaOptionRunner,
        request_from_payload as option_request_from_payload,
    )

try:
    from runtime_server import RoboCasaRuntimeService
except ModuleNotFoundError:
    from evaluation.robocasa365.worker.runtime_server import RoboCasaRuntimeService

try:
    from hey_robot.robocasa_runtime.v1 import robocasa_runtime_pb2_grpc
except ModuleNotFoundError:
    robocasa_runtime_pb2_grpc = None


class RoboCasaModelService(model_service_pb2_grpc.ModelServiceServicer):
    """Standalone ModelService v1 implementation for the RoboCasa worker."""

    def __init__(
        self,
        runner: RoboCasaRolloutRunner | None = None,
        *,
        option_runner: RoboCasaOptionRunner | None = None,
        resource_lock: asyncio.Lock | None = None,
    ) -> None:
        self.runner = runner or RoboCasaRolloutRunner()
        self.option_runner = option_runner or RoboCasaOptionRunner()
        self._resource_lock = resource_lock or asyncio.Lock()
        self._option_holds_resource = False
        self._last_result: dict[str, Any] | None = None
        self._last_error: str | None = None

    async def GetHealth(self, request, context):  # noqa: N802
        del request, context
        health = self.runner.health()
        option_health = self.option_runner.health()
        metrics = dict(health.get("metrics", {}) or {})
        metrics["option_runner"] = dict(option_health.get("metrics", {}) or {})
        metrics["last_result"] = self._last_result
        return model_service_pb2.GetHealthResponse(
            service_id="robocasa365",
            name="robocasa365",
            robot_id="robocasa365",
            online=bool(health.get("online", True)),
            loaded=bool(health.get("loaded", False))
            or bool(option_health.get("loaded", False)),
            busy=self.runner.busy or self._resource_lock.locked(),
            current_skill_id=self.runner.current_skill_id or "",
            error_message=str(self._last_error or health.get("error") or ""),
            metrics=_dict_to_struct(metrics),
            version="robocasa365-bridge-v1",
        )

    async def ExecuteSkill(self, request, context):  # noqa: N802
        del context
        if request.skill_name == "robocasa_option":
            return await self._execute_option(request)
        if request.skill_name != "robocasa_rollout":
            return _failure_response(
                "invalid_task",
                "UNSUPPORTED_SKILL",
                f"worker only supports robocasa_rollout/robocasa_option, got {request.skill_name!r}",
            )
        if self.runner.busy or self._resource_lock.locked():
            return _failure_response(
                "model_service_busy",
                "MODEL_SERVICE_BUSY",
                "RoboCasa worker is busy",
            )

        payload = {
            "skill_id": request.skill_id,
            "objective": request.objective,
            "arguments": _struct_to_dict(request.arguments),
            "timeout_sec": request.timeout_sec,
        }
        try:
            rollout_request = request_from_payload(payload)
            async with self._resource_lock:
                result = await asyncio.to_thread(self.runner.run, rollout_request)
        except RolloutError as exc:
            result = RolloutResult(
                success=False,
                status="failed",
                summary=f"RoboCasa request rejected: {exc}",
                failure_mode=exc.failure_mode,
                error=str(exc),
            )
        except Exception as exc:
            result = RolloutResult(
                success=False,
                status="failed",
                summary=f"RoboCasa worker failed: {type(exc).__name__}: {exc}",
                failure_mode="execution_failed",
                error=str(exc),
            )
        payload_result = result.to_dict()
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

    async def _execute_option(self, request) -> model_service_pb2.ExecuteSkillResponse:
        if self.runner.busy or (
            self._resource_lock.locked() and not self._option_holds_resource
        ):
            return _failure_response(
                "model_service_busy",
                "MODEL_SERVICE_BUSY",
                "RoboCasa worker is busy",
            )
        payload = {
            "skill_id": request.skill_id,
            "objective": request.objective,
            "arguments": _struct_to_dict(request.arguments),
            "timeout_sec": request.timeout_sec,
        }
        try:
            option_request = option_request_from_payload(payload)
            await self._acquire_option_resource()
            result = await asyncio.to_thread(self.option_runner.run, option_request)
            if (
                option_request.close_episode
                or self.option_runner.active_sessions == 0
                or bool((result.metrics or {}).get("episode_done", False))
            ):
                self._release_option_resource()
        except OptionError as exc:
            result = OptionResult(
                success=False,
                status="failed",
                summary=f"RoboCasa option request rejected: {exc}",
                failure_mode=exc.failure_mode,
                error=str(exc),
            )
            self._release_option_resource_if_idle()
        except Exception as exc:
            result = OptionResult(
                success=False,
                status="failed",
                summary=f"RoboCasa option worker failed: {type(exc).__name__}: {exc}",
                failure_mode="execution_failed",
                error=str(exc),
            )
            self._release_option_resource_if_idle()
        payload_result = result.to_dict()
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

    async def _acquire_option_resource(self) -> None:
        if self._option_holds_resource:
            return
        await self._resource_lock.acquire()
        self._option_holds_resource = True

    def _release_option_resource(self) -> None:
        if self._option_holds_resource:
            self._option_holds_resource = False
            self._resource_lock.release()

    def _release_option_resource_if_idle(self) -> None:
        if self.option_runner.active_sessions == 0:
            self._release_option_resource()

    async def CancelSkill(self, request, context):  # noqa: N802
        del context
        accepted = await asyncio.to_thread(self.runner.cancel, request.skill_id or None)
        if not accepted and request.skill_id:
            accepted = await asyncio.to_thread(
                self.option_runner.close_session, request.skill_id
            )
            self._release_option_resource_if_idle()
        return model_service_pb2.CancelSkillResponse(
            accepted=accepted,
            summary="cancel requested" if accepted else "no matching rollout is active",
            error_code="" if accepted else "SKILL_NOT_ACTIVE",
        )


async def serve(host: str, port: int) -> None:
    server = grpc.aio.server()
    resource_lock = asyncio.Lock()
    model_service_pb2_grpc.add_ModelServiceServicer_to_server(
        RoboCasaModelService(resource_lock=resource_lock), server
    )
    if robocasa_runtime_pb2_grpc is not None:
        robocasa_runtime_pb2_grpc.add_RoboCasaRuntimeServicer_to_server(
            RoboCasaRuntimeService(resource_lock=resource_lock), server
        )
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    await server.wait_for_termination()


def main() -> None:
    parser = argparse.ArgumentParser(description="RoboCasa365 ModelService v1 worker")
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104
    parser.add_argument("--port", type=int, default=9092)
    args = parser.parse_args()
    asyncio.run(serve(args.host, args.port))


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
