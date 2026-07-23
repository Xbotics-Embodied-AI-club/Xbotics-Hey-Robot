from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import grpc
from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.struct_pb2 import Struct

from evaluation.habitat3.worker.environment import (
    HabitatEnvironment,
    RuntimeObservation,
    RuntimeSkillResult,
)
from hey_robot.habitat_runtime.v1 import (
    habitat_runtime_pb2 as pb,
    habitat_runtime_pb2_grpc as pb_grpc,
)


class HabitatRuntimeService(pb_grpc.HabitatRuntimeServicer):
    """gRPC facade with exactly one Habitat/OpenGL owner thread."""

    def __init__(self, backend: HabitatEnvironment) -> None:
        self.backend = backend
        self._owner = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="habitat-owner"
        )
        self._busy = False
        self._load_error: str | None = None

    async def GetHealth(self, request, context):
        del request, context
        return pb.HealthResponse(
            online=True,
            # "loaded" means the runtime process can accept CreateEpisode;
            # an Env is intentionally created lazily for the selected profile.
            loaded=self._load_error is None,
            busy=self._busy,
            error_message=self._load_error or "",
            metrics=_struct({"profile": self.backend.profile, "owner_thread": True}),
        )

    async def CreateEpisode(self, request, context):
        try:
            if request.profile and request.profile != self.backend.profile:
                raise ValueError(
                    f"runtime serves profile {self.backend.profile}, requested {request.profile}"
                )
            observation = await self._call(
                self.backend.load,
                task=request.task or "RearrangePddlSocialNavTask-v0",
                split=request.split or "val",
                seed=request.seed,
                controlled_agent=request.controlled_agent or "agent_0",
                requested_dataset_episode_id=request.requested_dataset_episode_id
                or None,
            )
            self._load_error = None
            return pb.EpisodeResponse(observation=_observation(observation))
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, self._load_error)

    async def Observe(self, request, context):
        try:
            self._episode_matches(request.episode_id)
            return _observation(await self._call(self.backend.observe))
        except Exception as exc:
            await _abort(context, exc)

    async def Step(self, request, context):
        try:
            self._episode_matches(request.episode_id)
            observation = await self._call(
                self.backend.step,
                _dict(request.action),
                expected_frame_id=request.expected_frame_id,
            )
            return pb.StepResponse(
                observation=_observation(observation),
                done=observation.done,
                success=observation.success,
                metrics=_struct(observation.metrics),
            )
        except Exception as exc:
            await _abort(context, exc)

    async def ExecuteSkill(self, request, context):
        try:
            self._episode_matches(request.episode_id)
            result = await self._call(
                self.backend.execute_skill,
                operation_id=request.operation_id,
                skill_name=request.skill_name,
                arguments=_dict(request.arguments),
                expected_frame_id=request.expected_frame_id,
                max_steps=request.max_steps,
            )
            return _skill_result(result)
        except Exception as exc:
            await _abort(context, exc)

    async def CancelSkill(self, request, context):
        del context
        if request.episode_id != self.backend.episode_id:
            return pb.CancelSkillResponse(accepted=False, already_finished=True)
        # Deliberately bypass the command queue: a queued cancel cannot stop a
        # bounded executor that currently owns the sole Habitat thread.
        accepted = self.backend.cancel(request.operation_id)
        return pb.CancelSkillResponse(accepted=accepted, already_finished=not accepted)

    async def Reset(self, request, context):
        try:
            self._episode_matches(request.episode_id)
            return _observation(
                await self._call(
                    self.backend.reset, expected_frame_id=request.expected_frame_id
                )
            )
        except Exception as exc:
            await _abort(context, exc)

    async def CloseEpisode(self, request, context):
        try:
            self._episode_matches(request.episode_id)
            await self._call(
                self.backend.close, expected_frame_id=request.expected_frame_id
            )
            return pb.CloseEpisodeResponse(closed=True)
        except Exception as exc:
            await _abort(context, exc)

    async def close(self) -> None:
        self._owner.shutdown(wait=True, cancel_futures=True)

    async def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self._busy = True
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._owner, lambda: fn(*args, **kwargs))
        finally:
            self._busy = False

    def _episode_matches(self, episode_id: str) -> None:
        if not episode_id or episode_id != self.backend.episode_id:
            raise ValueError("episode does not exist or is not active")


def _observation(value: RuntimeObservation):
    return pb.ObservationResponse(
        episode_id=value.episode_id,
        frame_id=value.frame_id,
        assets=[
            pb.ObservationAsset(
                kind=item.kind,
                role=item.role or "",
                name=item.name or "",
                data=item.data,
                content_type=item.content_type or "application/octet-stream",
                width=item.width or 0,
                height=item.height or 0,
                metadata=_struct(item.metadata),
            )
            for item in value.assets
        ],
        proprioception=value.proprioception,
        task=value.task or "",
        done=value.done,
        success=value.success,
        metrics=_struct(value.metrics),
        entities=[_struct(item) for item in value.entities],
        metadata=_struct(value.metadata),
    )


def _skill_result(value: RuntimeSkillResult):
    return pb.SkillResponse(
        observation=_observation(value.observation),
        steps=value.steps,
        success=value.success,
        failure_mode=value.failure_mode or "",
        metrics=_struct(value.metrics),
        trace=[_struct(item) for item in value.trace],
        cancelled=value.cancelled,
    )


def _struct(value: dict[str, Any]) -> Struct:
    result = Struct()
    ParseDict(_json_value(value), result)
    return result


def _dict(value: Struct) -> dict[str, Any]:
    return MessageToDict(value, preserving_proto_field_name=True) if value else {}


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return _json_value(value.item())
    return value


async def _abort(context, exc: Exception):
    status = (
        grpc.StatusCode.INVALID_ARGUMENT
        if isinstance(exc, ValueError)
        else grpc.StatusCode.FAILED_PRECONDITION
    )
    await context.abort(status, f"{type(exc).__name__}: {exc}")


async def serve(*, host: str, port: int, data_root: str, profile: str) -> None:
    service = HabitatRuntimeService(
        HabitatEnvironment(data_root=data_root, profile=profile)
    )
    server = grpc.aio.server()
    pb_grpc.add_HabitatRuntimeServicer_to_server(service, server)
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    try:
        await server.wait_for_termination()
    finally:
        await service.close()
        await server.stop(grace=5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9093)
    parser.add_argument("--data-root", default="/opt/habitat/data")
    parser.add_argument("--profile", default="habitat3_social_spot_human_oracle")
    args = parser.parse_args()
    asyncio.run(
        serve(
            host=args.host,
            port=args.port,
            data_root=args.data_root,
            profile=args.profile,
        )
    )


if __name__ == "__main__":
    main()
