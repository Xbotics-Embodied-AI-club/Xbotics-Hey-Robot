from __future__ import annotations

import os
from typing import Any

import grpc
from google.protobuf.json_format import MessageToDict

from hey_robot.robocasa_backend.rpc.v1 import (
    robocasa_runtime_pb2 as _robocasa_runtime_pb2,
    robocasa_runtime_pb2_grpc,
)
from hey_robot.robot_backends.robocasa_remote.protocol import (
    RemoteImage,
    RemoteObservation,
    RemoteStep,
)

# Protobuf message attributes are installed dynamically by generated code.
# Keep this untyped protocol boundary in the transport adapter only.
robocasa_runtime_pb2: Any = _robocasa_runtime_pb2


class GrpcRoboCasaRuntimeClient:
    """Async client for the isolated RoboCasa runtime container."""

    def __init__(
        self,
        target: str,
        *,
        timeout_sec: float = 10.0,
        role: str = "data",
        token: str | None = None,
    ) -> None:
        normalized = target.removeprefix("grpc://")
        if not normalized:
            raise ValueError("RoboCasa runtime target must not be empty")
        self.target = normalized
        self.timeout_sec = float(timeout_sec)
        self.role = role
        self.token = token or os.environ.get(
            "ROBOCASA_EVALUATOR_TOKEN" if role == "evaluator" else "ROBOCASA_DATA_TOKEN"
        )
        self._channel: grpc.aio.Channel | None = None
        # gRPC's generated stub does not publish a useful static interface.
        # Keep that untyped boundary local to the generated protocol adapter.
        self._stub: Any | None = None

    def _runtime_stub(self) -> Any:
        # RobotManager is constructed before its async lifecycle starts, while
        # grpc.aio requires a running event loop to allocate a channel.
        if self._stub is None:
            self._channel = grpc.aio.insecure_channel(self.target)
            self._stub = robocasa_runtime_pb2_grpc.RoboCasaRuntimeStub(self._channel)
        return self._stub

    async def health(self) -> dict[str, Any]:
        response = await self._runtime_stub().GetHealth(
            robocasa_runtime_pb2.HealthRequest(),
            timeout=self.timeout_sec,
            metadata=self._metadata(),
        )
        return {
            "online": response.online,
            "loaded": response.loaded,
            "busy": response.busy,
            "error": response.error_message or None,
            "metrics": _struct_to_dict(response.metrics),
        }

    async def begin_trial(
        self,
        *,
        trial_id: str,
        task: str,
        seed: int,
        split: str = "target",
        registries: tuple[str, ...] = ("lightwheel",),
    ) -> RemoteObservation:
        response = await self._runtime_stub().BeginTrial(
            robocasa_runtime_pb2.BeginTrialRequest(
                trial_id=trial_id,
                task=task,
                seed=seed,
                split=split,
                registries=registries,
            ),
            timeout=self.timeout_sec,
            metadata=self._metadata(),
        )
        return _observation(response)

    async def observe(self) -> RemoteObservation:
        response = await self._runtime_stub().Observe(
            robocasa_runtime_pb2.EmptyRequest(),
            timeout=self.timeout_sec,
            metadata=self._metadata(),
        )
        return _observation(response)

    async def step(
        self,
        *,
        action: list[float],
        expected_frame_id: int,
        raw_action: list[float] | None = None,
        action_clipped: bool = False,
    ) -> RemoteStep:
        response = await self._runtime_stub().Step(
            robocasa_runtime_pb2.StepRequest(
                action=action,
                expected_frame_id=expected_frame_id,
                raw_action=raw_action or action,
                action_clipped=action_clipped,
            ),
            timeout=self.timeout_sec,
            metadata=self._metadata(),
        )
        return RemoteStep(
            observation=_observation(response.observation),
            reward=float(response.reward),
            done=bool(response.done),
            metrics=_struct_to_dict(response.metrics),
        )

    async def read_truth(self) -> dict[str, Any]:
        response = await self._runtime_stub().ReadTruth(
            robocasa_runtime_pb2.EmptyRequest(),
            timeout=self.timeout_sec,
            metadata=self._metadata(),
        )
        return {
            "done": bool(response.done),
            "official_success": bool(response.official_success),
            "frame_id": int(response.frame_id),
            "metrics": _struct_to_dict(response.metrics),
        }

    async def end_trial(self, *, reason: str = "completed") -> bool:
        response = await self._runtime_stub().EndTrial(
            robocasa_runtime_pb2.EndTrialRequest(reason=reason),
            timeout=self.timeout_sec,
            metadata=self._metadata(),
        )
        return bool(response.ended)

    async def close(self) -> None:
        if self._channel is not None:
            await self._channel.close()

    def _metadata(self) -> tuple[tuple[str, str], ...]:
        return (("authorization", f"Bearer {self.token}"),) if self.token else ()


def _observation(value) -> RemoteObservation:
    return RemoteObservation(
        episode_id=value.trial_id,
        frame_id=int(value.frame_id),
        state=[float(item) for item in value.state],
        images=[
            RemoteImage(
                camera=item.camera,
                data=bytes(item.data),
                content_type=item.content_type or "image/jpeg",
                width=int(item.width) or None,
                height=int(item.height) or None,
            )
            for item in value.images
        ],
        task=value.task or None,
        done=bool(value.done),
        metadata=_struct_to_dict(value.metadata),
    )


def _struct_to_dict(value) -> dict[str, Any]:
    return MessageToDict(value, preserving_proto_field_name=True) if value else {}
