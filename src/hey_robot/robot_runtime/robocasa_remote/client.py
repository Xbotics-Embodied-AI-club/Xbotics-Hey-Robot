from __future__ import annotations

from typing import Any

import grpc
from google.protobuf.json_format import MessageToDict

from hey_robot.robocasa_runtime.v1 import (
    robocasa_runtime_pb2 as _robocasa_runtime_pb2,
    robocasa_runtime_pb2_grpc,
)
from hey_robot.robot_runtime.robocasa_remote.protocol import (
    RemoteImage,
    RemoteObservation,
    RemoteStep,
)

# Protobuf message attributes are installed dynamically by generated code.
# Keep this untyped protocol boundary in the transport adapter only.
robocasa_runtime_pb2: Any = _robocasa_runtime_pb2


class GrpcRoboCasaRuntimeClient:
    """Async client for the isolated RoboCasa runtime container."""

    def __init__(self, target: str, *, timeout_sec: float = 10.0) -> None:
        normalized = target.removeprefix("grpc://")
        if not normalized:
            raise ValueError("RoboCasa runtime target must not be empty")
        self.target = normalized
        self.timeout_sec = float(timeout_sec)
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
            robocasa_runtime_pb2.HealthRequest(), timeout=self.timeout_sec
        )
        return {
            "online": response.online,
            "loaded": response.loaded,
            "busy": response.busy,
            "error": response.error_message or None,
            "metrics": _struct_to_dict(response.metrics),
        }

    async def create_episode(self, *, task: str, seed: int) -> RemoteObservation:
        response = await self._runtime_stub().CreateEpisode(
            robocasa_runtime_pb2.CreateEpisodeRequest(task=task, seed=seed),
            timeout=self.timeout_sec,
        )
        return _observation(response.observation)

    async def observe(self, *, episode_id: str) -> RemoteObservation:
        response = await self._runtime_stub().Observe(
            robocasa_runtime_pb2.EpisodeRequest(episode_id=episode_id),
            timeout=self.timeout_sec,
        )
        return _observation(response)

    async def step(
        self, *, episode_id: str, action: list[float], expected_frame_id: int
    ) -> RemoteStep:
        response = await self._runtime_stub().Step(
            robocasa_runtime_pb2.StepRequest(
                episode_id=episode_id,
                action=action,
                expected_frame_id=expected_frame_id,
            ),
            timeout=self.timeout_sec,
        )
        return RemoteStep(
            observation=_observation(response.observation),
            reward=float(response.reward),
            done=bool(response.done),
            success=bool(response.success),
            metrics=_struct_to_dict(response.metrics),
        )

    async def reset(self, *, episode_id: str) -> RemoteObservation:
        response = await self._runtime_stub().Reset(
            robocasa_runtime_pb2.EpisodeRequest(episode_id=episode_id),
            timeout=self.timeout_sec,
        )
        return _observation(response)

    async def close_episode(self, *, episode_id: str) -> bool:
        response = await self._runtime_stub().CloseEpisode(
            robocasa_runtime_pb2.EpisodeRequest(episode_id=episode_id),
            timeout=self.timeout_sec,
        )
        return bool(response.closed)

    async def close(self) -> None:
        if self._channel is not None:
            await self._channel.close()


def _observation(value) -> RemoteObservation:
    return RemoteObservation(
        episode_id=value.episode_id,
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
        success=bool(value.success),
        metadata=_struct_to_dict(value.metadata),
    )


def _struct_to_dict(value) -> dict[str, Any]:
    return MessageToDict(value, preserving_proto_field_name=True) if value else {}
