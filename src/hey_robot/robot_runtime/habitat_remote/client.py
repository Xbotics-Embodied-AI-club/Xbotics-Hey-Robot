from __future__ import annotations

import asyncio
from typing import Any

import grpc
from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.struct_pb2 import Struct

from hey_robot.habitat_runtime.v1 import (
    habitat_runtime_pb2 as _habitat_runtime_pb2,
    habitat_runtime_pb2_grpc,
)
from hey_robot.robot_runtime.habitat_remote.protocol import (
    HabitatAsset,
    HabitatObservation,
    HabitatSkillResult,
    HabitatStep,
)

habitat_runtime_pb2: Any = _habitat_runtime_pb2


class GrpcHabitatRuntimeClient:
    """Async transport for the isolated Habitat episode runtime."""

    def __init__(self, target: str, *, timeout_sec: float = 60.0) -> None:
        self.target = target.removeprefix("grpc://")
        if not self.target:
            raise ValueError("Habitat runtime target must not be empty")
        self.timeout_sec = float(timeout_sec)
        self._channel: grpc.aio.Channel | None = None
        self._stub: Any | None = None

    def _runtime_stub(self) -> Any:
        if self._stub is None:
            self._channel = grpc.aio.insecure_channel(self.target)
            self._stub = habitat_runtime_pb2_grpc.HabitatRuntimeStub(self._channel)
        return self._stub

    async def health(self) -> dict[str, Any]:
        response = await self._runtime_stub().GetHealth(
            habitat_runtime_pb2.HealthRequest(), timeout=self.timeout_sec
        )
        return {
            "online": bool(response.online),
            "loaded": bool(response.loaded),
            "busy": bool(response.busy),
            "error": response.error_message or None,
            "metrics": _struct_to_dict(response.metrics),
        }

    async def create_episode(
        self,
        *,
        profile: str,
        task: str,
        split: str,
        requested_dataset_episode_id: str | None,
        seed: int,
        controlled_agent: str,
    ) -> HabitatObservation:
        response = await self._runtime_stub().CreateEpisode(
            habitat_runtime_pb2.CreateEpisodeRequest(
                profile=profile,
                task=task,
                split=split,
                requested_dataset_episode_id=requested_dataset_episode_id or "",
                seed=int(seed),
                controlled_agent=controlled_agent,
            ),
            timeout=self.timeout_sec,
        )
        return _observation(response.observation)

    async def observe(self, *, episode_id: str) -> HabitatObservation:
        response = await self._runtime_stub().Observe(
            habitat_runtime_pb2.EpisodeRequest(episode_id=episode_id),
            timeout=self.timeout_sec,
        )
        return _observation(response)

    async def step(
        self, *, episode_id: str, action: dict[str, Any], expected_frame_id: int
    ) -> HabitatStep:
        response = await self._runtime_stub().Step(
            habitat_runtime_pb2.StepRequest(
                episode_id=episode_id,
                action=_struct(action),
                expected_frame_id=int(expected_frame_id),
            ),
            timeout=self.timeout_sec,
        )
        return HabitatStep(
            observation=_observation(response.observation),
            reward=float(response.reward),
            done=bool(response.done),
            success=bool(response.success),
            metrics=_struct_to_dict(response.metrics),
        )

    async def execute_skill(
        self,
        *,
        episode_id: str,
        operation_id: str,
        skill_name: str,
        arguments: dict[str, Any],
        expected_frame_id: int,
        max_steps: int,
    ) -> HabitatSkillResult:
        call = self._runtime_stub().ExecuteSkill(
            habitat_runtime_pb2.ExecuteSkillRequest(
                episode_id=episode_id,
                operation_id=operation_id,
                skill_name=skill_name,
                arguments=_struct(arguments),
                expected_frame_id=int(expected_frame_id),
                max_steps=max(1, int(max_steps)),
            ),
            timeout=self.timeout_sec,
        )
        try:
            response = await call
        except asyncio.CancelledError:
            # Server-side execution survives a cancelled unary RPC unless it is
            # explicitly signalled through the companion cancellation RPC.
            await self.cancel_skill(episode_id=episode_id, operation_id=operation_id)
            raise
        return HabitatSkillResult(
            observation=_observation(response.observation),
            steps=int(response.steps),
            success=bool(response.success),
            failure_mode=response.failure_mode or None,
            metrics=_struct_to_dict(response.metrics),
            trace=[_struct_to_dict(item) for item in response.trace],
            cancelled=bool(response.cancelled),
        )

    async def cancel_skill(self, *, episode_id: str, operation_id: str) -> bool:
        response = await self._runtime_stub().CancelSkill(
            habitat_runtime_pb2.CancelSkillRequest(
                episode_id=episode_id, operation_id=operation_id
            ),
            timeout=min(self.timeout_sec, 10.0),
        )
        return bool(response.accepted)

    async def reset(
        self, *, episode_id: str, expected_frame_id: int
    ) -> HabitatObservation:
        response = await self._runtime_stub().Reset(
            habitat_runtime_pb2.MutateEpisodeRequest(
                episode_id=episode_id, expected_frame_id=int(expected_frame_id)
            ),
            timeout=self.timeout_sec,
        )
        return _observation(response)

    async def close_episode(self, *, episode_id: str, expected_frame_id: int) -> bool:
        response = await self._runtime_stub().CloseEpisode(
            habitat_runtime_pb2.MutateEpisodeRequest(
                episode_id=episode_id, expected_frame_id=int(expected_frame_id)
            ),
            timeout=self.timeout_sec,
        )
        return bool(response.closed)

    async def close(self) -> None:
        if self._channel is not None:
            await self._channel.close()


def _struct(value: dict[str, Any]):
    result = Struct()
    ParseDict(_json_value(value), result)
    return result


def _observation(value: Any) -> HabitatObservation:
    return HabitatObservation(
        episode_id=value.episode_id,
        frame_id=int(value.frame_id),
        assets=[
            HabitatAsset(
                kind=item.kind,
                role=item.role or None,
                name=item.name or None,
                data=bytes(item.data),
                content_type=item.content_type or None,
                width=int(item.width) or None,
                height=int(item.height) or None,
                metadata=_struct_to_dict(item.metadata),
            )
            for item in value.assets
        ],
        proprioception=[float(item) for item in value.proprioception],
        task=value.task or None,
        done=bool(value.done),
        success=bool(value.success),
        metrics=_struct_to_dict(value.metrics),
        entities=[_struct_to_dict(item) for item in value.entities],
        metadata=_struct_to_dict(value.metadata),
    )


def _struct_to_dict(value: Any) -> dict[str, Any]:
    return MessageToDict(value, preserving_proto_field_name=True) if value else {}


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return _json_value(value.item())
    return value
