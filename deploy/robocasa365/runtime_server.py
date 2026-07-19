from __future__ import annotations

import asyncio
import io
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from google.protobuf.json_format import ParseDict
from google.protobuf.struct_pb2 import Struct
from PIL import Image

try:
    from rollout import ALLOWED_TASKS, CAMERA_RENAME_MAP, DEFAULT_REGISTRIES
except ModuleNotFoundError:
    from deploy.robocasa365.rollout import (
        ALLOWED_TASKS,
        CAMERA_RENAME_MAP,
        DEFAULT_REGISTRIES,
    )

from hey_robot.robocasa_runtime.v1 import (
    robocasa_runtime_pb2,
    robocasa_runtime_pb2_grpc,
)

_CAMERA_NAMES = tuple(
    source.removeprefix("observation.images.") for source in CAMERA_RENAME_MAP
)
_CAMERA_ALIASES = {
    source.removeprefix("observation.images."): target.removeprefix(
        "observation.images."
    )
    for source, target in CAMERA_RENAME_MAP.items()
}


@dataclass
class _Episode:
    task: str
    seed: int
    env: Any
    frame_id: int
    observation: dict[str, Any]
    done: bool = False
    success: bool = False


class RoboCasaRuntimeService(robocasa_runtime_pb2_grpc.RoboCasaRuntimeServicer):
    """One causal RoboCasa episode at a time, isolated inside the worker image."""

    def __init__(self, *, resource_lock: asyncio.Lock | None = None) -> None:
        self._episodes: dict[str, _Episode] = {}
        self._lock = asyncio.Lock()
        self._resource_lock = resource_lock or asyncio.Lock()
        self._owns_resource = False
        self._last_error: str | None = None

    @property
    def busy(self) -> bool:
        return bool(self._episodes) or self._resource_lock.locked()

    async def GetHealth(self, request, context):  # noqa: N802
        del request, context
        try:
            from lerobot.envs.robocasa import ACTION_DIM, OBS_STATE_DIM

            loaded = _assets_available()
            dimensions = {
                "action_dimensions": ACTION_DIM,
                "state_dimensions": OBS_STATE_DIM,
            }
        except Exception as exc:
            loaded = False
            dimensions = {}
            self._last_error = f"{type(exc).__name__}: {exc}"
        return robocasa_runtime_pb2.HealthResponse(
            online=True,
            loaded=loaded,
            busy=self.busy,
            error_message=self._last_error or "",
            metrics=_struct({"active_episodes": len(self._episodes), **dimensions}),
        )

    async def CreateEpisode(self, request, context):  # noqa: N802
        del context
        if request.task not in ALLOWED_TASKS:
            raise ValueError(f"task {request.task!r} is not allowlisted")
        async with self._lock:
            if self._episodes:
                raise RuntimeError("RoboCasa runtime already has an active episode")
            if self._resource_lock.locked():
                raise RuntimeError("RoboCasa worker is busy with a task-level rollout")
            await self._resource_lock.acquire()
            self._owns_resource = True
            episode_id = f"rc-{uuid.uuid4().hex}"
            try:
                episode = await asyncio.to_thread(
                    self._create_episode, request.task, int(request.seed)
                )
            except Exception:
                self._release_resource()
                raise
            self._episodes[episode_id] = episode
            return robocasa_runtime_pb2.EpisodeResponse(
                observation=self._response_observation(episode_id, episode)
            )

    async def Observe(self, request, context):  # noqa: N802
        del context
        async with self._lock:
            return self._response_observation(
                request.episode_id, self._episode(request.episode_id)
            )

    async def Step(self, request, context):  # noqa: N802
        del context
        action = [float(value) for value in request.action]
        if len(action) != 12 or not all(math.isfinite(value) for value in action):
            raise ValueError("action must contain exactly 12 finite values")
        async with self._lock:
            episode = self._episode(request.episode_id)
            if episode.done:
                raise ValueError("episode is done; call Reset before Step")
            if int(request.expected_frame_id) != episode.frame_id:
                raise ValueError(
                    f"stale action frame {request.expected_frame_id}; current frame is {episode.frame_id}"
                )
            action_array = np.asarray(action, dtype=np.float32)
            if not episode.env.action_space.contains(action_array):
                raise ValueError(
                    "action is outside the RoboCasa environment action_space"
                )
            observation, reward, terminated, truncated, info = await asyncio.to_thread(
                episode.env.step, action_array
            )
            episode.observation = observation
            episode.frame_id += 1
            episode.done = bool(terminated or truncated)
            episode.success = bool(info.get("is_success", False))
            return robocasa_runtime_pb2.StepResponse(
                observation=self._response_observation(request.episode_id, episode),
                reward=float(reward),
                done=episode.done,
                success=episode.success,
                metrics=_struct(
                    {"truncated": bool(truncated), "info": _json_safe(info)}
                ),
            )

    async def Reset(self, request, context):  # noqa: N802
        del context
        async with self._lock:
            episode = self._episode(request.episode_id)
            observation, _ = await asyncio.to_thread(
                episode.env.reset, seed=episode.seed
            )
            _validate_observation(observation)
            episode.observation = observation
            episode.frame_id += 1
            episode.done = False
            episode.success = False
            return self._response_observation(request.episode_id, episode)

    async def CloseEpisode(self, request, context):  # noqa: N802
        del context
        async with self._lock:
            episode = self._episodes.pop(request.episode_id, None)
            if episode is None:
                return robocasa_runtime_pb2.CloseEpisodeResponse(closed=False)
            try:
                await asyncio.to_thread(episode.env.close)
            finally:
                self._release_resource()
            return robocasa_runtime_pb2.CloseEpisodeResponse(closed=True)

    def _release_resource(self) -> None:
        if self._owns_resource:
            self._owns_resource = False
            self._resource_lock.release()

    def _create_episode(self, task: str, seed: int) -> _Episode:
        from lerobot.envs.robocasa import DEFAULT_CAMERAS, RoboCasaEnv

        env = RoboCasaEnv(
            task=task,
            camera_name=DEFAULT_CAMERAS,
            obs_type="pixels_agent_pos",
            obj_registries=DEFAULT_REGISTRIES,
            split="target",
        )
        try:
            observation, _ = env.reset(seed=seed)
            _validate_observation(observation)
        except Exception:
            env.close()
            raise
        return _Episode(
            task=task,
            seed=seed,
            env=env,
            frame_id=0,
            observation=observation,
        )

    def _episode(self, episode_id: str) -> _Episode:
        episode = self._episodes.get(episode_id)
        if episode is None:
            raise ValueError(f"unknown RoboCasa episode: {episode_id}")
        return episode

    def _response_observation(self, episode_id: str, episode: _Episode):
        pixels = dict(episode.observation.get("pixels", {}) or {})
        images = [
            robocasa_runtime_pb2.ImageFrame(
                camera=_CAMERA_ALIASES[camera],
                data=_jpeg(pixels[camera]),
                content_type="image/jpeg",
                width=int(pixels[camera].shape[1]),
                height=int(pixels[camera].shape[0]),
            )
            for camera in _CAMERA_NAMES
            if camera in pixels
        ]
        return robocasa_runtime_pb2.ObservationResponse(
            episode_id=episode_id,
            frame_id=episode.frame_id,
            state=[float(value) for value in episode.observation.get("agent_pos", [])],
            images=images,
            task=episode.task,
            done=episode.done,
            success=episode.success,
            metadata=_struct({"native_cameras": list(_CAMERA_NAMES)}),
        )


def _jpeg(frame: Any) -> bytes:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _validate_observation(observation: dict[str, Any]) -> None:
    state = np.asarray(observation.get("agent_pos", []))
    if state.shape != (16,) or not np.isfinite(state).all():
        raise RuntimeError(
            f"RoboCasa observation state must be 16 finite values, got {state.shape}"
        )
    pixels = dict(observation.get("pixels", {}) or {})
    missing = [camera for camera in _CAMERA_NAMES if camera not in pixels]
    if missing:
        raise RuntimeError(f"RoboCasa observation is missing cameras: {missing}")


def _assets_available() -> bool:
    marker = Path(
        os.environ.get(
            "ROBOCASA_ASSET_READY_FILE",
            "/opt/robocasa/robocasa/models/assets/.robocasa-assets-ready",
        )
    )
    asset_root = Path(
        os.environ.get(
            "ROBOCASA_MODEL_ASSET_ROOT",
            "/opt/robocasa/robocasa/models/assets",
        )
    )
    return marker.is_file() and all(
        path.is_dir()
        for path in (
            asset_root / "textures",
            asset_root / "generative_textures",
            asset_root / "fixtures",
            asset_root / "objects" / "lightwheel",
        )
    )


def _struct(value: dict[str, Any]) -> Struct:
    result = Struct()
    ParseDict(_json_safe(value), result)
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
