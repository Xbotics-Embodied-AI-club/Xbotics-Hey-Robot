from __future__ import annotations

import asyncio
import io
import math
import os
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import grpc
import numpy as np
from google.protobuf.json_format import ParseDict
from google.protobuf.struct_pb2 import Struct
from PIL import Image

from hey_robot.robocasa_runtime.v1 import (
    robocasa_runtime_pb2 as _robocasa_runtime_pb2,
    robocasa_runtime_pb2_grpc,
)
from hey_robot.robot_runtime.robocasa_remote.contract import (
    ALLOWED_TASKS,
    CAMERA_RENAME_MAP,
    DEFAULT_REGISTRIES,
    DEFAULT_SPLIT,
)
from hey_robot.robot_runtime.robocasa_remote.episode_manager import (
    ActiveTrial,
    EpisodeManager,
)

robocasa_runtime_pb2: Any = _robocasa_runtime_pb2

_CAMERA_NAMES = tuple(
    source.removeprefix("observation.images.") for source in CAMERA_RENAME_MAP
)
_CAMERA_ALIASES = {
    source.removeprefix("observation.images."): target.removeprefix(
        "observation.images."
    )
    for source, target in CAMERA_RENAME_MAP.items()
}


class RoboCasaRuntimeService(robocasa_runtime_pb2_grpc.RoboCasaRuntimeServicer):
    """One causal RoboCasa episode at a time, isolated inside the backend."""

    def __init__(
        self,
        *,
        manager: EpisodeManager | None = None,
        resource_lock: asyncio.Lock | None = None,
        evaluator_token: str | None = None,
        data_token: str | None = None,
        prepare_trial: Callable[[], None] | None = None,
    ) -> None:
        self.manager = manager or EpisodeManager(allowed_tasks=ALLOWED_TASKS)
        self._lock = asyncio.Lock()
        self._resource_lock = resource_lock or asyncio.Lock()
        self._owns_resource = False
        self._last_error: str | None = None
        self._evaluator_token = evaluator_token
        self._data_token = data_token
        self._prepare_trial = prepare_trial

    @property
    def busy(self) -> bool:
        return self.manager.active or self._resource_lock.locked()

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
            metrics=_struct({"active_trials": int(self.manager.active), **dimensions}),
        )

    async def BeginTrial(self, request, context):  # noqa: N802
        await self._authorize(context, role="evaluator")
        if request.task not in ALLOWED_TASKS:
            raise ValueError(f"task {request.task!r} is not allowlisted")
        async with self._lock:
            if self.manager.active:
                raise RuntimeError("RoboCasa runtime already has an active episode")
            if self._resource_lock.locked():
                raise RuntimeError("RoboCasa backend is busy with a task-level rollout")
            await self._resource_lock.acquire()
            self._owns_resource = True
            try:
                spec = self.manager.new_spec(
                    task=request.task,
                    seed=int(request.seed),
                    trial_id=request.trial_id or None,
                    split=request.split or DEFAULT_SPLIT,
                    registries=tuple(request.registries) or DEFAULT_REGISTRIES,
                )
                trial = await asyncio.to_thread(
                    self.manager.begin_trial,
                    spec,
                )
                if self._prepare_trial is not None:
                    await asyncio.to_thread(self._prepare_trial)
            except Exception:
                if self.manager.active:
                    await asyncio.to_thread(self.manager.end_trial)
                self._release_resource()
                raise
            return self._response_observation(trial)

    async def Observe(self, request, context):  # noqa: N802
        await self._authorize(context, role="data")
        async with self._lock:
            del request
            try:
                return self._response_observation(self.manager.observe())
            except Exception as exc:
                message = str(exc)
                if not self.manager.active:
                    message = f"trial_unavailable: {message}"
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
                raise AssertionError("context.abort must not return") from exc

    async def Step(self, request, context):  # noqa: N802
        await self._authorize(context, role="data")
        action = [float(value) for value in request.action]
        raw_action = [float(value) for value in request.raw_action]
        if len(action) != 12 or not all(math.isfinite(value) for value in action):
            raise ValueError("action must contain exactly 12 finite values")
        if len(raw_action) != 12 or not all(
            math.isfinite(value) for value in raw_action
        ):
            raise ValueError("raw_action must contain exactly 12 finite values")
        async with self._lock:
            outcome = await asyncio.to_thread(
                self.manager.step,
                action,
                expected_frame_id=int(request.expected_frame_id),
                raw_action=raw_action or action,
                action_clipped=bool(request.action_clipped),
            )
            trial = self.manager.current_trial()
            return robocasa_runtime_pb2.StepResponse(
                observation=self._response_observation(trial),
                reward=outcome.reward,
                done=outcome.done,
                metrics=_struct({"truncated": outcome.truncated}),
            )

    async def ReadTruth(self, request, context):  # noqa: N802
        await self._authorize(context, role="evaluator")
        async with self._lock:
            del request
            truth = self.manager.read_truth()
            truth["events"] = self.manager.evaluator_events()
            return robocasa_runtime_pb2.TruthResponse(
                done=bool(truth["episode_done"]),
                official_success=bool(truth["official_success"]),
                frame_id=int(truth["frame_id"]),
                metrics=_struct(_json_safe(truth)),
            )

    async def EndTrial(self, request, context):  # noqa: N802
        await self._authorize(context, role="evaluator")
        async with self._lock:
            del request
            if not self.manager.active:
                return robocasa_runtime_pb2.EndTrialResponse(ended=False)
            await asyncio.to_thread(self.manager.end_trial)
            self._release_resource()
            return robocasa_runtime_pb2.EndTrialResponse(ended=True)

    def _release_resource(self) -> None:
        if self._owns_resource:
            self._owns_resource = False
            self._resource_lock.release()

    async def _authorize(self, context: Any, *, role: str) -> None:
        """Require a role-specific bearer token when backend auth is configured."""
        expected = self._evaluator_token if role == "evaluator" else self._data_token
        if not expected:
            return
        metadata = dict(context.invocation_metadata())
        token = metadata.get("authorization", "").removeprefix("Bearer ")
        if token == expected:
            return
        await context.abort(
            grpc.StatusCode.PERMISSION_DENIED,
            f"RoboCasa {role}-plane credential is required",
        )

    def _response_observation(self, trial: ActiveTrial):
        pixels = dict(trial.observation.get("pixels", {}) or {})
        images = [
            robocasa_runtime_pb2.ImageFrame(
                camera=_CAMERA_ALIASES[camera],
                data=_png(pixels[camera]),
                content_type="image/png",
                width=int(pixels[camera].shape[1]),
                height=int(pixels[camera].shape[0]),
            )
            for camera in _CAMERA_NAMES
            if camera in pixels
        ]
        return robocasa_runtime_pb2.ObservationResponse(
            trial_id=trial.spec.trial_id,
            frame_id=trial.frame_id,
            state=[float(value) for value in trial.observation.get("agent_pos", [])],
            images=images,
            task=trial.spec.task,
            done=trial.done,
            metadata=_struct(
                {
                    "native_cameras": list(_CAMERA_NAMES),
                    "trial_id": trial.spec.trial_id,
                    "seed": trial.spec.seed,
                    "split": trial.spec.split,
                    "registries": list(trial.spec.registries),
                    "policy_task": str(
                        getattr(trial.env, "task_description", "") or trial.spec.task
                    ),
                }
            ),
        )


def _png(frame: Any) -> bytes:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
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
    explicit_root = Path(
        os.environ.get(
            "ROBOCASA_MODEL_ASSET_ROOT",
            "/opt/robocasa/robocasa/models/assets",
        )
    )
    roots = [explicit_root]
    with suppress(Exception):
        import robocasa

        roots.append(Path(robocasa.__file__).resolve().parent / "models" / "assets")
    marker_override = os.environ.get("ROBOCASA_ASSET_READY_FILE")
    for root in dict.fromkeys(roots):
        required = (
            root / "textures",
            root / "generative_textures",
            root / "fixtures",
            root / "objects" / "lightwheel",
        )
        if not all(path.is_dir() for path in required):
            continue
        marker = (
            Path(marker_override)
            if marker_override
            else root / ".robocasa-assets-ready"
        )
        if marker.is_file() or root != explicit_root:
            return True
    return False


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
