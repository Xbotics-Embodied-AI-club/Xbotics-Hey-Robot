from __future__ import annotations

import asyncio
import io
import os
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import grpc
import numpy as np
from google.protobuf.json_format import ParseDict
from google.protobuf.struct_pb2 import Struct
from PIL import Image

from hey_robot.foundation.options import LocalPolicyOptionRunner, OptionRequest
from hey_robot.robocasa_backend.contract import (
    ALLOWED_TASKS,
    CAMERA_RENAME_MAP,
    DEFAULT_REGISTRIES,
    DEFAULT_SPLIT,
)
from hey_robot.robocasa_backend.episode_manager import (
    ActiveTrial,
    EpisodeManager,
)
from hey_robot.robocasa_backend.rpc.v1 import (
    robocasa_runtime_pb2 as _robocasa_runtime_pb2,
    robocasa_runtime_pb2_grpc,
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
_CAMERA_SOURCES = {target: source for source, target in _CAMERA_ALIASES.items()}


class RoboCasaRuntimeService(robocasa_runtime_pb2_grpc.RoboCasaRuntimeServicer):
    """One causal RoboCasa episode at a time, isolated inside the backend."""

    def __init__(
        self,
        *,
        manager: EpisodeManager | None = None,
        resource_lock: asyncio.Lock | None = None,
        evaluator_token: str | None = None,
        data_token: str | None = None,
        create_option_runner: Callable[[EpisodeManager], LocalPolicyOptionRunner]
        | None = None,
        option_timeout_sec: float = 1800.0,
    ) -> None:
        self.manager = manager or EpisodeManager(allowed_tasks=ALLOWED_TASKS)
        self._lock = asyncio.Lock()
        self._resource_lock = resource_lock or asyncio.Lock()
        self._owns_resource = False
        self._last_error: str | None = None
        self._evaluator_token = evaluator_token
        self._data_token = data_token
        self._create_option_runner = create_option_runner
        self._option_runner: LocalPolicyOptionRunner | None = None
        self._option_timeout_sec = max(float(option_timeout_sec), 0.001)

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
                    execution_artifact_dir=request.execution_artifact_dir or None,
                )
                trial = await asyncio.to_thread(
                    self.manager.begin_trial,
                    spec,
                )
                if self._create_option_runner is None:
                    raise RuntimeError(
                        "RoboCasa backend has no foundation policy runner"
                    )
                self._option_runner = await asyncio.to_thread(
                    self._create_option_runner, self.manager
                )
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
        session_id = str(request.session_id).strip()
        instruction = str(request.instruction).strip()
        if not session_id or not instruction or int(request.max_actions) < 1:
            raise ValueError("Step requires session_id, instruction, and max_actions")
        async with self._lock:
            started = time.monotonic()
            try:
                if not self.manager.active or self._option_runner is None:
                    raise RuntimeError("trial_unavailable: begin a trial before Step")
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._option_runner.run,
                        OptionRequest(
                            session_id=session_id,
                            instruction=instruction,
                            max_actions=int(request.max_actions),
                            reset_session=bool(request.reset_session),
                        ),
                    ),
                    timeout=self._option_timeout_sec,
                )
            except TimeoutError as exc:
                self._last_error = (
                    f"RoboCasa policy option exceeded {self._option_timeout_sec:.1f}s"
                )
                await context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, self._last_error)
                raise AssertionError("context.abort must not return") from exc
            trial = self.manager.current_trial()
            return robocasa_runtime_pb2.StepResponse(
                observation=self._response_observation(trial),
                status=result.status.value,
                done=result.environment_done,
                actions_executed=result.actions_executed,
                chunks_executed=result.chunks_executed,
                progress=_struct(result.progress),
                diagnostics=_struct(
                    {
                        **result.diagnostics,
                        "duration_sec": round(time.monotonic() - started, 3),
                    }
                ),
                error_message=result.error or "",
            )

    async def StepNative(self, request, context):  # noqa: N802
        """Advance exactly one raw 12-D environment action.

        This is intentionally separate from ``Step``: it is a robot action
        transport, never a model-inference endpoint.
        """
        await self._authorize(context, role="data")
        async with self._lock:
            try:
                outcome = await asyncio.to_thread(
                    self.manager.step,
                    list(request.action),
                    expected_frame_id=int(request.expected_frame_id),
                )
            except Exception as exc:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
                raise AssertionError("context.abort must not return") from exc
            trial = self.manager.current_trial()
            return robocasa_runtime_pb2.NativeStepResponse(
                observation=self._response_observation(trial),
                done=bool(outcome.done),
                progress=_struct(
                    _json_safe(self.manager.read_truth().get("last_info", {}))
                ),
            )

    async def LocalizePixels(self, request, context):  # noqa: N802
        """Return RPent-style metric-depth world coordinates for selected pixels."""
        await self._authorize(context, role="data")
        async with self._lock:
            camera = _CAMERA_SOURCES.get(str(request.camera))
            if camera is None:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"unsupported RoboCasa camera {request.camera!r}",
                )
                raise AssertionError("context.abort must not return")
            try:
                localization = await asyncio.to_thread(
                    self.manager.localize_pixels,
                    camera=camera,
                    pixels=[
                        [int(pixel.row), int(pixel.col)] for pixel in request.pixels
                    ],
                    expected_frame_id=int(request.expected_frame_id),
                )
            except Exception as exc:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
                raise AssertionError("context.abort must not return") from exc
            localization["camera"] = str(request.camera)
            return robocasa_runtime_pb2.LocalizePixelsResponse(
                frame_id=int(localization["frame_id"]),
                camera=str(request.camera),
                localization=_struct(_json_safe(localization)),
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
            self._option_runner = None
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
                    "cameras": _camera_calibration(trial, pixels),
                    "task_progress": self.manager.get_task_progress(),
                    "execution_diagnostics": self.manager.get_execution_diagnostics(),
                }
            ),
        )


def _png(frame: Any) -> bytes:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _camera_calibration(trial: ActiveTrial, pixels: dict[str, Any]) -> dict[str, Any]:
    """Expose per-camera intrinsics + cam2world extrinsics for agent localization.

    The calibration is read live from the MuJoCo sim so it stays correct even
    after the mobile base / arm moves (agentview and wrist cameras are mounted
    on the robot). Rendering is NOT triggered; only the camera model state is
    read, so this is cheap enough to attach to every observation.
    """
    raw_env = getattr(trial.env, "_env", None)
    sim = getattr(getattr(raw_env, "env", None), "sim", None)
    if sim is None:
        return {}
    try:
        from robosuite.utils import camera_utils

        out: dict[str, Any] = {}
        for camera, frame in pixels.items():
            h = int(np.asarray(frame).shape[0])
            w = int(np.asarray(frame).shape[1])
            intrinsic = np.asarray(
                camera_utils.get_camera_intrinsic_matrix(sim, camera, h, w),
                dtype=np.float64,
            )
            extrinsic = np.asarray(
                camera_utils.get_camera_extrinsic_matrix(sim, camera), dtype=np.float64
            )
            out[camera] = {
                "intrinsic": intrinsic.tolist(),  # 3x3
                "extrinsic_cam2world": extrinsic.tolist(),  # 4x4
                "height": h,
                "width": w,
            }
        return out
    except Exception as exc:  # calibration is best-effort; never fail an observation
        return {"error": f"{type(exc).__name__}: {exc}"}


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
            root / "objects" / "objaverse",
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
