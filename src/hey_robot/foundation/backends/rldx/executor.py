"""RLDX-1 backend for the Hey Robot ModelService contract."""

from __future__ import annotations

import base64
import io
import os
import subprocess
import time
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path, PurePosixPath
from threading import Event, Lock
from typing import Any, Protocol, cast

import numpy as np
from PIL import Image

from hey_robot.config import ModelServiceSpec
from hey_robot.foundation.clients.models import PolicyStepResult


class PolicyExecutionError(RuntimeError):
    def __init__(self, failure_mode: str, message: str) -> None:
        super().__init__(message)
        self.failure_mode = failure_mode


class RLDXClient(Protocol):
    def ping(self) -> bool: ...

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any]
    ) -> Any: ...

    def close(self) -> None: ...


class RLDXPolicyExecutor:
    """Serve RLDX-1 through its official ZeroMQ policy-server interface."""

    _ACTION_KEYS = (
        "end_effector_position",
        "end_effector_rotation",
        "gripper_close",
        "base_motion",
        "control_mode",
    )

    def __init__(
        self,
        service_id: str,
        spec: ModelServiceSpec,
        *,
        client_factory: Callable[[str, int, int], RLDXClient] | None = None,
        process_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.service_id = service_id
        self.spec = spec
        self.settings = dict(spec.settings)
        self.policy_path = str(self.settings.get("policy_path") or "").strip()
        if not self.policy_path:
            raise ValueError("RLDX policy_path is required")
        self.device = str(self.settings.get("policy_device") or "cuda")
        self.embodiment = str(self.settings.get("embodiment") or spec.robot_id)
        self.action_space = str(
            self.settings.get("action_space") or "embodiment_native"
        )
        self.action_dimensions = int(self.settings.get("action_dimensions") or 0)
        if self.action_dimensions <= 0:
            raise ValueError("RLDX action_dimensions must be positive")
        self.state_dimensions = int(self.settings.get("state_dimensions") or 16)
        if self.state_dimensions != 16:
            raise ValueError("RLDX RoboCasa365 state_dimensions must be 16")
        self.execution_horizon = int(self.settings.get("execution_horizon") or 8)
        if not 1 <= self.execution_horizon <= 16:
            raise ValueError("RLDX execution_horizon must be in [1, 16]")
        self.prompt_mode = str(self.settings.get("prompt_mode") or "environment_root")
        if self.prompt_mode not in {"environment_root", "agent_subgoal"}:
            raise ValueError(
                "RLDX prompt_mode must be environment_root or agent_subgoal"
            )
        # base_clip > 0 clamps the VLA's base_motion (indices 7:11 of the 12D
        # action) to [-clip, clip], forcing the arm to do precise contact work
        # 0 = full base motion; a positive value limits base motion during
        # close-range manipulation.
        self.base_clip = float(self.settings.get("base_clip") or 0.0)
        self.host = str(self.settings.get("server_host") or "127.0.0.1")
        self.port = int(self.settings.get("server_port") or 5555)
        self.video_delta_indices = tuple(
            int(value)
            for value in self.settings.get("video_delta_indices", (-6, -4, -2, 0))
        )
        if (
            not self.video_delta_indices
            or self.video_delta_indices[-1] != 0
            or any(value > 0 for value in self.video_delta_indices)
        ):
            raise ValueError(
                "RLDX video_delta_indices must be non-empty, non-positive, and end in 0"
            )
        timeout_ms = max(
            1, int(float(self.settings.get("request_timeout_sec") or 600.0) * 1000)
        )
        self._client_factory = client_factory or _RLDXWireClient
        self._process_factory = process_factory or subprocess.Popen
        self._timeout_ms = timeout_ms
        self._client: RLDXClient | None = None
        self._process: Any | None = None
        self._action_queue: deque[np.ndarray] = deque()
        self._video_history: deque[dict[str, np.ndarray]] = deque(
            maxlen=1 - min(self.video_delta_indices)
        )
        self._cancel_event = Event()
        self._lock = Lock()
        self._loaded = False
        self._session_id: str | None = None
        self._policy_task: str | None = None
        self._needs_memory_reset = True
        self._last_video_frame_id: int | None = None
        self._last_error: str | None = None

    def health(self) -> dict[str, Any]:
        process_running = self._process is None or self._process.poll() is None
        return {
            "name": self.service_id,
            "robot_id": self.spec.robot_id,
            "online": process_running,
            "loaded": self._loaded and process_running,
            "error": self._last_error,
            "metrics": {
                "type": self.spec.type,
                "runtime": "rldx",
                "policy_path": self.policy_path,
                "policy_type": "rldx-1",
                "device": self.device,
                "embodiment": self.embodiment,
                "action_space": self.action_space,
                "action_dimensions": self.action_dimensions,
                "execution_horizon": self.execution_horizon,
                "video_delta_indices": list(self.video_delta_indices),
                "prompt_mode": self.prompt_mode,
                "active_session": self._session_id,
                "queued_actions": len(self._action_queue),
                "server": f"tcp://{self.host}:{self.port}",
                "hardware_ownership": "none",
            },
        }

    def create_chunk_policy(self, observation_encoder):
        """Create the foundation-owned local policy after starting RLDX once."""
        self.load()
        assert self._client is not None
        from hey_robot.foundation.backends.rldx.option_policy import RLDXChunkPolicy

        return RLDXChunkPolicy(
            self._client,
            settings=self.settings,
            observation_encoder=observation_encoder,
        )

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            if self._client is None:
                self._client = self._client_factory(
                    self.host, self.port, self._timeout_ms
                )
            # A policy process may be deployed independently from the harness.
            # Reuse a healthy endpoint instead of spawning a second process on
            # the same address (and, typically, a second full model on one GPU).
            try:
                if self._client.ping():
                    self._loaded = True
                    self._last_error = None
                    return
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
            self._start_server()
            timeout = float(self.settings.get("server_startup_timeout_sec") or 1200)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self._process is not None and self._process.poll() is not None:
                    raise PolicyExecutionError(
                        "policy_load_failed",
                        f"RLDX server exited with {self._process.returncode}",
                    )
                try:
                    if self._client.ping():
                        self._loaded = True
                        self._last_error = None
                        return
                except Exception as exc:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(1.0)
            raise PolicyExecutionError(
                "policy_load_failed",
                f"RLDX server did not become ready within {timeout:.0f}s"
                + (f": {self._last_error}" if self._last_error else ""),
            )

    def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        started_at = time.monotonic()
        try:
            skill_name = str(payload.get("skill_name") or "")
            if self.spec.provides and skill_name not in self.spec.provides:
                raise PolicyExecutionError(
                    "invalid_task",
                    f"model service {self.service_id} does not provide {skill_name!r}",
                )
            request = self._request(payload)
            if not self._loaded:
                self.load()
            with self._lock:
                if request[0] != self._session_id or request[1] != self._policy_task:
                    self._session_id = request[0]
                    self._policy_task = request[1]
                    self._needs_memory_reset = True
                    self._action_queue.clear()
                    self._video_history.clear()
                    self._last_video_frame_id = None
                if self._cancel_event.is_set():
                    self._cancel_event.clear()
                    return _cancelled_result()
                # The option runner executes a whole policy chunk before the
                # next inference.  It returns the intermediate observations so
                # the server can retain the same per-simulator-step video
                # cadence as the policy's per-simulator-step rollout.
                for history_frame, history_observation in _history_observations(
                    payload
                ):
                    self._append_video_frame(history_frame, history_observation)
                self._append_video_frame(request[3], request[2])
                inference_performed = False
                if not self._action_queue:
                    observation = _rldx_observation(
                        request[2],
                        request[1],
                        settings=self.settings,
                        video_history=self._video_history,
                        video_delta_indices=self.video_delta_indices,
                    )
                    assert self._client is not None
                    response = self._client.get_action(
                        observation,
                        {
                            "session_ids": [request[0]],
                            "reset_memory": [self._needs_memory_reset],
                        },
                    )
                    self._needs_memory_reset = False
                    self._action_queue.extend(
                        _action_chunk(
                            response,
                            action_keys=self._ACTION_KEYS,
                            dimensions=self.action_dimensions,
                            execution_horizon=self.execution_horizon,
                            base_clip=self.base_clip,
                        )
                    )
                    inference_performed = True
                # RLDX predicts an action chunk.  Return the complete chunk to
                # the option runner so it can execute its actions consecutively,
                # preserving ``predict -> 8 env.step`` cadence. Returning
                # one queued action here made Hey perform a model/RPC round-trip
                # for every simulator step.
                raw_actions = [
                    self._action_queue.popleft() for _ in range(len(self._action_queue))
                ]
            actions = [
                _clip_action(raw_action, self.settings) for raw_action in raw_actions
            ]
            clipped = any(
                not np.array_equal(action, raw_action)
                for action, raw_action in zip(actions, raw_actions, strict=True)
            )
            if self._cancel_event.is_set():
                self._cancel_event.clear()
                return _cancelled_result()
            primitives = [
                {
                    "name": "embodiment_native_action",
                    "arguments": {
                        "values": [float(value) for value in action.tolist()],
                        "raw_values": [float(value) for value in raw_action.tolist()],
                        "action_space": self.action_space,
                        "embodiment": self.embodiment,
                    },
                }
                for action, raw_action in zip(actions, raw_actions, strict=True)
            ]
            policy_result = PolicyStepResult(
                kind="action_chunk",
                action_space=self.action_space,
                embodiment=self.embodiment,
                horizon=len(primitives),
                dt=1.0 / max(float(self.settings.get("control_hz") or 20.0), 0.1),
                actions=primitives,
                done=False,
                raw={
                    "policy_type": "rldx-1",
                    "expected_frame_id": request[3],
                    "action_clipped": clipped,
                    "inference_performed": inference_performed,
                    "queued_actions": len(self._action_queue),
                },
            ).to_metrics()
            self._last_error = None
            return {
                "success": True,
                "status": "completed",
                "summary": "RLDX-1 produced one native action",
                "metrics": {
                    "policy_result": policy_result,
                    "action_chunk": policy_result,
                    "policy_type": "rldx-1",
                    "prompt_mode": self.prompt_mode,
                    "action_clipped": clipped,
                    "inference_performed": inference_performed,
                    "duration_sec": round(time.monotonic() - started_at, 3),
                },
            }
        except PolicyExecutionError as exc:
            self._last_error = str(exc)
            return {
                "success": False,
                "status": "failed",
                "summary": f"RLDX inference rejected: {exc}",
                "failure_mode": exc.failure_mode,
                "error": str(exc),
            }
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            return {
                "success": False,
                "status": "failed",
                "summary": f"RLDX inference failed: {type(exc).__name__}: {exc}",
                "failure_mode": "execution_failed",
                "error": str(exc),
            }

    def cancel(self) -> None:
        self._cancel_event.set()
        with self._lock:
            self._action_queue.clear()
            self._video_history.clear()
            self._session_id = None
            self._policy_task = None
            self._needs_memory_reset = True
            self._last_video_frame_id = None

    def close(self) -> None:
        self.cancel()
        client = self._client
        self._client = None
        if client is not None:
            with suppress(Exception):
                client.close()
        process = self._process
        self._process = None
        self._loaded = False
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    def _start_server(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        repo = Path(
            str(self.settings.get("rldx_repo") or "/root/.cache/rldx-src/RLDX-1")
        )
        python = Path(
            str(self.settings.get("rldx_python") or "/root/.venv-rldx/bin/python")
        )
        if not repo.is_dir():
            raise PolicyExecutionError(
                "policy_load_failed", f"RLDX repository does not exist: {repo}"
            )
        if not python.is_file():
            raise PolicyExecutionError(
                "policy_load_failed", f"RLDX Python does not exist: {python}"
            )
        command = [
            str(python),
            "-m",
            "hey_robot.foundation.backends.rldx.server",
            "--model-path",
            self.policy_path,
            "--embodiment-tag",
            str(self.settings.get("embodiment_tag") or "GENERAL_EMBODIMENT"),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--device",
            self.device,
            "--image-max-area",
            str(int(self.settings.get("image_max_area") or 65536)),
        ]
        environment = dict(os.environ)
        environment.setdefault("PYTHONUNBUFFERED", "1")
        # Evaluation inputs must already be provisioned.  RLDX constructs its
        # VLM through several HuggingFace entry points, not only
        # AutoProcessor, so enforce offline resolution for the whole worker.
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        cache_dir = str(self.settings.get("hf_home") or "").strip()
        if cache_dir:
            environment["HF_HOME"] = cache_dir
        endpoint = str(self.settings.get("hf_endpoint") or "").strip()
        if endpoint:
            environment["HF_ENDPOINT"] = endpoint
        if bool(self.settings.get("hf_xet_high_performance", False)):
            environment["HF_XET_HIGH_PERFORMANCE"] = "1"
        attention_impl = str(self.settings.get("attention_impl") or "").strip()
        if attention_impl:
            environment["RLDX_ATTN_IMPL"] = attention_impl
        self._process = self._process_factory(
            command,
            cwd=str(repo),
            env=environment,
        )

    def _request(self, payload: dict[str, Any]) -> tuple[str, str, dict[str, Any], int]:
        arguments = dict(payload.get("arguments", {}) or {})
        observation = dict(arguments.get("observation") or {})
        if not observation:
            raise PolicyExecutionError(
                "observation_unavailable", "RLDX policy requires an observation"
            )
        observation_raw = dict(observation.get("raw") or {})
        agent_subgoal = str(
            arguments.get("agent_subgoal")
            or arguments.get("task_prompt")
            or payload.get("objective")
            or ""
        ).strip()
        environment_root = str(
            observation_raw.get("policy_task")
            or arguments.get("policy_task")
            or payload.get("objective")
            or ""
        ).strip()
        task = (
            environment_root
            if self.prompt_mode == "environment_root"
            else agent_subgoal
        )
        if not task:
            # Safety net: if the agent produced no usable task text, fall back
            # to the environment's official task language (policy_task), which
            # the VLA was trained on.
            task = environment_root
        if not task:
            raise PolicyExecutionError("invalid_task", "policy task is required")
        return (
            str(
                arguments.get("policy_session_id")
                or payload.get("episode_id")
                or "default"
            ),
            task,
            observation,
            int(observation.get("frame_id") or 0),
        )

    def _append_video_frame(self, frame_id: int, observation: dict[str, Any]) -> None:
        if frame_id == self._last_video_frame_id:
            return
        self._video_history.append(
            _rldx_video_frame(observation, settings=self.settings)
        )
        self._last_video_frame_id = frame_id


def _history_observations(payload: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    """Read optional per-action observations supplied by the VLA chunk runner."""
    arguments = dict(payload.get("arguments", {}) or {})
    values = arguments.get("observation_history")
    if not isinstance(values, list):
        return []
    history: list[tuple[int, dict[str, Any]]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        try:
            frame_id = int(value.get("frame_id") or 0)
        except (TypeError, ValueError):
            continue
        if frame_id >= 0:
            history.append((frame_id, value))
    return history


class _RLDXWireClient:
    def __init__(self, host: str, port: int, timeout_ms: int) -> None:
        try:
            import msgpack
            import zmq
        except Exception as exc:
            raise RuntimeError(
                "RLDX client requires msgpack and pyzmq in the model-service environment"
            ) from exc
        self._msgpack = msgpack
        self._zmq = zmq
        self._host = host
        self._port = port
        self._timeout_ms = timeout_ms
        self._context = zmq.Context()
        self._socket: Any | None = None
        self._connect()

    def _connect(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = self._context.socket(self._zmq.REQ)
        self._socket.setsockopt(self._zmq.LINGER, 0)
        self._socket.setsockopt(self._zmq.SNDTIMEO, self._timeout_ms)
        self._socket.setsockopt(self._zmq.RCVTIMEO, self._timeout_ms)
        self._socket.connect(f"tcp://{self._host}:{self._port}")

    def ping(self) -> bool:
        try:
            response = self._call("ping", requires_input=False)
            return isinstance(response, dict) and response.get("status") == "ok"
        except Exception:
            self._connect()
            return False

    def get_action(self, observation: dict[str, Any], options: dict[str, Any]) -> Any:
        return self._call(
            "get_action", {"observation": observation, "options": options}
        )

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        self._context.term()

    def _call(
        self,
        endpoint: str,
        data: dict[str, Any] | None = None,
        *,
        requires_input: bool = True,
    ) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        try:
            assert self._socket is not None
            self._socket.send(self._to_bytes(request))
            response = self._from_bytes(self._socket.recv())
        except self._zmq.ZMQError:
            self._connect()
            raise
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"RLDX server error: {response['error']}")
        return response

    def _to_bytes(self, value: Any) -> bytes:
        return cast(bytes, self._msgpack.packb(value, default=_encode_wire_value))

    def _from_bytes(self, value: bytes) -> Any:
        return self._msgpack.unpackb(value, object_hook=_decode_wire_value)


def _encode_wire_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        output = io.BytesIO()
        np.save(output, value, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": output.getvalue()}
    return value


def _decode_wire_value(value: Any) -> Any:
    if isinstance(value, dict) and "__ndarray_class__" in value:
        return np.load(io.BytesIO(value["as_npy"]), allow_pickle=False)
    return value


def _rldx_observation(
    payload: dict[str, Any],
    task: str,
    *,
    settings: dict[str, Any],
    video_history: deque[dict[str, np.ndarray]] | None = None,
    video_delta_indices: tuple[int, ...] = (-6, -4, -2, 0),
) -> dict[str, Any]:
    state = np.asarray(payload.get("proprioception", []), dtype=np.float32)
    if state.shape != (16,) or not np.isfinite(state).all():
        raise PolicyExecutionError(
            "observation_schema_mismatch",
            f"RLDX RoboCasa365 state must have shape (16,), got {state.shape}",
        )
    camera_names = _camera_names(settings)
    history = list(video_history or ())
    if not history:
        history = [_rldx_video_frame(payload, settings=settings)]
    selected = [
        history[max(0, len(history) - 1 + delta)] for delta in video_delta_indices
    ]
    video_keys = (
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    )
    observation: dict[str, Any] = {
        f"video.{key}": np.stack([frame[camera] for frame in selected], axis=0)[
            None, ...
        ]
        for key, camera in zip(video_keys, camera_names, strict=True)
    }
    observation.update(
        {
            "state.end_effector_position_relative": state[0:3][None, None, :],
            "state.end_effector_rotation_relative": state[3:7][None, None, :],
            "state.gripper_qpos": state[14:16][None, None, :],
            "state.base_position": state[7:10][None, None, :],
            "state.base_rotation": state[10:14][None, None, :],
            "annotation.human.task_description": [task],
        }
    )
    return observation


def _camera_names(settings: dict[str, Any]) -> tuple[str, ...]:
    camera_names = tuple(
        str(value)
        for value in settings.get("camera_names", ("camera1", "camera2", "camera3"))
    )
    if len(camera_names) != 3:
        raise PolicyExecutionError(
            "observation_schema_mismatch", "RLDX requires exactly three cameras"
        )
    return camera_names


def _rldx_video_frame(
    payload: dict[str, Any], *, settings: dict[str, Any]
) -> dict[str, np.ndarray]:
    camera_names = _camera_names(settings)
    images: dict[str, np.ndarray] = {}
    for item in list(payload.get("images", []) or []):
        image = dict(item or {})
        camera = str(image.get("camera") or "")
        if camera not in camera_names:
            continue
        try:
            with Image.open(io.BytesIO(_image_bytes(image, settings))) as source:
                images[camera] = np.asarray(source.convert("RGB"), dtype=np.uint8)
        except Exception as exc:
            raise PolicyExecutionError(
                "observation_schema_mismatch", f"invalid {camera} image: {exc}"
            ) from exc
    if set(images) != set(camera_names):
        raise PolicyExecutionError(
            "observation_schema_mismatch",
            f"RLDX requires cameras {list(camera_names)}, got {sorted(images)}",
        )
    return images


def _action_chunk(
    response: Any,
    *,
    action_keys: tuple[str, ...],
    dimensions: int,
    execution_horizon: int,
    base_clip: float = 0.0,
) -> list[np.ndarray]:
    action = response[0] if isinstance(response, list | tuple) else response
    if not isinstance(action, dict):
        raise PolicyExecutionError(
            "action_schema_mismatch", "RLDX response does not contain an action dict"
        )
    components: list[np.ndarray] = []
    horizon: int | None = None
    for key in action_keys:
        wire_key = f"action.{key}"
        if wire_key not in action:
            raise PolicyExecutionError(
                "action_schema_mismatch", f"RLDX action is missing {wire_key}"
            )
        values = np.asarray(action[wire_key], dtype=np.float32)
        if values.ndim != 3 or values.shape[0] != 1:
            raise PolicyExecutionError(
                "action_schema_mismatch",
                f"RLDX {wire_key} must have shape (1,T,D), got {values.shape}",
            )
        horizon = values.shape[1] if horizon is None else horizon
        if values.shape[1] != horizon:
            raise PolicyExecutionError(
                "action_schema_mismatch", "RLDX action components disagree on horizon"
            )
        components.append(values[0])
    merged = np.concatenate(components, axis=-1)
    if merged.shape[1] != dimensions or horizon is None:
        raise PolicyExecutionError(
            "action_schema_mismatch",
            f"RLDX action must have {dimensions} dimensions, got {merged.shape}",
        )
    if horizon < execution_horizon:
        raise PolicyExecutionError(
            "action_schema_mismatch",
            f"RLDX returned {horizon} actions, requires {execution_horizon}",
        )
    if not np.isfinite(merged).all():
        raise PolicyExecutionError(
            "action_schema_mismatch", "RLDX action contains non-finite values"
        )
    if base_clip > 0.0 and merged.shape[1] >= 11:
        # Clamp base_motion (4 dims at [7:11]) so the arm does the precise work.
        merged[:, 7:11] = np.clip(merged[:, 7:11], -base_clip, base_clip)
    return [row.astype(np.float32, copy=False) for row in merged[:execution_horizon]]


def _image_bytes(image: dict[str, Any], settings: dict[str, Any]) -> bytes:
    encoded = str(image.get("data") or "")
    if encoded:
        return base64.b64decode(encoded, validate=True)
    uri = str(image.get("uri") or "")
    root_value = str(settings.get("media_root") or "").strip()
    prefix = "media://local/"
    if not uri.startswith(prefix) or not root_value:
        raise ValueError("image requires base64 data or a configured local media URI")
    relative = PurePosixPath(uri.removeprefix(prefix))
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError("unsafe local media URI")
    root = Path(root_value).resolve()
    path = (root / Path(*relative.parts)).resolve()
    path.relative_to(root)
    return path.read_bytes()


def _clip_action(action: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    low = np.asarray(settings.get("action_low", -1.0), dtype=np.float32)
    high = np.asarray(settings.get("action_high", 1.0), dtype=np.float32)
    return np.clip(action, low, high).astype(np.float32, copy=False)


def _cancelled_result() -> dict[str, Any]:
    return {
        "success": False,
        "status": "cancelled",
        "summary": "RLDX inference cancelled",
        "failure_mode": "cancelled",
    }
