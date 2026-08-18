"""Xiaomi-Robotics-1 adapter for the Hey Robot ModelService contract."""

from __future__ import annotations

import base64
import io
import os
import pickle
import socket
import struct
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


class XiaomiClient(Protocol):
    def ping(self) -> bool: ...

    def infer(
        self,
        state_history: np.ndarray,
        image_history: dict[str, np.ndarray],
        instruction: str,
    ) -> np.ndarray: ...

    def close(self) -> None: ...


class XiaomiPolicyExecutor:
    """Serve XR-1 through its official length-prefixed socket protocol."""

    def __init__(
        self,
        service_id: str,
        spec: ModelServiceSpec,
        *,
        client_factory: Callable[..., XiaomiClient] | None = None,
        process_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.service_id = service_id
        self.spec = spec
        self.settings = dict(spec.settings)
        self.policy_path = str(self.settings.get("policy_path") or "").strip()
        if not self.policy_path:
            raise ValueError("Xiaomi policy_path is required")
        self.device = str(self.settings.get("policy_device") or "cuda")
        self.embodiment = str(self.settings.get("embodiment") or spec.robot_id)
        self.action_space = str(
            self.settings.get("action_space") or "embodiment_native"
        )
        self.action_dimensions = int(self.settings.get("action_dimensions") or 0)
        if self.action_dimensions != 12:
            raise ValueError("Xiaomi RoboCasa365 action_dimensions must be 12")
        self.state_dimensions = int(self.settings.get("state_dimensions") or 16)
        if self.state_dimensions != 16:
            raise ValueError("Xiaomi RoboCasa365 input state_dimensions must be 16")
        self.execution_horizon = int(self.settings.get("execution_horizon") or 16)
        if not 1 <= self.execution_horizon <= 16:
            raise ValueError("Xiaomi execution_horizon must be in [1, 16]")
        self.prompt_mode = str(self.settings.get("prompt_mode") or "environment_root")
        if self.prompt_mode not in {"environment_root", "agent_subgoal"}:
            raise ValueError(
                "Xiaomi prompt_mode must be environment_root or agent_subgoal"
            )
        self.host = str(self.settings.get("server_host") or "127.0.0.1")
        if self.host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Xiaomi policy server must use a loopback host")
        self.port = int(self.settings.get("server_port") or 10086)
        self.robot_type = str(self.settings.get("robot_type") or "robocasa365")
        self.crop_ratio = float(self.settings.get("crop_ratio") or 0.95)
        if not 0 < self.crop_ratio <= 1:
            raise ValueError("Xiaomi crop_ratio must be in (0, 1]")
        self.observation_delta_indices = tuple(
            int(value)
            for value in self.settings.get("observation_delta_indices", (-6, -4, -2, 0))
        )
        if (
            not self.observation_delta_indices
            or self.observation_delta_indices[-1] != 0
            or any(value > 0 for value in self.observation_delta_indices)
        ):
            raise ValueError(
                "Xiaomi observation_delta_indices must be non-empty, "
                "non-positive, and end in 0"
            )
        self.manage_server = bool(self.settings.get("manage_server", True))
        timeout_sec = float(self.settings.get("request_timeout_sec") or 600.0)
        self._timeout_sec = max(timeout_sec, 1.0)
        self._client_factory = client_factory or _XiaomiWireClient
        self._process_factory = process_factory or subprocess.Popen
        history_length = 1 - min(self.observation_delta_indices)
        self._image_history: deque[dict[str, np.ndarray]] = deque(maxlen=history_length)
        self._state_history: deque[np.ndarray] = deque(maxlen=history_length)
        self._action_queue: deque[np.ndarray] = deque()
        self._client: XiaomiClient | None = None
        self._process: Any | None = None
        self._cancel_event = Event()
        self._lock = Lock()
        self._loaded = False
        self._session_id: str | None = None
        self._policy_task: str | None = None
        self._last_frame_id: int | None = None
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
                "runtime": "xiaomi",
                "policy_path": self.policy_path,
                "policy_type": "xiaomi-robotics-1",
                "device": self.device,
                "embodiment": self.embodiment,
                "action_space": self.action_space,
                "action_dimensions": self.action_dimensions,
                "execution_horizon": self.execution_horizon,
                "observation_delta_indices": list(self.observation_delta_indices),
                "prompt_mode": self.prompt_mode,
                "active_session": self._session_id,
                "queued_actions": len(self._action_queue),
                "server": f"tcp://{self.host}:{self.port}",
                "managed_server": self.manage_server,
                "hardware_ownership": "none",
            },
        }

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._start_server()
            if self._client is None:
                self._client = self._client_factory(
                    self.host,
                    self.port,
                    self.policy_path,
                    self.robot_type,
                    self.crop_ratio,
                    self._timeout_sec,
                )
            timeout = float(self.settings.get("server_startup_timeout_sec") or 1800)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self._process is not None and self._process.poll() is not None:
                    raise PolicyExecutionError(
                        "policy_load_failed",
                        f"Xiaomi server exited with {self._process.returncode}",
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
                f"Xiaomi server did not become ready within {timeout:.0f}s"
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
            session_id, task, observation, frame_id = self._request(payload)
            if not self._loaded:
                self.load()
            with self._lock:
                if session_id != self._session_id:
                    self._reset_session(session_id, task)
                elif task != self._policy_task:
                    self._switch_task(task)
                if self._cancel_event.is_set():
                    self._cancel_event.clear()
                    return _cancelled_result()
                if frame_id != self._last_frame_id:
                    self._image_history.append(
                        _video_frame(observation, settings=self.settings)
                    )
                    self._state_history.append(_xiaomi_state(observation))
                    self._last_frame_id = frame_id
                inference_performed = False
                if not self._action_queue:
                    states, images = _sample_history(
                        self._state_history,
                        self._image_history,
                        self.observation_delta_indices,
                    )
                    assert self._client is not None
                    action_chunk = self._client.infer(states, images, task)
                    self._action_queue.extend(
                        _action_chunk(
                            action_chunk,
                            dimensions=self.action_dimensions,
                            execution_horizon=self.execution_horizon,
                        )
                    )
                    inference_performed = True
                raw_action = self._action_queue.popleft()
            action = _clip_action(raw_action, self.settings)
            clipped = not np.array_equal(action, raw_action)
            if self._cancel_event.is_set():
                self._cancel_event.clear()
                return _cancelled_result()
            primitive = {
                "name": "embodiment_native_action",
                "arguments": {
                    "values": [float(value) for value in action.tolist()],
                    "raw_values": [float(value) for value in raw_action.tolist()],
                    "action_space": self.action_space,
                    "embodiment": self.embodiment,
                },
            }
            policy_result = PolicyStepResult(
                kind="action_chunk",
                action_space=self.action_space,
                embodiment=self.embodiment,
                horizon=1,
                dt=1.0 / max(float(self.settings.get("control_hz") or 20.0), 0.1),
                actions=[primitive],
                done=False,
                raw={
                    "policy_type": "xiaomi-robotics-1",
                    "expected_frame_id": frame_id,
                    "action_clipped": clipped,
                    "inference_performed": inference_performed,
                    "queued_actions": len(self._action_queue),
                },
            ).to_metrics()
            self._last_error = None
            return {
                "success": True,
                "status": "completed",
                "summary": "Xiaomi-Robotics-1 produced one native action",
                "metrics": {
                    "policy_result": policy_result,
                    "action_chunk": policy_result,
                    "policy_type": "xiaomi-robotics-1",
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
                "summary": f"Xiaomi inference rejected: {exc}",
                "failure_mode": exc.failure_mode,
                "error": str(exc),
            }
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            return {
                "success": False,
                "status": "failed",
                "summary": f"Xiaomi inference failed: {type(exc).__name__}: {exc}",
                "failure_mode": "execution_failed",
                "error": str(exc),
            }

    def cancel(self) -> None:
        self._cancel_event.set()
        with self._lock:
            self._reset_session(None, None)

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

    def _reset_session(self, session_id: str | None, task: str | None) -> None:
        self._session_id = session_id
        self._policy_task = task
        self._action_queue.clear()
        self._image_history.clear()
        self._state_history.clear()
        self._last_frame_id = None

    def _switch_task(self, task: str) -> None:
        """Switch subgoals without discarding the current episode history."""
        self._policy_task = task
        self._action_queue.clear()

    def _start_server(self) -> None:
        if not self.manage_server:
            return
        if self._process is not None and self._process.poll() is None:
            return
        repo = Path(
            str(
                self.settings.get("xiaomi_repo")
                or "/root/.cache/xiaomi-robotics-1-src/Xiaomi-Robotics-1"
            )
        ).expanduser()
        python = Path(
            str(self.settings.get("xiaomi_python") or "/root/.venv-xiaomi/bin/python")
        ).expanduser()
        repo = repo.resolve()
        # Keep the virtualenv's python symlink intact. Path.resolve() would turn it
        # into the base uv interpreter and silently discard the venv site-packages.
        python = python.absolute()
        server = repo / "deploy" / "server.py"
        if not server.is_file():
            raise PolicyExecutionError(
                "policy_load_failed", f"Xiaomi server does not exist: {server}"
            )
        if not python.is_file():
            raise PolicyExecutionError(
                "policy_load_failed", f"Xiaomi Python does not exist: {python}"
            )
        model_path = self.policy_path
        local_model_path = Path(model_path).expanduser()
        if local_model_path.exists():
            model_path = str(local_model_path.resolve())
        command = [
            str(python),
            "-u",
            str(server),
            "--model",
            model_path,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        environment = dict(os.environ)
        environment.setdefault("PYTHONUNBUFFERED", "1")
        environment.setdefault("TOKENIZERS_PARALLELISM", "false")
        cuda_device = _cuda_visible_device(self.device)
        if cuda_device is not None:
            environment["CUDA_VISIBLE_DEVICES"] = cuda_device
        cache_dir = str(self.settings.get("hf_home") or "").strip()
        if cache_dir:
            environment["HF_HOME"] = cache_dir
        self._process = self._process_factory(command, cwd=str(repo), env=environment)

    def _request(self, payload: dict[str, Any]) -> tuple[str, str, dict[str, Any], int]:
        arguments = dict(payload.get("arguments", {}) or {})
        observation = dict(arguments.get("observation") or {})
        if not observation:
            raise PolicyExecutionError(
                "observation_unavailable", "Xiaomi policy requires an observation"
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


class _XiaomiWireClient:
    """Local-only client compatible with Xiaomi's official deploy/server.py."""

    def __init__(
        self,
        host: str,
        port: int,
        model_path: str,
        robot_type: str,
        crop_ratio: float,
        timeout_sec: float,
    ) -> None:
        try:
            import torch
            from transformers import AutoProcessor
        except Exception as exc:
            raise RuntimeError(
                "Xiaomi client requires torch and transformers==4.57.1"
            ) from exc
        self._torch = torch
        self._host = host
        self._port = port
        self._robot_type = robot_type
        self._crop_ratio = crop_ratio
        self._timeout_sec = timeout_sec
        self._socket: socket.socket | None = None
        self._processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        robot_types = self._processor.list_robot_types()
        if robot_type not in robot_types:
            raise ValueError(
                f"robot type {robot_type!r} is missing from checkpoint; "
                f"available: {robot_types}"
            )

    def ping(self) -> bool:
        if self._socket is not None:
            return True
        candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        candidate.settimeout(min(self._timeout_sec, 2.0))
        try:
            candidate.connect((self._host, self._port))
        except OSError:
            candidate.close()
            return False
        candidate.settimeout(self._timeout_sec)
        self._socket = candidate
        return True

    def infer(
        self,
        state_history: np.ndarray,
        image_history: dict[str, np.ndarray],
        instruction: str,
    ) -> np.ndarray:
        if not self.ping():
            raise ConnectionError("Xiaomi policy server is unavailable")
        state_history = np.asarray(state_history, dtype=np.float32)
        state = np.zeros((1, state_history.shape[0], 60), dtype=np.float32)
        state[0, :, : state_history.shape[-1]] = state_history
        inputs = self._processor.apply_chat_template(
            self._messages(image_history, instruction),
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            do_resize=False,
            state=state,
            robot_type=self._robot_type,
        )
        request = dict(inputs)
        request["task_id"] = self._robot_type
        try:
            self._send(request)
            encoded_actions = self._receive()
        except Exception:
            self.close()
            raise
        actions = self._processor.decode_action(
            encoded_actions, robot_type=self._robot_type
        )
        if isinstance(actions, self._torch.Tensor):
            actions = actions[0, :, :12].float().cpu().numpy()
        return np.asarray(actions, dtype=np.float32)

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def _messages(
        self, images: dict[str, np.ndarray], instruction: str
    ) -> list[dict[str, Any]]:
        keys = ("camera1", "camera2", "camera3")
        videos = {
            key: [_center_crop(frame, self._crop_ratio) for frame in images[key]]
            for key in keys
        }
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Left camera: "},
                    {"type": "video", "video": videos[keys[0]]},
                    {"type": "text", "text": "\nRight camera: "},
                    {"type": "video", "video": videos[keys[1]]},
                    {"type": "text", "text": "\nWrist camera: "},
                    {"type": "video", "video": videos[keys[2]]},
                    {
                        "type": "text",
                        "text": (
                            "\n\nGenerate robot actions for the task:\n"
                            f"{instruction} /no_cot"
                        ),
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<cot></cot>"}],
            },
        ]

    def _send(self, value: Any) -> None:
        encoded = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        assert self._socket is not None
        self._socket.sendall(struct.pack(">I", len(encoded)) + encoded)

    def _receive(self) -> Any:
        assert self._socket is not None
        length = struct.unpack(">I", _receive_exact(self._socket, 4))[0]
        return pickle.loads(_receive_exact(self._socket, length))  # noqa: S301


def _xiaomi_state(payload: dict[str, Any]) -> np.ndarray:
    state = np.asarray(payload.get("proprioception", []), dtype=np.float32)
    if state.shape != (16,) or not np.isfinite(state).all():
        raise PolicyExecutionError(
            "observation_schema_mismatch",
            f"Xiaomi RoboCasa365 state must have shape (16,), got {state.shape}",
        )
    converted = np.concatenate(
        (
            state[0:3],
            _quat_xyzw_to_axis_angle(state[3:7]),
            state[14:16],
            state[7:10],
            _quat_xyzw_to_axis_angle(state[10:14]),
        )
    ).astype(np.float32)
    if converted.shape != (14,):
        raise PolicyExecutionError(
            "observation_schema_mismatch",
            f"Xiaomi converted state must have shape (14,), got {converted.shape}",
        )
    return cast(np.ndarray, converted)


def _quat_xyzw_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    quaternion = quaternion / norm
    if quaternion[3] < 0:
        quaternion = -quaternion
    xyz = quaternion[:3]
    sin_half = np.linalg.norm(xyz)
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(sin_half, np.clip(quaternion[3], -1.0, 1.0))
    return cast(np.ndarray, (xyz / sin_half * angle).astype(np.float32))


def _camera_names(settings: dict[str, Any]) -> tuple[str, ...]:
    names = tuple(
        str(value)
        for value in settings.get("camera_names", ("camera1", "camera2", "camera3"))
    )
    if len(names) != 3:
        raise PolicyExecutionError(
            "observation_schema_mismatch", "Xiaomi requires exactly three cameras"
        )
    return names


def _video_frame(
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
            f"Xiaomi requires cameras {list(camera_names)}, got {sorted(images)}",
        )
    return images


def _sample_history(
    states: deque[np.ndarray],
    images: deque[dict[str, np.ndarray]],
    delta_indices: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not states or not images or len(states) != len(images):
        raise PolicyExecutionError(
            "observation_unavailable", "Xiaomi observation history is empty"
        )
    state_items = list(states)
    image_items = list(images)
    selected_states = [
        state_items[max(0, len(state_items) - 1 + delta)] for delta in delta_indices
    ]
    selected_images = [
        image_items[max(0, len(image_items) - 1 + delta)] for delta in delta_indices
    ]
    camera_names = tuple(selected_images[0])
    return (
        np.ascontiguousarray(np.stack(selected_states, axis=0)),
        {
            camera: np.ascontiguousarray(
                np.stack([frame[camera] for frame in selected_images], axis=0)
            )
            for camera in camera_names
        },
    )


def _action_chunk(
    response: Any, *, dimensions: int, execution_horizon: int
) -> list[np.ndarray]:
    values = np.asarray(response, dtype=np.float32)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2 or values.shape[1] < dimensions:
        raise PolicyExecutionError(
            "action_schema_mismatch",
            f"Xiaomi action must have shape (T,{dimensions}+), got {values.shape}",
        )
    if values.shape[0] < execution_horizon:
        raise PolicyExecutionError(
            "action_schema_mismatch",
            f"Xiaomi returned {values.shape[0]} actions, requires {execution_horizon}",
        )
    values = values[:execution_horizon, :dimensions]
    if not np.isfinite(values).all():
        raise PolicyExecutionError(
            "action_schema_mismatch", "Xiaomi action contains non-finite values"
        )
    return [row.astype(np.float32, copy=False) for row in values]


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


def _center_crop(image: np.ndarray, crop_ratio: float) -> Image.Image:
    source = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if crop_ratio >= 1:
        return source
    width, height = source.size
    crop_width = max(1, int(width * crop_ratio))
    crop_height = max(1, int(height * crop_ratio))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = source.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height), Image.Resampling.BILINEAR)


def _receive_exact(connection: socket.socket, length: int) -> bytes:
    output = bytearray()
    while len(output) < length:
        packet = connection.recv(length - len(output))
        if not packet:
            raise ConnectionError("Xiaomi policy server closed the connection")
        output.extend(packet)
    return bytes(output)


def _cuda_visible_device(device: str) -> str | None:
    if device == "cuda":
        return None
    if device.startswith("cuda:") and device.removeprefix("cuda:").isdigit():
        return device.removeprefix("cuda:")
    if device == "cpu":
        raise ValueError("Xiaomi-Robotics-1 official server requires CUDA")
    raise ValueError(f"unsupported Xiaomi policy_device {device!r}")


def _clip_action(action: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    low = np.asarray(settings.get("action_low", -1.0), dtype=np.float32)
    high = np.asarray(settings.get("action_high", 1.0), dtype=np.float32)
    return np.clip(action, low, high).astype(np.float32, copy=False)


def _cancelled_result() -> dict[str, Any]:
    return {
        "success": False,
        "status": "cancelled",
        "summary": "Xiaomi inference cancelled",
        "failure_mode": "cancelled",
    }
