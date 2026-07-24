"""Configuration-driven LeRobot policy inference.

This module owns policy loading and inference only. Robot and environment
control remain behind Robot Runtime.
"""

from __future__ import annotations

import base64
import io
import multiprocessing
import os
import random
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from threading import Event, Lock
from typing import Any, Protocol

import numpy as np
from PIL import Image

from hey_robot.config import ModelServiceSpec
from hey_robot.foundation.backends.lerobot.checkpoint import (
    _load_raw_config,
    offline_processor_overrides,
    register_policy_processors,
)
from hey_robot.foundation.clients.models import PolicyStepResult


class PolicyExecutionError(RuntimeError):
    def __init__(self, failure_mode: str, message: str) -> None:
        super().__init__(message)
        self.failure_mode = failure_mode


class PolicyRuntime(Protocol):
    policy_type: str

    def reset(self, seed: int | None = None) -> None: ...

    def select_action(self, observation: dict[str, Any], task: str) -> np.ndarray: ...

    def cancel(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class _PolicyRequest:
    session_id: str
    task: str
    observation: dict[str, Any]
    frame_id: int
    seed: int


class LeRobotPolicyExecutor:
    """Run any configured LeRobot policy behind the ModelService contract.

    The checkpoint's ``type`` selects the concrete policy through LeRobot's
    factory. Hey Robot deliberately does not branch on VLA, WAM, or classic
    learned-policy families.
    """

    def __init__(
        self,
        service_id: str,
        spec: ModelServiceSpec,
        *,
        runtime_loader: Any | None = None,
    ) -> None:
        self.service_id = service_id
        self.spec = spec
        self.settings = dict(spec.settings)
        self.policy_path = str(self.settings.get("policy_path") or "").strip()
        if not self.policy_path:
            raise ValueError("LeRobot policy_path is required")
        self.device = str(self.settings.get("policy_device") or "cuda")
        self.embodiment = str(self.settings.get("embodiment") or spec.robot_id)
        self.action_space = str(
            self.settings.get("action_space") or "embodiment_native"
        )
        self.action_dimensions = int(self.settings.get("action_dimensions") or 0)
        if self.action_dimensions <= 0:
            raise ValueError("LeRobot action_dimensions must be positive")
        self.prompt_mode = str(self.settings.get("prompt_mode") or "agent_subgoal")
        if self.prompt_mode not in {"environment_root", "agent_subgoal"}:
            raise ValueError(
                "LeRobot prompt_mode must be environment_root or agent_subgoal"
            )
        self._runtime_loader = runtime_loader or _load_policy_runtime
        self._runtime: PolicyRuntime | None = None
        self._runtime_lock = Lock()
        self._cancel_event = Event()
        self._session_id: str | None = None
        self._policy_task: str | None = None
        self._last_error: str | None = None

    def health(self) -> dict[str, Any]:
        dependency_error = _dependency_error()
        return {
            "name": self.service_id,
            "robot_id": self.spec.robot_id,
            "online": True,
            "loaded": self._runtime is not None,
            "error": dependency_error or self._last_error,
            "metrics": {
                "type": self.spec.type,
                "runtime": "lerobot",
                "policy_path": self.policy_path,
                "policy_type": getattr(self._runtime, "policy_type", None),
                "device": self.device,
                "embodiment": self.embodiment,
                "action_space": self.action_space,
                "action_dimensions": self.action_dimensions,
                "prompt_mode": self.prompt_mode,
                "active_session": self._session_id,
                "hardware_ownership": "none",
            },
        }

    def load(self) -> None:
        """Load and validate the configured policy before accepting traffic."""
        self._get_runtime()

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
            runtime = self._get_runtime()
            if (
                request.session_id != self._session_id
                or request.task != self._policy_task
            ):
                runtime.reset(seed=request.seed)
                self._session_id = request.session_id
                self._policy_task = request.task
            if self._cancel_event.is_set():
                self._cancel_event.clear()
                return _cancelled_result()

            raw_action = runtime.select_action(request.observation, request.task)
            raw_action = _validate_action(raw_action, self.action_dimensions)
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
                    "policy_type": runtime.policy_type,
                    "expected_frame_id": request.frame_id,
                    "action_clipped": clipped,
                },
            ).to_metrics()
            self._last_error = None
            return {
                "success": True,
                "status": "completed",
                "summary": "LeRobot policy produced one native action",
                "metrics": {
                    "policy_result": policy_result,
                    "action_chunk": policy_result,
                    "policy_type": runtime.policy_type,
                    "prompt_mode": self.prompt_mode,
                    "action_clipped": clipped,
                    "duration_sec": round(time.monotonic() - started_at, 3),
                },
            }
        except PolicyExecutionError as exc:
            self._last_error = str(exc)
            return {
                "success": False,
                "status": "failed",
                "summary": f"LeRobot inference rejected: {exc}",
                "failure_mode": exc.failure_mode,
                "error": str(exc),
            }
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            return {
                "success": False,
                "status": "failed",
                "summary": f"LeRobot inference failed: {type(exc).__name__}: {exc}",
                "failure_mode": "execution_failed",
                "error": str(exc),
            }

    def cancel(self) -> None:
        self._cancel_event.set()
        with self._runtime_lock:
            runtime = self._runtime
            self._runtime = None
            self._session_id = None
            self._policy_task = None
        if runtime is not None:
            with suppress(Exception):
                runtime.cancel()
            with suppress(Exception):
                runtime.close()

    def close(self) -> None:
        self._cancel_event.set()
        with self._runtime_lock:
            runtime = self._runtime
            self._runtime = None
            self._session_id = None
            self._policy_task = None
        if runtime is not None:
            with suppress(Exception):
                runtime.close()

    def _request(self, payload: dict[str, Any]) -> _PolicyRequest:
        arguments = dict(payload.get("arguments", {}) or {})
        observation = dict(arguments.get("observation") or {})
        if not observation:
            raise PolicyExecutionError(
                "observation_unavailable", "LeRobot policy requires an observation"
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
        return _PolicyRequest(
            session_id=str(
                arguments.get("policy_session_id")
                or payload.get("episode_id")
                or "default"
            ),
            task=task,
            observation=observation,
            frame_id=int(observation.get("frame_id") or 0),
            seed=int(arguments.get("seed") or 0),
        )

    def _get_runtime(self) -> PolicyRuntime:
        with self._runtime_lock:
            if self._runtime is not None:
                return self._runtime
            try:
                self._runtime = self._runtime_loader(
                    self.policy_path,
                    self.device,
                    self.settings,
                )
            except Exception as exc:
                raise PolicyExecutionError(
                    "policy_load_failed",
                    f"failed to load policy {self.policy_path!r}: "
                    f"{type(exc).__name__}: {exc}",
                ) from exc
            return self._runtime


@dataclass
class _DirectPolicyRuntime:
    policy_path: str
    policy_type: str
    device: str
    settings: dict[str, Any]
    input_features: dict[str, tuple[int, ...]]
    policy: Any
    preprocessor: Any
    postprocessor: Any

    def reset(self, seed: int | None = None) -> None:
        _seed_policy(seed)
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

    def select_action(self, observation: dict[str, Any], task: str) -> np.ndarray:
        sample = _policy_observation(
            observation,
            task,
            settings=self.settings,
            input_features=self.input_features,
        )
        processed = self.preprocessor(sample)
        action = self.policy.select_action(processed)
        return _action_to_numpy(self.postprocessor(action))

    def cancel(self) -> None:
        return None

    def close(self) -> None:
        return None


@dataclass
class _IsolatedPolicyRuntime:
    policy_path: str
    policy_type: str
    device: str
    process: Any
    connection: Any
    request_timeout_sec: float

    def _request(self, payload: tuple[Any, ...]) -> Any:
        if not self.process.is_alive():
            raise PolicyExecutionError(
                "policy_process_failed", "isolated LeRobot process is not running"
            )
        self.connection.send(payload)
        if not self.connection.poll(self.request_timeout_sec):
            raise PolicyExecutionError(
                "policy_timeout", "isolated LeRobot process did not respond in time"
            )
        response = self.connection.recv()
        if not response.get("ok", False):
            raise PolicyExecutionError(
                "policy_inference_failed", str(response.get("error") or "unknown")
            )
        return response.get("result")

    def reset(self, seed: int | None = None) -> None:
        self._request(("reset", seed))

    def select_action(self, observation: dict[str, Any], task: str) -> np.ndarray:
        return np.asarray(
            self._request(("select_action", observation, task)), dtype=np.float32
        )

    def cancel(self) -> None:
        self._terminate()

    def close(self) -> None:
        if self.process.is_alive():
            with suppress(Exception):
                self._request(("close",))
            self.process.join(timeout=5.0)
        self._terminate()
        with suppress(Exception):
            self.connection.close()

    def _terminate(self) -> None:
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5.0)


def _load_policy_runtime(
    policy_path: str, device: str, settings: dict[str, Any]
) -> PolicyRuntime:
    if bool(settings.get("isolate_policy", False)):
        return _load_isolated_policy_runtime(policy_path, device, settings)
    return _load_direct_policy_runtime(policy_path, device, settings)


def _load_direct_policy_runtime(
    policy_path: str, device: str, settings: dict[str, Any]
) -> _DirectPolicyRuntime:
    if Path(policy_path).is_dir() or bool(settings.get("offline", False)):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    raw_config, _ = _load_raw_config(policy_path)
    policy_type = str(raw_config.get("type") or "")
    if not policy_type:
        raise ValueError("policy config has no type")
    register_policy_processors(policy_type)
    config = PreTrainedConfig.from_pretrained(policy_path)
    config.device = device
    policy_class = get_policy_class(policy_type)
    policy = policy_class.from_pretrained(policy_path, config=config)
    policy.to(torch.device(device))
    policy.eval()
    try:
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            pretrained_path=policy_path,
            **offline_processor_overrides(policy_type),
        )
    except Exception:
        if policy_type != "pi052" or not getattr(
            config, "enable_fast_action_loss", False
        ):
            raise
        config.enable_fast_action_loss = False
        preprocessor, postprocessor = make_pre_post_processors(config)
    return _DirectPolicyRuntime(
        policy_path=policy_path,
        policy_type=policy_type,
        device=device,
        settings=dict(settings),
        input_features=_feature_shapes(raw_config.get("input_features")),
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )


def _load_isolated_policy_runtime(
    policy_path: str, device: str, settings: dict[str, Any]
) -> _IsolatedPolicyRuntime:
    raw_config, _ = _load_raw_config(policy_path)
    policy_type = str(raw_config.get("type") or "")
    if not policy_type:
        raise ValueError("policy config has no type")
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe()
    process = context.Process(
        target=_policy_process_main,
        args=(child_connection, policy_path, device, dict(settings)),
        name="lerobot-policy",
        daemon=True,
    )
    process.start()
    child_connection.close()
    load_timeout = float(settings.get("load_timeout_sec") or 600.0)
    if not parent_connection.poll(load_timeout):
        process.terminate()
        process.join(timeout=5.0)
        raise TimeoutError("isolated LeRobot process did not load in time")
    response = parent_connection.recv()
    if not response.get("ok", False):
        process.join(timeout=5.0)
        raise RuntimeError(str(response.get("error") or "policy process failed"))
    return _IsolatedPolicyRuntime(
        policy_path=policy_path,
        policy_type=policy_type,
        device=device,
        process=process,
        connection=parent_connection,
        request_timeout_sec=float(settings.get("request_timeout_sec") or 300.0),
    )


def _policy_process_main(
    connection: Any,
    policy_path: str,
    device: str,
    settings: dict[str, Any],
) -> None:
    try:
        settings["isolate_policy"] = False
        runtime = _load_direct_policy_runtime(policy_path, device, settings)
        connection.send({"ok": True})
        while True:
            command = connection.recv()
            if command[0] == "reset":
                runtime.reset(command[1] if len(command) > 1 else None)
                connection.send({"ok": True})
            elif command[0] == "select_action":
                result = runtime.select_action(command[1], command[2])
                connection.send({"ok": True, "result": result})
            elif command[0] == "close":
                connection.send({"ok": True})
                return
            else:
                raise ValueError(f"unknown policy process command: {command[0]!r}")
    except EOFError:
        return
    except Exception as exc:
        with suppress(BrokenPipeError, EOFError, OSError):
            connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def _policy_observation(
    payload: dict[str, Any],
    task: str,
    *,
    settings: dict[str, Any],
    input_features: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
    state = np.asarray(payload.get("proprioception", []), dtype=np.float32)
    expected_state_dimensions = int(settings.get("state_dimensions") or state.size)
    if state.shape != (expected_state_dimensions,) or not np.isfinite(state).all():
        raise PolicyExecutionError(
            "observation_schema_mismatch",
            f"policy state must have shape ({expected_state_dimensions},), got {state.shape}",
        )
    camera_names = tuple(str(value) for value in settings.get("camera_names", ()))
    pixels: dict[str, np.ndarray] = {}
    for item in list(payload.get("images", []) or []):
        image = dict(item or {})
        camera = str(image.get("camera") or "")
        if camera_names and camera not in camera_names:
            continue
        if not camera:
            continue
        try:
            raw = _image_bytes(image, settings)
            with Image.open(io.BytesIO(raw)) as source:
                pixels[camera] = np.asarray(source.convert("RGB"), dtype=np.uint8)
        except Exception as exc:
            raise PolicyExecutionError(
                "observation_schema_mismatch", f"invalid {camera} image: {exc}"
            ) from exc
    if camera_names and set(pixels) != set(camera_names):
        raise PolicyExecutionError(
            "observation_schema_mismatch",
            f"policy requires cameras {list(camera_names)}, got {sorted(pixels)}",
        )

    environment_observation = {"agent_pos": state, "pixels": pixels}
    try:
        from lerobot.envs import preprocess_observation
    except ModuleNotFoundError:
        sample = _fallback_preprocess_observation(environment_observation)
    else:
        sample = preprocess_observation(environment_observation)

    feature_map = {
        str(policy_key): str(runtime_key)
        for policy_key, runtime_key in dict(
            settings.get("observation_features") or {}
        ).items()
    }
    for policy_key, runtime_key in feature_map.items():
        if runtime_key not in sample:
            raise PolicyExecutionError(
                "observation_schema_mismatch",
                f"observation feature source {runtime_key!r} is unavailable",
            )
        if policy_key != runtime_key:
            sample[policy_key] = sample.pop(runtime_key)
    missing_features = [
        key
        for key in input_features
        if key.startswith("observation.") and key not in sample
    ]
    if missing_features:
        raise PolicyExecutionError(
            "observation_schema_mismatch",
            f"policy observation features are missing: {', '.join(missing_features)}",
        )
    sample["task"] = [task]
    sample["robot_type"] = str(settings.get("robot_type") or "")
    return sample


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


def _fallback_preprocess_observation(
    observation: dict[str, Any],
) -> dict[str, Any]:
    import torch

    sample: dict[str, Any] = {}
    for camera, frame in dict(observation.get("pixels", {}) or {}).items():
        tensor = torch.from_numpy(np.array(frame, dtype=np.uint8, copy=True))
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        sample[f"observation.images.{camera}"] = (
            tensor.permute(0, 3, 1, 2).contiguous().float() / 255.0
        )
    state = torch.from_numpy(
        np.asarray(observation.get("agent_pos", []), dtype=np.float32)
    )
    sample["observation.state"] = state.unsqueeze(0) if state.ndim == 1 else state
    return sample


def _action_to_numpy(action: Any) -> np.ndarray:
    if isinstance(action, dict):
        action = action.get("action", action.get("actions", action))
    if hasattr(action, "detach"):
        action = action.detach()
    if hasattr(action, "to"):
        action = action.to("cpu")
    if hasattr(action, "numpy"):
        action = action.numpy()
    array = np.squeeze(np.asarray(action, dtype=np.float32))
    if array.ndim != 1:
        raise PolicyExecutionError(
            "action_schema_mismatch", f"policy action must be rank 1, got {array.shape}"
        )
    return array


def _validate_action(action: np.ndarray, dimensions: int) -> np.ndarray:
    if action.shape != (dimensions,):
        raise PolicyExecutionError(
            "action_schema_mismatch",
            f"policy action must have shape ({dimensions},), got {action.shape}",
        )
    if not np.isfinite(action).all():
        raise PolicyExecutionError(
            "action_schema_mismatch", "policy action contains non-finite values"
        )
    return action.astype(np.float32, copy=False)


def _clip_action(action: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    low = np.asarray(settings.get("action_low", -1.0), dtype=np.float32)
    high = np.asarray(settings.get("action_high", 1.0), dtype=np.float32)
    return np.clip(action, low, high).astype(np.float32, copy=False)


def _feature_shapes(features: dict[str, Any] | None) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for key, feature in (features or {}).items():
        shape = feature.get("shape") if isinstance(feature, dict) else None
        if isinstance(shape, list | tuple):
            shapes[str(key)] = tuple(int(value) for value in shape)
    return shapes


def _seed_policy(seed: int | None) -> None:
    if seed is None:
        return
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dependency_error() -> str | None:
    try:
        __import__("lerobot")
    except Exception as exc:
        return f"LeRobot dependency unavailable: {type(exc).__name__}: {exc}"
    return None


def _cancelled_result() -> dict[str, Any]:
    return {
        "success": False,
        "status": "cancelled",
        "summary": "LeRobot inference cancelled",
        "failure_mode": "cancelled",
    }
