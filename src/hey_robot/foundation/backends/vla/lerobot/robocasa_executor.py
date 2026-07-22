from __future__ import annotations

import base64
import hashlib
import io
import multiprocessing
import os
import random
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from threading import Event, Lock
from typing import Any

import numpy as np
from PIL import Image

from hey_robot.foundation.backends.vla.lerobot.robocasa_policy_probe import (
    _load_raw_config,
    offline_processor_overrides,
    register_policy_processors,
)
from hey_robot.robot_runtime.robocasa_remote.contract import (
    CAMERA_RENAME_MAP,
)


class OptionExecutionError(RuntimeError):
    def __init__(self, failure_mode: str, message: str) -> None:
        super().__init__(message)
        self.failure_mode = failure_mode


@dataclass
class _PolicyBundle:
    policy_path: str
    policy_type: str
    device: str
    input_features: dict[str, tuple[int, ...]]
    policy: Any
    preprocessor: Any
    postprocessor: Any

    def reset_action_queue(self, seed: int | None = None) -> None:
        """Reset policy episode state, optionally seeding inference RNGs."""
        if seed is not None:
            import torch

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

    def select_action(
        self, observation: dict[str, Any], option_command: str
    ) -> np.ndarray:
        sample = _policy_observation(
            observation,
            option_command,
            input_features=self.input_features,
        )
        processed = self.preprocessor(sample)
        action = self.policy.select_action(processed)
        action = self.postprocessor(action)
        action_array = _action_to_numpy(action)
        if action_array.shape != (12,):
            raise OptionExecutionError(
                "action_schema_mismatch",
                f"RoboCasa policy action must be shape (12,), got {action_array.shape}",
            )
        if not np.isfinite(action_array).all():
            raise OptionExecutionError(
                "action_schema_mismatch", "policy action is non-finite"
            )
        return action_array.astype(np.float32, copy=False)


@dataclass
class _IsolatedPolicyBundle:
    """Proxy that keeps CUDA policy inference out of the EGL simulator process."""

    policy_path: str
    policy_type: str
    device: str
    process: Any
    connection: Any
    request_timeout_sec: float = 300.0

    def _request(self, payload: tuple[Any, ...]) -> Any:
        if not self.process.is_alive():
            raise OptionExecutionError(
                "policy_process_failed", "isolated PI0.5 process is not running"
            )
        self.connection.send(payload)
        if not self.connection.poll(self.request_timeout_sec):
            raise OptionExecutionError(
                "policy_timeout", "isolated PI0.5 process did not respond in time"
            )
        response = self.connection.recv()
        if not response.get("ok", False):
            raise OptionExecutionError(
                "policy_inference_failed", str(response.get("error", "unknown error"))
            )
        return response.get("result")

    def reset_action_queue(self, seed: int | None = None) -> None:
        self._request(("reset", seed))

    def select_action(
        self, observation: dict[str, Any], option_command: str
    ) -> np.ndarray:
        result = self._request(("select_action", observation, option_command))
        return np.asarray(result, dtype=np.float32)


class RoboCasaLeRobotPolicyExecutor:
    """Pure RoboCasa policy inference behind the standard ModelService contract.

    This object deliberately has no simulator or ``EpisodeManager`` reference.
    One request contains one observation and produces one native 12-D action;
    Robot Runtime remains the only component allowed to advance the environment.
    CUDA inference is isolated from MuJoCo EGL in a spawned process.
    """

    def __init__(
        self,
        *,
        environ: dict[str, str] | None = None,
        policy_loader: Any | None = None,
    ) -> None:
        self.environ = dict(environ or os.environ)
        self.default_policy = str(self.environ.get("ROBOCASA_POLICY") or "").strip()
        if not self.default_policy:
            raise ValueError("RoboCasa policy_path must come from deployment config")
        self.default_device = self.environ.get("ROBOCASA_POLICY_DEVICE", "cuda")
        self.option_horizon = int(self.environ.get("ROBOCASA_OPTION_HORIZON", "50"))
        self.prompt_mode = self.environ.get("ROBOCASA_PROMPT_MODE", "environment_root")
        if self.prompt_mode not in {"environment_root", "agent_subgoal"}:
            raise ValueError(
                "ROBOCASA_PROMPT_MODE must be environment_root or agent_subgoal"
            )
        self._policy_loader = policy_loader or self._load_isolated_policy
        self._policies: dict[tuple[str, str], Any] = {}
        self._lock = Lock()
        self._cancel_event = Event()
        self._session_id: str | None = None
        self._policy_task: str | None = None
        self._last_error: str | None = None

    @property
    def busy(self) -> bool:
        return False

    def health(self) -> dict[str, Any]:
        imports_ok, import_error = self._imports_available()
        assets_ok = self._assets_available()
        loaded = imports_ok and assets_ok
        return {
            "online": True,
            "loaded": loaded,
            "error": import_error
            or (None if loaded else "RoboCasa option runner is waiting for assets"),
            "metrics": {
                "benchmark": "robocasa365",
                "mode": "single_action_policy",
                "asset_profile": "lightwheel",
                "policy_path": self.default_policy,
                "prompt_mode": self.prompt_mode,
                "option_horizon": self.option_horizon,
                "active_session": self._session_id,
                "imports_available": imports_ok,
                "assets_available": assets_ok,
                "versions": _versions(),
            },
        }

    def cancel(self) -> None:
        self._cancel_event.set()

    def close(self) -> None:
        """Release isolated policy processes during managed backend shutdown."""
        self.cancel()
        with self._lock:
            bundles = list(self._policies.values())
            self._policies.clear()
        for bundle in bundles:
            if not isinstance(bundle, _IsolatedPolicyBundle):
                continue
            with suppress(Exception):
                if bundle.process.is_alive():
                    bundle.connection.send(("close",))
                    if bundle.connection.poll(5.0):
                        bundle.connection.recv()
                    bundle.process.join(timeout=5.0)
                if bundle.process.is_alive():
                    bundle.process.terminate()
                    bundle.process.join(timeout=5.0)
            with suppress(Exception):
                bundle.connection.close()

    def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        started_at = time.monotonic()
        try:
            if str(payload.get("skill_name") or "") != "manipulate":
                raise OptionExecutionError(
                    "invalid_task", "RoboCasa policy only serves manipulate"
                )
            arguments = dict(payload.get("arguments", {}) or {})
            observation_payload = dict(arguments.get("observation") or {})
            observation = _observation_from_payload(observation_payload)
            observation_raw = dict(observation_payload.get("raw") or {})
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
            policy_task = (
                environment_root
                if self.prompt_mode == "environment_root"
                else agent_subgoal
            )
            if not policy_task:
                raise OptionExecutionError("invalid_task", "policy task is required")
            session_id = str(
                arguments.get("policy_session_id")
                or payload.get("episode_id")
                or "default"
            )
            seed = int(arguments.get("seed") or 0)
            bundle = self._policy_bundle(self.default_policy, self.default_device)
            if session_id != self._session_id or policy_task != self._policy_task:
                bundle.reset_action_queue(seed=seed)
                self._session_id = session_id
                self._policy_task = policy_task
            if self._cancel_event.is_set():
                self._cancel_event.clear()
                return {
                    "success": False,
                    "status": "cancelled",
                    "summary": "RoboCasa inference cancelled",
                    "failure_mode": "cancelled",
                }
            raw_action = bundle.select_action(observation, policy_task)
            action = np.clip(raw_action, -1.0, 1.0).astype(np.float32, copy=False)
            clipped = not np.array_equal(action, raw_action)
            return {
                "success": True,
                "status": "completed",
                "summary": "RoboCasa policy produced one native action",
                "metrics": {
                    "benchmark": "robocasa365",
                    "mode": "single_action_policy",
                    "policy_result": {
                        "kind": "native_action",
                        "values": [float(value) for value in action.tolist()],
                        "raw_values": [float(value) for value in raw_action.tolist()],
                        "expected_frame_id": int(
                            dict(arguments.get("observation") or {}).get("frame_id", 0)
                        ),
                    },
                    "policy_type": bundle.policy_type,
                    "prompt_mode": self.prompt_mode,
                    "option_horizon": self.option_horizon,
                    "effective_policy_prompt_sha256": hashlib.sha256(
                        policy_task.encode("utf-8")
                    ).hexdigest(),
                    "action_clipped": clipped,
                    "duration_sec": round(time.monotonic() - started_at, 3),
                },
            }
        except OptionExecutionError as exc:
            self._last_error = str(exc)
            return {
                "success": False,
                "status": "failed",
                "summary": f"RoboCasa inference rejected: {exc}",
                "failure_mode": exc.failure_mode,
                "error": str(exc),
            }
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            return {
                "success": False,
                "status": "failed",
                "summary": f"RoboCasa inference failed: {type(exc).__name__}: {exc}",
                "failure_mode": "execution_failed",
                "error": str(exc),
            }

    def _policy_bundle(self, policy_path: str, device: str) -> Any:
        key = (policy_path, device)
        with self._lock:
            bundle = self._policies.get(key)
        if bundle is not None:
            return bundle
        try:
            bundle = self._policy_loader(policy_path, device)
        except Exception as exc:
            raise OptionExecutionError(
                "policy_load_failed",
                f"failed to load policy {policy_path!r}: {type(exc).__name__}: {exc}",
            ) from exc
        with self._lock:
            self._policies[key] = bundle
        return bundle

    def _load_isolated_policy(
        self, policy_path: str, device: str
    ) -> _IsolatedPolicyBundle:
        raw_config, _ = _load_raw_config(policy_path)
        policy_type = str(raw_config.get("type") or "")
        if not policy_type:
            raise ValueError("policy config has no type")
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe()
        process = context.Process(
            target=_policy_process_main,
            args=(child_connection, policy_path, device),
            name="robocasa365-pi05",
            daemon=True,
        )
        process.start()
        child_connection.close()
        load_timeout = float(self.environ.get("ROBOCASA_POLICY_LOAD_TIMEOUT", "600"))
        if not parent_connection.poll(load_timeout):
            process.terminate()
            process.join(timeout=5)
            raise TimeoutError("isolated PI0.5 process did not load in time")
        response = parent_connection.recv()
        if not response.get("ok", False):
            process.join(timeout=5)
            raise RuntimeError(str(response.get("error", "policy process failed")))
        return _IsolatedPolicyBundle(
            policy_path=policy_path,
            policy_type=policy_type,
            device=device,
            process=process,
            connection=parent_connection,
            request_timeout_sec=float(
                self.environ.get("ROBOCASA_POLICY_REQUEST_TIMEOUT", "300")
            ),
        )

    def _imports_available(self) -> tuple[bool, str | None]:
        try:
            __import__("robocasa")
            __import__("robosuite")
            __import__("mujoco")
            __import__("lerobot")
        except Exception as exc:
            return False, f"dependency unavailable: {type(exc).__name__}: {exc}"
        return True, None

    def _assets_available(self) -> bool:
        explicit_root = Path(
            self.environ.get(
                "ROBOCASA_MODEL_ASSET_ROOT",
                "/opt/robocasa/robocasa/models/assets",
            )
        )
        roots = [explicit_root]
        try:
            import robocasa

            roots.append(Path(robocasa.__file__).resolve().parent / "models" / "assets")
        except Exception as exc:
            self._last_error = f"robocasa asset root unavailable: {exc}"
        marker_override = self.environ.get("ROBOCASA_ASSET_READY_FILE")
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


def _load_policy_bundle(policy_path: str, device: str) -> _PolicyBundle:
    if Path(policy_path).is_dir() or os.environ.get("ROBOCASA_OFFLINE") == "1":
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
    preprocessor, postprocessor = _make_runtime_processors(
        config=config,
        policy_path=policy_path,
        policy_type=policy_type,
        make_pre_post_processors=make_pre_post_processors,
    )
    return _PolicyBundle(
        policy_path=policy_path,
        policy_type=policy_type,
        device=device,
        input_features=_feature_shapes(raw_config.get("input_features")),
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )


def _policy_process_main(connection: Any, policy_path: str, device: str) -> None:
    """Own all Torch/CUDA state and serve the small synchronous policy API."""
    try:
        bundle = _load_policy_bundle(policy_path, device)
        connection.send({"ok": True})
        while True:
            command = connection.recv()
            if command[0] == "reset":
                seed = command[1] if len(command) > 1 else None
                bundle.reset_action_queue(seed=seed)
                connection.send({"ok": True})
            elif command[0] == "select_action":
                action = bundle.select_action(command[1], command[2])
                connection.send({"ok": True, "result": action})
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


def _observation_from_payload(value: Any) -> dict[str, Any]:
    payload = dict(value or {})
    state = np.asarray(payload.get("proprioception", []), dtype=np.float32)
    if state.shape != (16,) or not np.isfinite(state).all():
        raise OptionExecutionError(
            "observation_schema_mismatch",
            f"RoboCasa proprioception must have shape (16,), got {state.shape}",
        )
    pixels: dict[str, np.ndarray] = {}
    for item in list(payload.get("images", []) or []):
        image = dict(item or {})
        camera = str(image.get("camera") or "")
        encoded = str(image.get("data") or "")
        if camera not in {"camera1", "camera2", "camera3"} or not encoded:
            continue
        try:
            raw = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(raw)) as source:
                pixels[camera] = np.asarray(source.convert("RGB"), dtype=np.uint8)
        except Exception as exc:
            raise OptionExecutionError(
                "observation_schema_mismatch", f"invalid {camera} image: {exc}"
            ) from exc
    expected = {"camera1", "camera2", "camera3"}
    if set(pixels) != expected:
        raise OptionExecutionError(
            "observation_schema_mismatch",
            f"RoboCasa observation requires {sorted(expected)}, got {sorted(pixels)}",
        )
    return {"agent_pos": state, "pixels": pixels}


def _policy_observation(
    observation: dict[str, Any],
    option_command: str,
    *,
    input_features: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
    try:
        from lerobot.envs import preprocess_observation
    except ModuleNotFoundError:
        sample = _fallback_preprocess_observation(observation)
    else:
        sample = preprocess_observation(observation)
    # Runtime exposes stable camera1/2/3 names. The checkpoint retains the
    # native RoboCasa feature keys, so translate only at the policy boundary.
    for native_key, runtime_key in CAMERA_RENAME_MAP.items():
        if native_key in input_features and runtime_key in sample:
            sample[native_key] = sample.pop(runtime_key)
    # Match lerobot_eval: language-conditioned policies receive a batch-sized
    # list, not a bare string.
    sample["task"] = [option_command]
    sample["robot_type"] = "robocasa"
    return sample


def _fallback_preprocess_observation(observation: dict[str, Any]) -> dict[str, Any]:
    import torch

    sample: dict[str, Any] = {}
    for camera, frame in dict(observation.get("pixels", {}) or {}).items():
        tensor = torch.from_numpy(np.array(frame, dtype=np.uint8, copy=True))
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.permute(0, 3, 1, 2).contiguous().float() / 255.0
        sample[f"observation.images.{camera}"] = tensor
    state = torch.from_numpy(
        np.asarray(observation.get("agent_pos", []), dtype=np.float32)
    )
    if state.ndim == 1:
        state = state.unsqueeze(0)
    sample["observation.state"] = state
    return sample


def _make_runtime_processors(
    *,
    config: Any,
    policy_path: str,
    policy_type: str,
    make_pre_post_processors: Callable[..., tuple[Any, Any]],
) -> tuple[Any, Any]:
    try:
        return make_pre_post_processors(
            config,
            pretrained_path=policy_path,
            **offline_processor_overrides(policy_type),
        )
    except Exception:
        if policy_type != "pi052":
            raise
        if not getattr(config, "enable_fast_action_loss", False):
            raise

    config.enable_fast_action_loss = False
    return make_pre_post_processors(config)


def _feature_shapes(features: dict[str, Any] | None) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for key, feature in (features or {}).items():
        shape = feature.get("shape") if isinstance(feature, dict) else None
        if isinstance(shape, list | tuple):
            shapes[str(key)] = tuple(int(value) for value in shape)
    return shapes


def _validate_observation(observation: dict[str, Any]) -> None:
    state = np.asarray(observation.get("agent_pos", []))
    if state.shape != (16,) or not np.isfinite(state).all():
        raise OptionExecutionError(
            "observation_schema_mismatch",
            f"RoboCasa observation state must be 16 finite values, got {state.shape}",
        )
    pixels = dict(observation.get("pixels", {}) or {})
    if len(pixels) < 3:
        raise OptionExecutionError(
            "observation_schema_mismatch",
            f"RoboCasa observation must contain 3 camera images, got {len(pixels)}",
        )


def _action_to_numpy(action: Any) -> np.ndarray:
    if isinstance(action, dict):
        action = action.get("action", action.get("actions", action))
    if hasattr(action, "detach"):
        action = action.detach()
    if hasattr(action, "to"):
        action = action.to("cpu")
    if hasattr(action, "numpy"):
        action = action.numpy()
    array = np.asarray(action, dtype=np.float32)
    array = np.squeeze(array)
    if array.ndim != 1:
        raise OptionExecutionError(
            "action_schema_mismatch", f"policy action must be rank 1, got {array.shape}"
        )
    return array


def _versions() -> dict[str, str | None]:
    return {
        "python": sys.version.split()[0],
        "lerobot": _distribution_version("lerobot"),
        "robocasa": _distribution_version("robocasa"),
        "robosuite": _distribution_version("robosuite"),
        "mujoco": _distribution_version("mujoco"),
    }


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
