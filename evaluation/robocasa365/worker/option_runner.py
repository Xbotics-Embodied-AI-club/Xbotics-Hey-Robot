from __future__ import annotations

import math
import os
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

try:
    from policy_probe import _load_raw_config, register_policy_processors
    from rollout import (
        ALLOWED_TASKS,
        DEFAULT_POLICY,
        DEFAULT_REGISTRIES,
        DEFAULT_SPLIT,
    )
except ModuleNotFoundError:
    from evaluation.robocasa365.worker.policy_probe import (
        _load_raw_config,
        register_policy_processors,
    )
    from evaluation.robocasa365.worker.rollout import (
        ALLOWED_TASKS,
        DEFAULT_POLICY,
        DEFAULT_REGISTRIES,
        DEFAULT_SPLIT,
    )


class OptionError(RuntimeError):
    def __init__(self, failure_mode: str, message: str) -> None:
        super().__init__(message)
        self.failure_mode = failure_mode


@dataclass(frozen=True)
class OptionRequest:
    skill_id: str
    objective: str
    task: str
    option_command: str
    seed: int = 1000
    max_steps: int = 30
    policy_path: str = DEFAULT_POLICY
    session_id: str | None = None
    reset_episode: bool = False
    close_episode: bool = False
    device: str = "cuda"


@dataclass(frozen=True)
class OptionResult:
    success: bool
    status: str
    summary: str
    failure_mode: str | None = None
    error: str | None = None
    metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value is not None}


@dataclass
class _PolicyBundle:
    policy_path: str
    policy_type: str
    device: str
    input_features: dict[str, tuple[int, ...]]
    policy: Any
    preprocessor: Any
    postprocessor: Any

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
            raise OptionError(
                "action_schema_mismatch",
                f"RoboCasa policy action must be shape (12,), got {action_array.shape}",
            )
        if not np.isfinite(action_array).all():
            raise OptionError("action_schema_mismatch", "policy action is non-finite")
        return action_array.astype(np.float32, copy=False)


@dataclass
class _Session:
    session_id: str
    task: str
    seed: int
    policy_path: str
    env: Any
    observation: dict[str, Any]
    frame_id: int = 0
    done: bool = False
    success: bool = False
    started_at_unix: float = 0.0


class RoboCasaOptionRunner:
    """Stateful, option-level RoboCasa executor for Hey Robot agent evaluation.

    Unlike ``RoboCasaRolloutRunner`` this runner keeps one simulator episode open
    across bounded option calls.  The VLA runs inside the worker process and
    emits the native 12-D RoboCasa action, so no real-robot primitive adapter is
    involved.
    """

    def __init__(
        self,
        *,
        environ: dict[str, str] | None = None,
        env_factory: Callable[[str, int], tuple[Any, dict[str, Any]]] | None = None,
        policy_loader: Callable[[str, str], _PolicyBundle] | None = None,
    ) -> None:
        self.environ = dict(environ or os.environ)
        self.default_policy = self.environ.get("ROBOCASA_POLICY", DEFAULT_POLICY)
        self.default_device = self.environ.get("ROBOCASA_POLICY_DEVICE", "cuda")
        self._env_factory = env_factory or self._create_episode
        self._policy_loader = policy_loader or self._load_policy
        self._sessions: dict[str, _Session] = {}
        self._policies: dict[tuple[str, str], _PolicyBundle] = {}
        self._lock = Lock()
        self._last_error: str | None = None

    @property
    def busy(self) -> bool:
        return False

    @property
    def active_sessions(self) -> int:
        with self._lock:
            return len(self._sessions)

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
                "mode": "embodied_agent_option",
                "asset_profile": "lightwheel",
                "policy_path": self.default_policy,
                "active_sessions": self.active_sessions,
                "imports_available": imports_ok,
                "assets_available": assets_ok,
                "versions": _versions(),
            },
        }

    def close_session(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        _close_env(session.env)
        return True

    def run(self, request: OptionRequest) -> OptionResult:
        started_at = time.monotonic()
        try:
            self._validate_request(request)
            bundle = self._policy_bundle(request.policy_path, request.device)
            session = self._session_for(request)
            trace: list[dict[str, Any]] = []
            last_reward = 0.0
            last_info: dict[str, Any] = {}

            for option_step in range(request.max_steps):
                action = bundle.select_action(
                    session.observation, request.option_command
                )
                observation, reward, terminated, truncated, info = session.env.step(
                    action
                )
                last_reward = float(reward)
                last_info = dict(info or {})
                session.frame_id += 1
                session.observation = observation
                session.done = bool(terminated or truncated)
                session.success = bool(last_info.get("is_success", False))
                trace.append(
                    {
                        "option_step": option_step,
                        "frame_id": session.frame_id,
                        "reward": last_reward,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "success": session.success,
                        "action": [float(value) for value in action.tolist()],
                    }
                )
                if session.success or session.done:
                    break

            status = "completed" if session.success else "failed"
            failure_mode = None if session.success else "option_timeout"
            metrics = {
                "benchmark": "robocasa365",
                "mode": "embodied_agent_option",
                "task": session.task,
                "split": DEFAULT_SPLIT,
                "seed": session.seed,
                "policy_path": request.policy_path,
                "policy_type": bundle.policy_type,
                "session_id": session.session_id,
                "option_command": request.option_command,
                "steps": len(trace),
                "frame_id": session.frame_id,
                "episode_done": session.done,
                "episode_success": session.success,
                "last_reward": last_reward,
                "last_info": _json_safe(last_info),
                "duration_sec": round(time.monotonic() - started_at, 3),
                "trace": trace,
            }
            summary = (
                f"RoboCasa option {session.task}: success after {len(trace)} steps"
                if session.success
                else f"RoboCasa option {session.task}: no success in {len(trace)} steps"
            )
            if request.close_episode or session.done:
                self.close_session(session.session_id)
            return OptionResult(
                success=session.success,
                status=status,
                summary=summary,
                failure_mode=failure_mode,
                metrics=metrics,
            )
        except OptionError as exc:
            self._last_error = str(exc)
            if exc.failure_mode != "option_timeout":
                self.close_session(request.session_id or request.skill_id)
            return OptionResult(
                success=False,
                status="failed",
                summary=f"RoboCasa option did not run: {exc}",
                failure_mode=exc.failure_mode,
                error=str(exc),
                metrics={"duration_sec": round(time.monotonic() - started_at, 3)},
            )
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            self.close_session(request.session_id or request.skill_id)
            return OptionResult(
                success=False,
                status="failed",
                summary=f"RoboCasa option failed: {type(exc).__name__}: {exc}",
                failure_mode="execution_failed",
                error=str(exc),
                metrics={"duration_sec": round(time.monotonic() - started_at, 3)},
            )

    def _validate_request(self, request: OptionRequest) -> None:
        if request.task not in ALLOWED_TASKS:
            raise OptionError("invalid_task", f"task {request.task!r} is not allowed")
        if not request.option_command.strip():
            raise OptionError("invalid_task", "option_command is required")
        if not request.policy_path.strip():
            raise OptionError("checkpoint_unavailable", "policy_path is required")
        if request.max_steps < 1 or request.max_steps > 2000:
            raise OptionError("invalid_task", "max_steps must be in [1, 2000]")
        if not request.skill_id:
            raise OptionError("invalid_task", "skill_id is required")

    def _policy_bundle(self, policy_path: str, device: str) -> _PolicyBundle:
        key = (policy_path, device)
        with self._lock:
            bundle = self._policies.get(key)
        if bundle is not None:
            return bundle
        try:
            bundle = self._policy_loader(policy_path, device)
        except Exception as exc:
            raise OptionError(
                "policy_load_failed",
                f"failed to load policy {policy_path!r}: {type(exc).__name__}: {exc}",
            ) from exc
        with self._lock:
            self._policies[key] = bundle
        return bundle

    def _session_for(self, request: OptionRequest) -> _Session:
        session_id = request.session_id or request.skill_id or f"rc-{uuid.uuid4().hex}"
        with self._lock:
            existing = self._sessions.get(session_id)
        if existing is not None and not request.reset_episode:
            if existing.done:
                raise OptionError(
                    "episode_already_done",
                    "episode is done; pass reset_episode=true to start again",
                )
            if existing.task != request.task:
                raise OptionError(
                    "session_task_mismatch",
                    f"session task is {existing.task!r}, got {request.task!r}",
                )
            return existing

        if existing is not None:
            self.close_session(session_id)
        try:
            env, observation = self._env_factory(request.task, request.seed)
        except Exception as exc:
            raise OptionError(
                "environment_reset_failed",
                f"failed to create RoboCasa episode: {type(exc).__name__}: {exc}",
            ) from exc
        _validate_observation(observation)
        session = _Session(
            session_id=session_id,
            task=request.task,
            seed=request.seed,
            policy_path=request.policy_path,
            env=env,
            observation=observation,
            started_at_unix=time.time(),
        )
        with self._lock:
            self._sessions[session_id] = session
        return session

    def _create_episode(self, task: str, seed: int) -> tuple[Any, dict[str, Any]]:
        from lerobot.envs.robocasa import DEFAULT_CAMERAS, RoboCasaEnv

        env = RoboCasaEnv(
            task=task,
            camera_name=DEFAULT_CAMERAS,
            obs_type="pixels_agent_pos",
            obj_registries=DEFAULT_REGISTRIES,
            split=DEFAULT_SPLIT,
        )
        try:
            observation, _ = env.reset(seed=seed)
        except Exception:
            _close_env(env)
            raise
        return env, observation

    def _load_policy(self, policy_path: str, device: str) -> _PolicyBundle:
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


def request_from_payload(payload: dict[str, Any]) -> OptionRequest:
    arguments = dict(payload.get("arguments", {}) or {})
    task = str(
        arguments.get("task") or os.environ.get("ROBOCASA_DEFAULT_TASK", "CloseFridge")
    )
    objective = str(payload.get("objective") or arguments.get("objective") or task)
    option_command = str(
        arguments.get("option_command")
        or arguments.get("task_prompt")
        or arguments.get("command")
        or objective
    )
    device = str(
        arguments.get("device") or os.environ.get("ROBOCASA_POLICY_DEVICE") or "cuda"
    )
    return OptionRequest(
        skill_id=str(payload.get("skill_id") or arguments.get("skill_id") or ""),
        objective=objective,
        task=task,
        option_command=option_command,
        seed=int(arguments.get("seed", 1000)),
        max_steps=int(arguments.get("max_steps", 30)),
        policy_path=str(
            arguments.get("policy_path")
            or os.environ.get("ROBOCASA_POLICY", DEFAULT_POLICY)
        ),
        session_id=(
            str(arguments["session_id"]) if arguments.get("session_id") else None
        ),
        reset_episode=bool(arguments.get("reset_episode", False)),
        close_episode=bool(arguments.get("close_episode", False)),
        device=device,
    )


def _policy_observation(
    observation: dict[str, Any],
    option_command: str,
    *,
    input_features: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
    del input_features
    try:
        from lerobot.envs import preprocess_observation
    except ModuleNotFoundError:
        sample = _fallback_preprocess_observation(observation)
    else:
        sample = preprocess_observation(observation)
    # Match lerobot_eval: language-conditioned policies receive a batch-sized
    # list, not a bare string.
    sample["task"] = [option_command]
    sample["robot_type"] = "robocasa"
    return sample


def _fallback_preprocess_observation(observation: dict[str, Any]) -> dict[str, Any]:
    import torch

    sample: dict[str, Any] = {}
    for camera, frame in dict(observation.get("pixels", {}) or {}).items():
        tensor = torch.from_numpy(np.asarray(frame, dtype=np.uint8))
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
        return make_pre_post_processors(config, pretrained_path=policy_path)
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
        raise OptionError(
            "observation_schema_mismatch",
            f"RoboCasa observation state must be 16 finite values, got {state.shape}",
        )
    pixels = dict(observation.get("pixels", {}) or {})
    if len(pixels) < 3:
        raise OptionError(
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
        raise OptionError(
            "action_schema_mismatch", f"policy action must be rank 1, got {array.shape}"
        )
    return array


def _close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


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
