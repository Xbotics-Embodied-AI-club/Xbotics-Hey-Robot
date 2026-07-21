from __future__ import annotations

import math
import multiprocessing
import os
import random
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from threading import Event, Lock
from typing import Any

import numpy as np

try:
    from contract import (
        ALLOWED_TASKS,
        DEFAULT_POLICY,
    )
    from episode_manager import EpisodeError, EpisodeManager
    from policy_probe import _load_raw_config, register_policy_processors
except ModuleNotFoundError:
    from evaluation.robocasa365.worker.contract import (
        ALLOWED_TASKS,
        DEFAULT_POLICY,
    )
    from evaluation.robocasa365.worker.episode_manager import (
        EpisodeError,
        EpisodeManager,
    )
    from evaluation.robocasa365.worker.policy_probe import (
        _load_raw_config,
        register_policy_processors,
    )


class OptionExecutionError(RuntimeError):
    def __init__(self, failure_mode: str, message: str) -> None:
        super().__init__(message)
        self.failure_mode = failure_mode


@dataclass(frozen=True)
class OptionRequest:
    skill_id: str
    option_command: str


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


class VLAOptionExecutor:
    """Stateful, option-level RoboCasa executor for Hey Robot agent evaluation.

    Unlike ``RoboCasaRolloutRunner`` this runner keeps one simulator episode open
    across bounded option calls. The VLA emits the native 12-D RoboCasa action,
    so no real-robot primitive adapter is involved. CUDA inference is isolated
    in a spawned child process because sharing one process with MuJoCo EGL
    corrupts rendered frames on the evaluation host.
    """

    def __init__(
        self,
        *,
        environ: dict[str, str] | None = None,
        env_factory: Callable[[str, int], tuple[Any, dict[str, Any]]] | None = None,
        policy_loader: Callable[[str, str], _PolicyBundle] | None = None,
        manager: EpisodeManager | None = None,
    ) -> None:
        self.environ = dict(environ or os.environ)
        self.default_policy = self.environ.get("ROBOCASA_POLICY", DEFAULT_POLICY)
        self.default_device = self.environ.get("ROBOCASA_POLICY_DEVICE", "cuda")
        # PI052 was trained and evaluated with 50-action chunks. Cutting an
        # option at 30 and resetting the policy discards the final 20 actions
        # of every trajectory before contact-rich completion can happen.
        self.option_horizon = int(self.environ.get("ROBOCASA_OPTION_HORIZON", "50"))
        if env_factory is not None and manager is not None:
            raise ValueError("env_factory and manager cannot both be provided")
        manager_factory = None
        if env_factory is not None:

            def manager_factory(spec):
                return env_factory(spec.task, spec.seed)

        self.manager = manager or EpisodeManager(
            allowed_tasks=ALLOWED_TASKS, env_factory=manager_factory
        )
        self._policy_loader = policy_loader or self._load_isolated_policy
        self._policies: dict[tuple[str, str], Any] = {}
        self._lock = Lock()
        self._cancel_lock = Lock()
        self._cancel_events: dict[str, Event] = {}
        self._last_error: str | None = None

    @property
    def busy(self) -> bool:
        return False

    @property
    def active_sessions(self) -> int:
        return int(self.manager.active)

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

    def cancel(self, operation_id: str) -> bool:
        with self._cancel_lock:
            event = self._cancel_events.get(operation_id)
        if event is None:
            return False
        event.set()
        return True

    def prepare(self) -> None:
        """Load and reset the fixed policy once at the simulator trial boundary."""
        bundle = self._policy_bundle(self.default_policy, self.default_device)
        # Match LeRobot's standalone evaluator: PI052.reset() is an episode
        # operation. It clears both the action queue and hierarchical subtask
        # state, so invoking it at every Agent option boundary breaks policy
        # continuity even when the 50-action queue has already been consumed.
        trial = self.manager.current_trial()
        bundle.reset_action_queue(seed=int(trial.spec.seed))

    def run(self, request: OptionRequest) -> OptionResult:
        started_at = time.monotonic()
        try:
            self._validate_request(request)
            operation_id = request.skill_id
            cancel_event = Event()
            with self._cancel_lock:
                self._cancel_events[operation_id] = cancel_event
            bundle = self._policy_bundle(self.default_policy, self.default_device)
            trial = self.manager.current_trial()
            policy_task = _policy_task_prompt(trial, request.option_command)
            trace: list[dict[str, Any]] = []
            last_reward = 0.0
            clipped_action_count = 0

            for option_step in range(self.option_horizon):
                if cancel_event.is_set():
                    return OptionResult(
                        False,
                        "cancelled",
                        "RoboCasa option cancelled",
                        "cancelled",
                        metrics={"steps_executed": len(trace)},
                    )
                raw_action = bundle.select_action(trial.observation, policy_task)
                action, clipped = _clip_action_to_space(
                    raw_action, trial.env.action_space
                )
                clipped_action_count += int(clipped)
                outcome = self.manager.step(action, expected_frame_id=trial.frame_id)
                last_reward = outcome.reward
                trace.append(
                    {
                        "option_step": option_step,
                        "frame_id": outcome.frame_id,
                        "reward": last_reward,
                        "terminated": outcome.terminated,
                        "truncated": outcome.truncated,
                        "action": [float(value) for value in action.tolist()],
                        "action_clipped": clipped,
                        **(
                            {
                                "raw_action": [
                                    float(value) for value in raw_action.tolist()
                                ]
                            }
                            if clipped
                            else {}
                        ),
                    }
                )
                if outcome.done:
                    break

            option_state = "failed" if trial.done else "boundary_reached"
            status = "failed" if trial.done else "completed"
            failure_mode = "episode_terminal" if trial.done else None
            termination_reason = "episode_terminal" if trial.done else "max_steps"
            metrics = {
                "benchmark": "robocasa365",
                "mode": "embodied_agent_option",
                "policy_type": bundle.policy_type,
                "policy_prompt_source": "environment_task_description",
                "steps_executed": len(trace),
                "option_state": option_state,
                "termination_reason": termination_reason,
                "requires_reobservation": True,
                "before_frame_id": trace[0]["frame_id"] - 1
                if trace
                else trial.frame_id,
                "after_frame_id": trial.frame_id,
                "last_reward": last_reward,
                "clipped_action_count": clipped_action_count,
                "duration_sec": round(time.monotonic() - started_at, 3),
                "trace": trace,
            }
            summary = f"RoboCasa bounded option ended after {len(trace)} steps"
            return OptionResult(
                success=not trial.done,
                status=status,
                summary=summary,
                failure_mode=failure_mode,
                metrics=metrics,
            )
        except (OptionExecutionError, EpisodeError) as exc:
            self._last_error = str(exc)
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
            return OptionResult(
                success=False,
                status="failed",
                summary=f"RoboCasa option failed: {type(exc).__name__}: {exc}",
                failure_mode="execution_failed",
                error=str(exc),
                metrics={"duration_sec": round(time.monotonic() - started_at, 3)},
            )
        finally:
            with self._cancel_lock:
                self._cancel_events.pop(request.skill_id, None)

    def _validate_request(self, request: OptionRequest) -> None:
        if not request.option_command.strip():
            raise OptionExecutionError("invalid_task", "option_command is required")
        if not request.skill_id:
            raise OptionExecutionError("invalid_task", "skill_id is required")

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

    def _load_policy(self, policy_path: str, device: str) -> _PolicyBundle:
        return _load_policy_bundle(policy_path, device)

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


def request_from_payload(payload: dict[str, Any]) -> OptionRequest:
    arguments = dict(payload.get("arguments", {}) or {})
    option_command = str(
        arguments.get("option_command")
        or arguments.get("task_prompt")
        or arguments.get("command")
        or payload.get("objective")
        or ""
    )
    return OptionRequest(
        skill_id=str(payload.get("skill_id") or arguments.get("skill_id") or ""),
        option_command=option_command,
    )


def _policy_task_prompt(trial: Any, option_command: str) -> str:
    """Preserve the task-language contract used by standalone LeRobot eval.

    PI052 treats ``task`` as a high-level root task and generates its own
    low-level subtask internally. Agent option labels must therefore not replace
    RoboCasa's native task description.
    """
    task_description = str(getattr(trial.env, "task_description", "") or "").strip()
    return task_description or option_command.strip()


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


def _clip_action_to_space(
    action: np.ndarray, action_space: Any
) -> tuple[np.ndarray, bool]:
    """Project an unnormalized policy action onto the environment Box contract."""
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    if low.shape != action.shape or high.shape != action.shape:
        raise OptionExecutionError(
            "action_schema_mismatch",
            f"environment action bounds must match {action.shape}",
        )
    clipped = np.clip(action, low, high).astype(np.float32, copy=False)
    return clipped, not np.array_equal(clipped, action)


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
