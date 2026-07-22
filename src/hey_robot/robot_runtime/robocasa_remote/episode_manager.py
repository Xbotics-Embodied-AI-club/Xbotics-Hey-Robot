from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Any

import numpy as np


class EpisodeError(RuntimeError):
    def __init__(self, failure_mode: str, message: str) -> None:
        super().__init__(message)
        self.failure_mode = failure_mode


@dataclass(frozen=True)
class TrialSpec:
    trial_id: str
    task: str
    seed: int
    split: str = "target"
    registries: tuple[str, ...] = ("lightwheel",)


@dataclass
class ActiveTrial:
    spec: TrialSpec
    env: Any
    observation: dict[str, Any]
    frame_id: int = 0
    done: bool = False
    official_success: bool = False
    last_reward: float = 0.0
    last_info: dict[str, Any] | None = None
    started_at: float = 0.0
    horizon: int = 1000


@dataclass(frozen=True)
class StepOutcome:
    observation: dict[str, Any]
    frame_id: int
    reward: float
    done: bool
    official_success: bool
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class EpisodeManager:
    """The sole owner of the active RoboCasa environment in one backend."""

    def __init__(
        self,
        *,
        allowed_tasks: frozenset[str],
        env_factory: Callable[[TrialSpec], tuple[Any, dict[str, Any]]] | None = None,
    ) -> None:
        self.allowed_tasks = allowed_tasks
        self._env_factory = env_factory or self._create_environment
        self._active: ActiveTrial | None = None
        self._events: list[dict[str, Any]] = []
        self._lock = RLock()

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active is not None

    def begin_trial(self, spec: TrialSpec) -> ActiveTrial:
        with self._lock:
            if self._active is not None:
                raise EpisodeError("trial_active", "a RoboCasa trial is already active")
            if spec.task not in self.allowed_tasks:
                raise EpisodeError(
                    "invalid_task", f"task {spec.task!r} is not allowlisted"
                )
            try:
                env, observation = self._env_factory(spec)
                _validate_observation(observation)
            except EpisodeError:
                raise
            except Exception as exc:
                raise EpisodeError(
                    "environment_reset_failed",
                    f"failed to create RoboCasa trial: {type(exc).__name__}: {exc}",
                ) from exc
            self._active = ActiveTrial(
                spec=spec,
                env=env,
                observation=observation,
                started_at=time.time(),
                horizon=int(getattr(env, "_max_episode_steps", 1000)),
            )
            self._events = [
                {
                    "kind": "trial_begin",
                    "timestamp": time.time(),
                    "trial_id": spec.trial_id,
                    "task": spec.task,
                    "seed": spec.seed,
                    "split": spec.split,
                    "registries": list(spec.registries),
                    "horizon": self._active.horizon,
                }
            ]
            return self._active

    def current_trial(self) -> ActiveTrial:
        with self._lock:
            if self._active is None:
                raise EpisodeError("trial_unavailable", "no RoboCasa trial is active")
            return self._active

    def observe(self) -> ActiveTrial:
        return self.current_trial()

    def step(
        self,
        action: Any,
        *,
        expected_frame_id: int,
        raw_action: Any | None = None,
        action_clipped: bool = False,
    ) -> StepOutcome:
        with self._lock:
            trial = self.current_trial()
            if trial.done:
                raise EpisodeError("episode_done", "the active trial is already done")
            if expected_frame_id != trial.frame_id:
                raise EpisodeError(
                    "stale_action",
                    f"expected frame {expected_frame_id}, current frame {trial.frame_id}",
                )
            action_array = np.asarray(action, dtype=np.float32)
            if action_array.shape != (12,) or not np.isfinite(action_array).all():
                raise EpisodeError(
                    "action_schema_mismatch", "action must contain 12 finite values"
                )
            if not trial.env.action_space.contains(action_array):
                raise EpisodeError(
                    "action_out_of_bounds", "action is outside the environment space"
                )
            observation, reward, terminated, truncated, info = trial.env.step(
                action_array
            )
            _validate_observation(observation)
            trial.observation = observation
            trial.frame_id += 1
            trial.done = bool(terminated or truncated)
            trial.official_success = bool(dict(info or {}).get("is_success", False))
            trial.last_reward = float(reward)
            trial.last_info = dict(info or {})
            self._events.append(
                {
                    "kind": "action",
                    "timestamp": time.time(),
                    "frame_id": trial.frame_id,
                    "action": [float(value) for value in action_array.tolist()],
                    "raw_action": [
                        float(value)
                        for value in np.asarray(
                            action if raw_action is None else raw_action,
                            dtype=np.float32,
                        ).tolist()
                    ],
                    "action_clipped": bool(action_clipped),
                    "reward": float(reward),
                    "done": trial.done,
                }
            )
            return StepOutcome(
                observation=observation,
                frame_id=trial.frame_id,
                reward=float(reward),
                done=trial.done,
                official_success=trial.official_success,
                terminated=bool(terminated),
                truncated=bool(truncated),
                info=dict(info or {}),
            )

    def read_truth(self) -> dict[str, Any]:
        trial = self.current_trial()
        return {
            "trial_id": trial.spec.trial_id,
            "task": trial.spec.task,
            "seed": trial.spec.seed,
            "split": trial.spec.split,
            "registries": list(trial.spec.registries),
            "frame_id": trial.frame_id,
            "episode_done": trial.done,
            "official_success": trial.official_success,
            "last_reward": trial.last_reward,
            "last_info": dict(trial.last_info or {}),
            "horizon": trial.horizon,
        }

    def record_event(self, kind: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._events.append({"kind": kind, "timestamp": time.time(), **payload})

    def evaluator_events(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(event) for event in self._events]

    def end_trial(self) -> bool:
        with self._lock:
            trial = self._active
            self._active = None
        if trial is None:
            return False
        self.record_event("trial_end", {"trial_id": trial.spec.trial_id})
        close = getattr(trial.env, "close", None)
        if callable(close):
            close()
        return True

    @staticmethod
    def new_spec(
        *,
        task: str,
        seed: int,
        trial_id: str | None = None,
        split: str = "target",
        registries: tuple[str, ...] = ("lightwheel",),
    ) -> TrialSpec:
        return TrialSpec(
            trial_id=trial_id or f"rc-{uuid.uuid4().hex}",
            task=task,
            seed=seed,
            split=split,
            registries=registries,
        )

    @staticmethod
    def _create_environment(spec: TrialSpec) -> tuple[Any, dict[str, Any]]:
        from hey_robot.robot_runtime.robocasa_remote.egl_config import (
            configure_headless_egl,
        )

        configure_headless_egl()
        from lerobot.envs.robocasa import DEFAULT_CAMERAS, RoboCasaEnv
        from robocasa.utils.dataset_registry import (
            ATOMIC_TASK_DATASETS,
            COMPOSITE_TASK_DATASETS,
        )

        task_config = (ATOMIC_TASK_DATASETS | COMPOSITE_TASK_DATASETS).get(
            spec.task, {}
        )
        horizon = int(task_config.get("horizon", 1000))

        env = RoboCasaEnv(
            task=spec.task,
            camera_name=DEFAULT_CAMERAS,
            obs_type="pixels_agent_pos",
            obj_registries=spec.registries,
            split=spec.split,
            episode_length=horizon,
            horizon=horizon,
        )
        try:
            observation, _ = env.reset(seed=spec.seed)
        except Exception:
            env.close()
            raise
        return env, observation


def _validate_observation(observation: dict[str, Any]) -> None:
    state = np.asarray(observation.get("agent_pos", []))
    if state.shape != (16,) or not np.isfinite(state).all():
        raise EpisodeError(
            "observation_schema_mismatch",
            f"observation state must be 16 finite values, got {state.shape}",
        )
    pixels = dict(observation.get("pixels", {}) or {})
    if len(pixels) != 3:
        raise EpisodeError(
            "observation_schema_mismatch",
            f"observation must contain three cameras, got {sorted(pixels)}",
        )


def validate_finite_action(action: list[float]) -> None:
    if len(action) != 12 or not all(math.isfinite(value) for value in action):
        raise EpisodeError(
            "action_schema_mismatch", "action must contain exactly 12 finite values"
        )
