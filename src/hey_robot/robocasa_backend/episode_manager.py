from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
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
    execution_artifact_dir: str | None = None


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
    recorder: ExecutionRecorder | None = None


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


class ExecutionRecorder:
    """Write simulator-owned video and state records for one trial."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.video_path = self.root / "video.mp4"
        self.trajectory_path = self.root / "trajectory.jsonl"
        self._container: Any | None = None
        self._stream: Any | None = None
        self._writer: Any | None = None
        self._trajectory = self.trajectory_path.open("w", encoding="utf-8")
        self.frame_count = 0

    def record(
        self,
        observation: dict[str, Any],
        *,
        frame_id: int,
        action: np.ndarray | None,
        reward: float | None,
        done: bool,
    ) -> None:
        pixels = dict(observation.get("pixels") or {})
        frame = pixels.get("robot0_agentview_left")
        if frame is None:
            raise EpisodeError("recording_failed", "agentview camera is unavailable")
        self._append_video_frame(np.asarray(frame, dtype=np.uint8))
        self.frame_count += 1
        state = np.asarray(observation.get("agent_pos", []), dtype=np.float32)
        self._trajectory.write(
            json.dumps(
                {
                    "frame_id": frame_id,
                    "agent_pos": state.tolist(),
                    "action": action.tolist() if action is not None else None,
                    "reward": reward,
                    "done": done,
                },
                sort_keys=True,
            )
            + "\n"
        )
        self._trajectory.flush()

    def _append_video_frame(self, frame: np.ndarray) -> None:
        """Encode an RGB frame with the backend's pinned PyAV runtime.

        ImageIO selects its PyAV plugin when ``av`` is installed. That plugin
        does not provide a default output codec, which makes the first frame
        fail at runtime. RoboCasa365 already pins PyAV, so own the tiny MP4
        encoding loop here and explicitly request the portable H.264 codec.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise EpisodeError(
                "recording_failed",
                f"agentview frame must be HxWx3 RGB, got {frame.shape}",
            )
        try:
            import av
        except ModuleNotFoundError:
            # Unit and gateway-only environments deliberately omit the heavy
            # RoboCasa dependency group.  Their normal imageio-ffmpeg path
            # remains a valid encoder when PyAV is absent.
            import imageio.v2 as imageio

            if self._writer is None:
                self._writer = imageio.get_writer(
                    self.video_path, fps=5, codec="libx264"
                )
            self._writer.append_data(frame)
            return
        if self._container is None:
            self._container = av.open(str(self.video_path), mode="w")
            self._stream = self._container.add_stream("libx264", rate=5)
            self._stream.width = int(frame.shape[1])
            self._stream.height = int(frame.shape[0])
            self._stream.pix_fmt = "yuv420p"
        video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        container = self._container
        stream = self._stream
        if container is None or stream is None:
            raise EpisodeError("recording_failed", "video encoder was not initialized")
        for packet in stream.encode(video_frame):
            container.mux(packet)

    def close(self) -> None:
        if self._container is not None:
            container = self._container
            stream = self._stream
            if stream is not None:
                for packet in stream.encode():
                    container.mux(packet)
            container.close()
        if self._writer is not None:
            self._writer.close()
        self._trajectory.close()


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
            recorder = (
                ExecutionRecorder(Path(spec.execution_artifact_dir))
                if spec.execution_artifact_dir
                else None
            )
            self._active = ActiveTrial(
                spec=spec,
                env=env,
                observation=observation,
                started_at=time.time(),
                horizon=int(getattr(env, "_max_episode_steps", 1000)),
                recorder=recorder,
            )
            if recorder is not None:
                recorder.record(
                    observation,
                    frame_id=0,
                    action=None,
                    reward=None,
                    done=False,
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
            # RoboCasa's official Xiaomi evaluator passes decoded actions
            # straight through ``convert_action``.  Quantization can produce
            # small, meaningful excursions such as -1.0078125 in the
            # gripper/control-mode channels.  RoboCasaEnv performs that same
            # conversion internally, so do not silently alter or reject a
            # finite checkpoint action here.  Other policy adapters retain
            # their own clipping before they reach this boundary.
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
            if trial.recorder is not None:
                trial.recorder.record(
                    observation,
                    frame_id=trial.frame_id,
                    action=action_array,
                    reward=float(reward),
                    done=trial.done,
                )
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

    def step_chunk(
        self,
        actions: list[Any],
        *,
        expected_frame_id: int,
        raw_actions: list[Any] | None = None,
        action_clipped: bool = False,
    ) -> list[StepOutcome]:
        """Advance a contiguous policy action block under one runtime lock."""
        outcomes: list[StepOutcome] = []
        frame_id = expected_frame_id
        raw_values = raw_actions or actions
        if len(raw_values) != len(actions):
            raise EpisodeError(
                "action_schema_mismatch", "raw action chunk length differs"
            )
        for action, raw_action in zip(actions, raw_values, strict=True):
            outcome = self.step(
                action,
                expected_frame_id=frame_id,
                raw_action=raw_action,
                action_clipped=action_clipped,
            )
            outcomes.append(outcome)
            frame_id = outcome.frame_id
            if outcome.done:
                break
        return outcomes

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
            "execution_artifacts": (
                {
                    "video": str(trial.recorder.video_path),
                    "trajectory": str(trial.recorder.trajectory_path),
                    "recorded_frames": trial.recorder.frame_count,
                }
                if trial.recorder is not None
                else {}
            ),
        }

    def get_task_progress(self) -> dict[str, Any]:
        """Return structured task progress for the active RoboCasa trial.

        Resolves the task env (``wrapper._env.env``) and extracts the live
        ``_check_success`` predicates (e.g. ``success_time``, ``washed_time``,
        ``kettle_on_site``) so the agent can verify sub-goal progress precisely
        instead of relying on a VLM ``inspect_scene`` question.
        """
        trial = self.current_trial()
        raw_env = getattr(trial.env, "_env", None)
        task_env = getattr(raw_env, "env", None) if raw_env is not None else None
        if task_env is None or not hasattr(task_env, "_check_success"):
            return {}
        from hey_robot.robocasa_backend.task_progress import extract_task_progress

        return extract_task_progress(task_env)

    def get_execution_diagnostics(self) -> dict[str, Any]:
        """Return physical signals used to gate a policy retry.

        This deliberately reports ``grasp_contact`` as unknown when a RoboCasa
        wrapper does not expose the simulator helper.  Treating unavailable
        contact as ``False`` would turn missing instrumentation into a false
        empty-grasp diagnosis.
        """
        trial = self.current_trial()
        state = np.asarray(trial.observation.get("agent_pos", []), dtype=np.float64)
        result: dict[str, Any] = {
            "frame_id": trial.frame_id,
            "task": trial.spec.task,
            "task_progress": self.get_task_progress(),
        }
        if state.shape == (16,):
            result.update(
                {
                    "eef_position_relative": state[:3].round(5).tolist(),
                    "base_position": state[7:10].round(5).tolist(),
                    "gripper_qpos": state[14:16].round(5).tolist(),
                    "gripper_closed": bool(float(state[14]) < 0.004),
                }
            )
        raw_env = getattr(trial.env, "_env", None)
        task_env = getattr(raw_env, "env", None) if raw_env is not None else None
        for candidate in (trial.env, raw_env, task_env):
            contact = getattr(candidate, "grasp_contact", None)
            if not callable(contact):
                continue
            try:
                value = contact()
                if isinstance(value, tuple):
                    result["grasp_contact"] = bool(value[0])
                    result["grasp_obj"] = str(value[1]) if value[1] else None
                else:
                    result["grasp_contact"] = bool(value)
                break
            except Exception:  # noqa: S112 - diagnostics are optional
                continue
        if "grasp_contact" not in result and task_env is not None:
            grasp_contact, grasp_obj = _task_grasp_contact(task_env)
            result["grasp_contact"] = grasp_contact
            result["grasp_obj"] = grasp_obj
        return result

    def localize_pixels(
        self,
        *,
        camera: str,
        pixels: list[list[int]],
        expected_frame_id: int,
    ) -> dict[str, Any]:
        """Back-project selected RGB pixels using RPent's metric-depth world map."""
        with self._lock:
            trial = self.current_trial()
            if expected_frame_id != trial.frame_id:
                raise EpisodeError(
                    "stale_observation",
                    f"expected frame {expected_frame_id}, current frame {trial.frame_id}",
                )
            frame = dict(trial.observation.get("pixels") or {}).get(camera)
            if frame is None:
                raise EpisodeError(
                    "camera_unavailable", f"camera {camera!r} is unavailable"
                )
            raw_env = getattr(trial.env, "_env", None)
            task_env = getattr(raw_env, "env", None) if raw_env is not None else None
            sim = getattr(task_env, "sim", None)
            if sim is None:
                raise EpisodeError(
                    "camera_unavailable", "RoboCasa simulator camera is unavailable"
                )
            height, width = np.asarray(frame).shape[:2]
            world_map, depth = _render_world_map(
                sim, camera=camera, height=height, width=width
            )
            results: list[dict[str, Any]] = []
            valid_xyzs: list[list[float]] = []
            for pixel in pixels:
                if len(pixel) != 2:
                    results.append(
                        {
                            "pixel": list(pixel),
                            "world_xyz": None,
                            "valid": False,
                            "error": "pixel must be [row, col]",
                        }
                    )
                    continue
                row, col = int(pixel[0]), int(pixel[1])
                if row < 0 or row >= height or col < 0 or col >= width:
                    results.append(
                        {
                            "pixel": [row, col],
                            "world_xyz": None,
                            "valid": False,
                            "error": (
                                f"pixel ({row},{col}) out of bounds ({height}x{width})"
                            ),
                        }
                    )
                    continue
                xyz = world_map[row, col, :3]
                valid = bool(
                    np.isfinite(xyz).all()
                    and np.isfinite(depth[row, col])
                    and float(depth[row, col]) > 0.0
                )
                rounded = [round(float(value), 4) for value in xyz] if valid else None
                results.append(
                    {
                        "pixel": [row, col],
                        "world_xyz": rounded,
                        "depth_m": round(float(depth[row, col]), 4) if valid else None,
                        "valid": valid,
                        "error": None if valid else "invalid metric depth",
                    }
                )
                if rounded is not None:
                    valid_xyzs.append(rounded)
            summary: dict[str, Any] = {
                "valid_count": len(valid_xyzs),
                "total_count": len(pixels),
            }
            if valid_xyzs:
                summary["median_xyz"] = (
                    np.median(np.asarray(valid_xyzs), axis=0).round(4).tolist()
                )
            return {
                "frame_id": trial.frame_id,
                "camera": camera,
                "method": "simulator_metric_depth",
                "results": results,
                "summary": summary,
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
        if trial.recorder is not None:
            trial.recorder.close()
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
        execution_artifact_dir: str | None = None,
    ) -> TrialSpec:
        return TrialSpec(
            trial_id=trial_id or f"rc-{uuid.uuid4().hex}",
            task=task,
            seed=seed,
            split=split,
            registries=registries,
            execution_artifact_dir=execution_artifact_dir,
        )

    @staticmethod
    def _create_environment(spec: TrialSpec) -> tuple[Any, dict[str, Any]]:
        from hey_robot.robocasa_backend.egl_config import (
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


def _task_grasp_contact(task_env: Any) -> tuple[bool, str | None]:
    """Use RoboCasa's own grasp predicate exactly as RPent's env facade does."""
    try:
        gripper = task_env.robots[0].gripper
        for name, obj in task_env.objects.items():
            try:
                if task_env._check_grasp(gripper, obj):
                    return True, str(name)
            except Exception:  # noqa: S112 - skip unsupported object types
                continue
    except Exception:
        return False, None
    return False, None


def _render_world_map(
    sim: Any, *, camera: str, height: int, width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Render RPent's top-down metric-depth map and pixel-to-world transform."""
    import robosuite.utils.camera_utils as camera_utils

    _rgb, normalized_depth = sim.render(
        width=width,
        height=height,
        camera_name=camera,
        depth=True,
    )
    normalized = np.clip(
        np.nan_to_num(np.asarray(normalized_depth), nan=1.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    )
    if normalized.ndim == 2:
        normalized = normalized[..., None]
    metric_depth = camera_utils.get_real_depth_map(sim, normalized)[..., 0]
    # MuJoCo's depth buffer is bottom-up while the transported RGB is top-down.
    depth = metric_depth[::-1]
    camera_transform = camera_utils.get_camera_transform_matrix(
        sim, camera, height, width
    )
    pixel_to_world = np.linalg.inv(camera_transform)
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    homogeneous = np.stack(
        [cols * depth, rows * depth, depth, np.ones_like(depth)], axis=-1
    )
    world = homogeneous @ pixel_to_world.T
    return world[..., :3], depth


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
