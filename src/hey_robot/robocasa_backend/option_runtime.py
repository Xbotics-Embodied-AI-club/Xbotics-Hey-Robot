"""RoboCasa embodiment adapter for foundation policy options.

This module owns only RoboCasa sensor/action translation. The option loop and
the policy remain in ``hey_robot.foundation``; evaluation code does not take
part in inference.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from PIL import Image

from hey_robot.foundation.options.protocol import OptionRequest
from hey_robot.foundation.options.runner import EmbodimentRuntime

from .contract import CAMERA_RENAME_MAP
from .episode_manager import EpisodeManager


class RoboCasaOptionRuntime(EmbodimentRuntime):
    """Expose an active RoboCasa trial through the generic runtime port."""

    def __init__(self, manager: EpisodeManager) -> None:
        self._manager = manager
        self._attempt_before: dict[str, Any] = {}
        self._grasp_detected = False
        self._grasp_obj: str | None = None
        self._eef_min_z: float | None = None
        self._peak_lift = 0.0
        self._last_cmd_close = False
        self._last_progress: dict[str, Any] = {}
        self._consecutive_no_interaction = 0
        self._last_counted_frame: int | None = None

    def begin_option(self, request: OptionRequest | None = None) -> None:
        """Reset the RPent-style diagnostics accumulated for one VLA call."""
        self._attempt_before = self._manager.get_execution_diagnostics()
        if request is not None and request.reset_session:
            self._consecutive_no_interaction = 0
            self._last_progress = dict(self._attempt_before.get("task_progress") or {})
        self._grasp_detected = bool(self._attempt_before.get("grasp_contact", False))
        obj = self._attempt_before.get("grasp_obj")
        self._grasp_obj = str(obj) if obj else None
        eef = self._attempt_before.get("eef_position_relative")
        self._eef_min_z = (
            float(eef[2]) if isinstance(eef, list) and len(eef) >= 3 else None
        )
        self._peak_lift = 0.0
        self._last_cmd_close = False

    def observe(self) -> dict[str, Any]:
        trial = self._manager.current_trial()
        observation = {**dict(trial.observation), "frame_id": trial.frame_id}
        raw = dict(observation.get("raw") or {})
        # The environment owns this language.  Keep it separate from the
        # option instruction so foundation policies configured for
        # ``environment_root`` never receive an agent-rewritten subgoal.
        policy_task = str(
            getattr(trial.env, "task_description", "")
            or getattr(trial.spec, "task", "")
            or ""
        ).strip()
        if policy_task:
            raw["policy_task"] = policy_task
        if raw:
            observation["raw"] = raw
        return observation

    def step_block(self, actions: list[Any]) -> tuple[dict[str, Any], bool]:
        outcomes = self._manager.step_chunk(
            actions,
            expected_frame_id=self._manager.current_trial().frame_id,
        )
        if not outcomes:
            raise RuntimeError("policy returned an empty action block")
        for action in actions[: len(outcomes)]:
            values = np.asarray(action, dtype=np.float64).reshape(-1)
            if values.size >= 7:
                self._last_cmd_close = bool(float(values[6]) > 0.0)
        diagnostics = self._manager.get_execution_diagnostics()
        if diagnostics.get("grasp_contact") is True:
            self._grasp_detected = True
            obj = diagnostics.get("grasp_obj")
            self._grasp_obj = self._grasp_obj or (str(obj) if obj else None)
        eef = diagnostics.get("eef_position_relative")
        if isinstance(eef, list) and len(eef) >= 3:
            z = float(eef[2])
            self._eef_min_z = z if self._eef_min_z is None else min(self._eef_min_z, z)
            self._peak_lift = max(self._peak_lift, z - self._eef_min_z)
        outcome = outcomes[-1]
        return dict(outcome.observation), outcome.done

    def progress(self) -> dict[str, Any]:
        return self._manager.get_task_progress()

    def diagnostics(self) -> dict[str, Any]:
        current = self._manager.get_execution_diagnostics()
        qpos = current.get("gripper_qpos")
        grip = float(qpos[0]) if isinstance(qpos, list) and qpos else 0.0
        grasp_contact = bool(current.get("grasp_contact", False))
        held_apart = bool(self._last_cmd_close and 0.004 < grip < 0.039)
        grasped = bool(grasp_contact or held_apart)
        grasp_detected = bool(self._grasp_detected or grasped)
        before_base = self._attempt_before.get("base_position")
        after_base = current.get("base_position")
        base_drift = 0.0
        if (
            isinstance(before_base, list)
            and isinstance(after_base, list)
            and len(before_base) >= 2
            and len(after_base) >= 2
        ):
            base_drift = float(
                np.linalg.norm(
                    np.asarray(after_base[:2], dtype=np.float64)
                    - np.asarray(before_base[:2], dtype=np.float64)
                )
            )
        frame_id = int(current.get("frame_id") or 0)
        progress = dict(current.get("task_progress") or {})
        if frame_id != self._last_counted_frame:
            if grasp_detected or progress != self._last_progress:
                self._consecutive_no_interaction = 0
            else:
                self._consecutive_no_interaction += 1
            self._last_progress = progress
            self._last_counted_frame = frame_id
        return {
            **current,
            "grasped": grasped,
            "grasp_detected": grasp_detected,
            "grasp_contact": grasp_contact,
            "held_apart": held_apart,
            "grasp_obj": current.get("grasp_obj") or self._grasp_obj,
            "peak_lift": round(self._peak_lift, 3),
            "base_drift": round(base_drift, 3),
            "consecutive_no_interaction": self._consecutive_no_interaction,
            "reposition_required": self._consecutive_no_interaction >= 2,
        }


class ExecutorChunkPolicy:
    """Adapt a local foundation worker to the option-loop policy port."""

    def __init__(self, executor: Any) -> None:
        self._executor = executor
        self._session_id: str | None = None

    def reset(self, session_id: str) -> None:
        self._session_id = session_id

    def predict(self, observation: dict[str, Any], instruction: str) -> list[Any]:
        if self._session_id is None:
            raise RuntimeError("policy session is not initialized")
        # The Xiaomi executor is co-located with this RoboCasa backend.  Keep
        # its low-level loop equivalent to the official evaluator by passing
        # the simulator's RGB arrays directly, rather than PNG/base64 hopping
        # through the generic model-service representation every control step.
        # Other executors retain the portable encoded contract.
        local_xiaomi = bool(
            getattr(self._executor, "uses_local_robocasa_observation", False)
        )
        encoded = (
            _local_xiaomi_observation(observation)
            if local_xiaomi
            else encode_rldx_observation(observation)
        )
        result = self._executor.execute(
            {
                "skill_name": "manipulate",
                "episode_id": self._session_id,
                "objective": instruction,
                "arguments": {
                    "policy_session_id": self._session_id,
                    "task_prompt": instruction,
                    "agent_subgoal": instruction,
                    "observation": encoded,
                },
            }
        )
        if not result.get("success"):
            raise RuntimeError(str(result.get("error") or result.get("summary")))
        metrics = dict(result.get("metrics") or {})
        policy_result = dict(metrics.get("policy_result") or {})
        actions = list(policy_result.get("actions") or [])
        values: list[Any] = []
        for action in actions:
            arguments = dict(dict(action).get("arguments") or {})
            values.append(arguments.get("values"))
        if not values or any(item is None for item in values):
            raise RuntimeError("foundation worker returned no native action values")
        return values


def encode_rldx_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Translate RoboCasa sensors to the canonical RLDX policy input."""
    pixels = dict(observation.get("pixels") or {})
    images = []
    for source, target in CAMERA_RENAME_MAP.items():
        native_camera = source.removeprefix("observation.images.")
        camera = target.removeprefix("observation.images.")
        frame = pixels.get(native_camera)
        if frame is None:
            raise ValueError(f"RoboCasa observation is missing {native_camera}")
        encoded = io.BytesIO()
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(encoded, format="PNG")
        images.append(
            {
                "camera": camera,
                "data": base64.b64encode(encoded.getvalue()).decode("ascii"),
            }
        )
    proprioception = observation.get("agent_pos")
    encoded_observation = {
        "frame_id": int(observation.get("frame_id") or 0),
        "proprioception": list(proprioception) if proprioception is not None else [],
        "images": images,
    }
    raw = dict(observation.get("raw") or {})
    policy_task = str(raw.get("policy_task") or "").strip()
    if policy_task:
        encoded_observation["raw"] = {"policy_task": policy_task}
    return encoded_observation


def _local_xiaomi_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Keep raw RoboCasa observations inside the co-located Xiaomi loop."""
    pixels = dict(observation.get("pixels") or {})
    raw_pixels: dict[str, np.ndarray] = {}
    for source, target in CAMERA_RENAME_MAP.items():
        native_camera = source.removeprefix("observation.images.")
        camera = target.removeprefix("observation.images.")
        frame = pixels.get(native_camera)
        if frame is None:
            raise ValueError(f"RoboCasa observation is missing {native_camera}")
        raw_pixels[camera] = np.ascontiguousarray(frame, dtype=np.uint8)
    raw = dict(observation.get("raw") or {})
    return {
        "frame_id": int(observation.get("frame_id") or 0),
        "proprioception": list(
            observation["agent_pos"] if observation.get("agent_pos") is not None else []
        ),
        "raw_pixels": raw_pixels,
        "raw": {"policy_task": str(raw.get("policy_task") or "")},
    }
