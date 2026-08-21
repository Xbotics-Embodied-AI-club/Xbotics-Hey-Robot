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

from hey_robot.foundation.options.runner import EmbodimentRuntime

from .contract import CAMERA_RENAME_MAP
from .episode_manager import EpisodeManager


class RoboCasaOptionRuntime(EmbodimentRuntime):
    """Expose an active RoboCasa trial through the generic runtime port."""

    def __init__(self, manager: EpisodeManager) -> None:
        self._manager = manager

    def observe(self) -> dict[str, Any]:
        trial = self._manager.current_trial()
        return {**dict(trial.observation), "frame_id": trial.frame_id}

    def step_block(self, actions: list[Any]) -> tuple[dict[str, Any], bool]:
        outcomes = self._manager.step_chunk(
            actions,
            expected_frame_id=self._manager.current_trial().frame_id,
        )
        if not outcomes:
            raise RuntimeError("policy returned an empty action block")
        outcome = outcomes[-1]
        return dict(outcome.observation), outcome.done

    def progress(self) -> dict[str, Any]:
        return self._manager.get_task_progress()

    def diagnostics(self) -> dict[str, Any]:
        return self._manager.get_execution_diagnostics()


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
        encoded = encode_rldx_observation(observation)
        encoded["raw"] = {"policy_task": instruction}
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
    return {
        "frame_id": int(observation.get("frame_id") or 0),
        "proprioception": list(proprioception) if proprioception is not None else [],
        "images": images,
    }
