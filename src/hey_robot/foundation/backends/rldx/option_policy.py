"""RLDX implementation of Hey's local foundation-policy port."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

from .executor import RLDXClient, _action_chunk, _rldx_observation, _rldx_video_frame


class RLDXChunkPolicy:
    """Stateful RLDX chunk policy independent of an agent or model-service RPC.

    The embedding runtime supplies a canonical policy observation through the
    injected encoder.  This keeps the RLDX session/history in Hey's foundation
    layer while leaving RoboCasa-specific observation extraction in its adapter.
    """

    _ACTION_KEYS = (
        "end_effector_position",
        "end_effector_rotation",
        "gripper_close",
        "base_motion",
        "control_mode",
    )

    def __init__(
        self,
        client: RLDXClient,
        *,
        settings: dict[str, Any],
        observation_encoder: Callable[[Any], dict[str, Any]],
    ) -> None:
        self._client = client
        self._settings = dict(settings)
        self._encode = observation_encoder
        self._session_id: str | None = None
        self._reset_memory = True
        self._last_frame_id: int | None = None
        indices = tuple(int(item) for item in self._settings["video_delta_indices"])
        self._video_delta_indices = indices
        self._history: deque[dict[str, np.ndarray]] = deque(maxlen=1 - min(indices))

    def reset(self, session_id: str) -> None:
        self._session_id = session_id
        self._reset_memory = True
        self._last_frame_id = None
        self._history.clear()

    def predict(self, observation: Any, instruction: str) -> list[np.ndarray]:
        if self._session_id is None:
            raise RuntimeError("RLDX session must be reset before predict")
        payload = self._encode(observation)
        frame_id = int(payload["frame_id"])
        if frame_id != self._last_frame_id:
            self._history.append(_rldx_video_frame(payload, settings=self._settings))
            self._last_frame_id = frame_id
        response = self._client.get_action(
            _rldx_observation(
                payload,
                instruction,
                settings=self._settings,
                video_history=self._history,
                video_delta_indices=self._video_delta_indices,
            ),
            {"session_ids": [self._session_id], "reset_memory": [self._reset_memory]},
        )
        self._reset_memory = False
        return _action_chunk(
            response,
            action_keys=self._ACTION_KEYS,
            dimensions=int(self._settings["action_dimensions"]),
            execution_horizon=int(self._settings["execution_horizon"]),
            base_clip=float(self._settings.get("base_clip") or 0.0),
        )
