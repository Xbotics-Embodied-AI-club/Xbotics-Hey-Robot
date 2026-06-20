from __future__ import annotations

from typing import Any

import numpy as np


def build_groot_observation(
    *,
    frames: dict[str, np.ndarray],
    joint_state_rad: np.ndarray,
    task_prompt: str,
    camera_key_map: dict[str, str] | None = None,
    state_key_map: dict[str, str] | None = None,
    language_key: str = "language",
    add_batch_time_dims: bool = True,
) -> dict[str, Any]:
    """Build a VLA policy observation dict matching GR00T modality format.

    Produces the nested structure expected by GR00T policy servers
    (verified against RoboCrew's ``_groot_build_observation``)::

        {
            "video": {"camera1": (1, 1, H, W, 3), "camera2": (1, 1, H, W, 3)},
            "state": {"single_arm": (1, 1, 5), "gripper": (1, 1, 1)},
            "language": {"annotation.human.task_description": task_prompt},
        }

    Args:
        frames: camera_name → (H, W, 3) uint8 ndarray.
        joint_state_rad: 6-element canonical joint vector in radians.
        task_prompt: Natural-language task description.
        camera_key_map: Optional rename of camera keys (e.g. front→camera1).
        state_key_map: Optional rename of state sub-keys.
        language_key: Top-level key for the language block (default ``"language"``).
        add_batch_time_dims: Whether to prepend (B=1, T=1) dimensions on all
            ndarray leaves (default ``True``).

    Returns:
        Observation dict ready for ``policy.get_action()``.
    """
    resolved_camera_map = dict(camera_key_map or {})

    video: dict[str, np.ndarray] = {}
    for cam_name, frame in frames.items():
        key = resolved_camera_map.get(cam_name, cam_name)
        video[key] = np.asarray(frame, dtype=np.uint8)

    arm_vec = np.asarray(joint_state_rad[:5], dtype=np.float32)
    gripper_vec = np.asarray(joint_state_rad[5:6], dtype=np.float32)

    state: dict[str, np.ndarray] = {
        "single_arm": arm_vec,
        "gripper": gripper_vec,
    }
    if state_key_map:
        state = {state_key_map.get(k, k): v for k, v in state.items()}

    observation: dict[str, Any] = {
        "video": video,
        "state": state,
        language_key: {
            "annotation.human.task_description": task_prompt,
        },
    }

    if add_batch_time_dims:
        observation = _recursive_add_extra_dim(observation)
        observation = _recursive_add_extra_dim(observation)

    return observation


def _recursive_add_extra_dim(obs: dict) -> dict:
    """Prepend one (batch or time) dimension to every ndarray leaf in *obs*.

    Mirrors RoboCrew's ``_groot_recursive_add_extra_dim``.
    """
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = _recursive_add_extra_dim(val)
        else:
            obs[key] = [val]
    return obs
