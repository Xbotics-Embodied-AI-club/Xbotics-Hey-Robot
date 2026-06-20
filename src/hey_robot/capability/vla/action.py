from __future__ import annotations

from typing import Any

import numpy as np


def decode_action_chunk(
    action_chunk: dict[str, Any],
    *,
    t: int,
    action_mode: str,
    current_joint_state_rad: np.ndarray,
) -> np.ndarray:
    """Extract the t-th step from a policy action chunk.

    Handles GR00T format (``{"single_arm": ..., "gripper": ...}``) with
    optional legacy fallback for flat ``{"action": ...}`` chunks.

    Args:
        action_chunk: Raw policy output dict.
        t: Which step in the chunk to decode (0-indexed).
        action_mode: One of absolute_joint_position_rad, delta_joint_position_rad,
                     normalized_joint_position.
        current_joint_state_rad: Current 6D joint state for delta / reference.

    Returns:
        6D ndarray of joint targets in radians.
    """
    raw = _extract_arm_action_vector(action_chunk, t)
    if raw is None:
        raise ValueError(
            f"action_chunk missing recognised action key at step {t}. "
            f"Keys: {sorted(action_chunk.keys())}"
        )
    raw = np.asarray(raw, dtype=np.float64)

    if action_mode == "delta_joint_position_rad":
        return np.asarray(current_joint_state_rad, dtype=np.float64) + raw  # type: ignore[no-any-return]
    if action_mode == "normalized_joint_position":
        lo = np.array([-3.14, -0.1, -0.2, -1.8, -3.14, 0.0], dtype=np.float64)
        hi = np.array([3.14, 3.45, 3.14, 1.8, 3.14, 0.08], dtype=np.float64)
        return lo + (raw + 1.0) / 2.0 * (hi - lo)  # type: ignore[no-any-return]
    # absolute_joint_position_rad (default)
    return raw  # type: ignore[return-value]


def get_action_horizon(action_chunk: dict[str, Any]) -> int:
    """Return the number of timesteps in an action chunk.

    Derives horizon from GR00T-format ``single_arm`` or legacy ``action`` key.
    """
    for key in ("single_arm", "action", "actions", "action_chunk"):
        val = action_chunk.get(key)
        if val is not None:
            arr = np.asarray(val)
            if arr.ndim >= 2:
                return int(arr.shape[-2] if arr.ndim == 3 else arr.shape[0])
    return 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_arm_action_vector(
    action_chunk: dict[str, Any], t: int
) -> np.ndarray | None:
    """Extract 6D joint vector at step *t* from a GR00T-format or legacy chunk.

    GR00T format (preferred)::

        {"single_arm": (B, T, 5), "gripper": (B, T, 1)}

    Legacy format (fallback)::

        {"action": (T, 6)}  or  {"actions": (T, 6)}
    """
    # ── GR00T format ──────────────────────────────────────────────────
    single_arm = action_chunk.get("single_arm")
    gripper = action_chunk.get("gripper")
    if single_arm is not None and gripper is not None:
        sa = np.asarray(single_arm, dtype=np.float64)
        gr = np.asarray(gripper, dtype=np.float64)
        if sa.ndim == 3:
            sa = sa[0]
        if gr.ndim == 3:
            gr = gr[0]
        if t >= sa.shape[0] or t >= gr.shape[0]:
            return None
        return np.concatenate([sa[t], gr[t]], axis=0).astype(np.float64)  # type: ignore[no-any-return]

    # ── Legacy flat format (fallback) ─────────────────────────────────
    for key in ("action", "actions", "action_chunk", "output"):
        val = action_chunk.get(key)
        if val is None:
            continue
        arr = np.asarray(val, dtype=np.float64)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim == 2 and t < arr.shape[0]:
            return arr[t]  # type: ignore[no-any-return]
        if arr.ndim == 1 and t == 0:
            return arr  # type: ignore[no-any-return]
    return None
