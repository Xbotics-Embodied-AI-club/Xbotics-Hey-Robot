"""VLA vector codec for the SO101 arm used by XLeRobot."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

SO101_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
SO101_VECTOR_NAMES: tuple[str, ...] = (*SO101_JOINT_NAMES, "gripper")
SO101_STATE_SCHEMA = "so101_single_arm_rad_gripper01"


def state_from_sim_driver(driver: Any, *, arm: str = "right") -> list[float]:
    state = driver.read_arm_state(arm)
    values = [float(state.get(name, 0.0)) for name in SO101_VECTOR_NAMES]
    gripper_open = float(getattr(driver.adapter, "gripper_open_value", 1.0) or 1.0)
    values[-1] = max(0.0, min(1.0, values[-1] / gripper_open))
    return values


def action_vector_to_targets(action: Sequence[float], driver: Any) -> np.ndarray:
    values = np.asarray(list(action), dtype=np.float64).reshape(-1)
    if values.size < len(SO101_VECTOR_NAMES):
        padded = np.zeros((len(SO101_VECTOR_NAMES),), dtype=np.float64)
        padded[: values.size] = values
        values = padded
    targets = values[: len(SO101_VECTOR_NAMES)].copy()
    gripper_open = float(getattr(driver.adapter, "gripper_open_value", 1.0) or 1.0)
    targets[-1] = max(0.0, min(1.0, float(targets[-1]))) * gripper_open
    return targets
