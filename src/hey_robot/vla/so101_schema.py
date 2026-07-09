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
ACTION_SPACE = "xlerobot_single_arm_joint"


def gripper_pct_to_unit(value: Any) -> float:
    return max(0.0, min(1.0, float(value or 0.0) / 100.0))


def state_from_arm_status(raw: dict[str, Any]) -> list[float] | None:
    existing = raw.get("vla_state")
    if isinstance(existing, list) and existing:
        return [float(value) for value in existing]

    arm_status = raw.get("arm_status")
    if not isinstance(arm_status, dict):
        return None
    joint_states = arm_status.get("joint_states")
    if not isinstance(joint_states, dict):
        return None
    if not all(name in joint_states for name in SO101_JOINT_NAMES):
        return None

    gripper_pct = arm_status.get("gripper_opening_pct")
    if gripper_pct is None and "gripper" in joint_states:
        gripper_pct = float(joint_states["gripper"]) * 100.0
    return [float(joint_states[name]) for name in SO101_JOINT_NAMES] + [
        gripper_pct_to_unit(gripper_pct)
    ]


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


def action_vector_to_chunk(
    action: Sequence[float], *, embodiment: str = "xlerobot", done: bool = False
) -> dict[str, Any]:
    values = [float(value) for value in action]
    joints = {
        name: round(values[index], 6)
        for index, name in enumerate(SO101_JOINT_NAMES)
        if index < len(values)
    }
    gripper = (
        values[len(SO101_JOINT_NAMES)] if len(values) > len(SO101_JOINT_NAMES) else 0.5
    )
    gripper = max(0.0, min(1.0, float(gripper)))
    return {
        "kind": "action_chunk",
        "action_space": ACTION_SPACE,
        "embodiment": embodiment,
        "horizon": 1,
        "actions": [{"joints": joints, "gripper": gripper, "done": bool(done)}],
        "done": bool(done),
    }


def action_chunk_first_vector(action_chunk: dict[str, Any]) -> list[float] | None:
    actions = action_chunk.get("actions")
    if not isinstance(actions, list) or not actions:
        return None
    first = actions[0]
    if not isinstance(first, dict):
        return None
    joints = dict(first.get("joints") or first.get("joint_angles") or {})
    gripper = first.get("gripper")
    if gripper is None:
        gripper = first.get("gripper_action", 0.5)
    return [float(joints.get(name, 0.0)) for name in SO101_JOINT_NAMES] + [
        max(0.0, min(1.0, float(gripper)))
    ]
