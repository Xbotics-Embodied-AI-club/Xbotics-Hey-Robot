from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ArmPrimitive:
    primitive: str
    arguments: dict[str, Any]
    reason: str


def vla_output_to_primitives(vla_result: dict[str, Any]) -> list[ArmPrimitive]:
    """Convert VLA inference output into robot arm primitives.

    Each VLA inference returns joint targets and a gripper action.
    We produce 1-2 primitives: optionally a move_arm_joints, and optionally
    a set_gripper.
    """
    policy_result = vla_result.get("policy_result")
    if isinstance(policy_result, dict) and policy_result.get("kind") == "action_chunk":
        actions = policy_result.get("actions")
        if isinstance(actions, list) and actions:
            return _action_chunk_to_primitives(actions)

    action_chunk = vla_result.get("action_chunk")
    if isinstance(action_chunk, dict):
        actions = action_chunk.get("actions")
        if isinstance(actions, list) and actions:
            return _action_chunk_to_primitives(actions)

    vla = vla_result.get("vla", vla_result)
    joint_angles: dict[str, float] = dict(vla.get("joint_angles", {}) or {})
    gripper_action: float | None = vla.get("gripper_action")
    task_done: bool = bool(vla.get("task_done", False))

    primitives: list[ArmPrimitive] = []

    if joint_angles:
        primitives.append(
            ArmPrimitive(
                primitive="move_arm_joints",
                arguments={"joints": joint_angles, "mode": "absolute"},
                reason="VLA predicted arm joint targets",
            )
        )

    if gripper_action is not None:
        opening_pct = max(0.0, min(100.0, float(gripper_action) * 100.0))
        primitives.append(
            ArmPrimitive(
                primitive="set_gripper",
                arguments={"opening_pct": opening_pct},
                reason="VLA predicted gripper action",
            )
        )

    if task_done and not primitives:
        primitives.append(
            ArmPrimitive(
                primitive="stop_motion",
                arguments={},
                reason="VLA task completed",
            )
        )

    return primitives


def _action_chunk_to_primitives(actions: list[Any]) -> list[ArmPrimitive]:
    # Only execute the first action from each chunk — the VLA control loop
    # re-runs inference at every step, so executing subsequent actions
    # without re-observing would be open-loop drift.
    if actions and isinstance(actions[0], dict):
        return _single_action_to_primitives(actions[0], action_index=0)
    return []


def _single_action_to_primitives(
    action: dict[str, Any], *, action_index: int
) -> list[ArmPrimitive]:
    joint_angles = dict(
        action.get("joints")
        or action.get("joint_angles")
        or action.get("single_arm")
        or {}
    )
    gripper_action = action.get("gripper")
    if gripper_action is None:
        gripper_action = action.get("gripper_action")
    done = bool(action.get("done", False))
    primitives = vla_output_to_primitives(
        {
            "joint_angles": joint_angles,
            "gripper_action": gripper_action,
            "task_done": done,
        }
    )
    if action_index:
        return [
            ArmPrimitive(
                primitive=primitive.primitive,
                arguments=dict(primitive.arguments),
                reason=f"{primitive.reason} (chunk action {action_index + 1})",
            )
            for primitive in primitives
        ]
    return primitives
