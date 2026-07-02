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
