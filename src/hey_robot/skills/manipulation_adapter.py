from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ArmPrimitive:
    primitive: str
    arguments: dict[str, Any]
    reason: str


def vla_output_to_primitives(vla_result: dict[str, Any]) -> list[ArmPrimitive]:
    """把 VLA 推理输出转换为机器人机械臂 primitive。

    每次 VLA 推理会返回关节目标和夹爪动作。这里最多生成两个 primitive：
    一个可选的 move_arm_joints，以及一个可选的 set_gripper。
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
    # 每个 chunk 只执行第一个动作。VLA 控制循环会在每一步重新推理；
    # 如果不重新观测就继续执行后续动作，会变成开环漂移。
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
