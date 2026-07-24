"""Dock wand 的 native Skill；编排与 Robot Runtime primitive 保持分层。"""

from __future__ import annotations

from typing import Any

from hey_robot.skills.context import SkillContext
from hey_robot.skills.models import Skill, SkillResult
from hey_robot.skills.registry import SkillRegistry

_JOINT_NAMES = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")


async def pick_wand_from_dock(
    ctx: SkillContext, arguments: dict[str, Any]
) -> SkillResult:
    """在 simulation oracle 路径中抓取 dock wand。"""
    mode = str(arguments.get("mode", "oracle")).lower()
    if mode != "oracle":
        return _failure(
            "native dock 当前只支持 simulation oracle 模式。",
            "mode_not_migrated",
        )
    label = str(arguments.get("object_label", "wand"))
    opened = await _call(ctx, "set_gripper", {"action": "open"})
    if isinstance(opened, SkillResult):
        return opened
    located = await _call(ctx, "sim_locate_object", {"query": label, "sample_count": 1})
    if isinstance(located, SkillResult):
        return located
    if not bool(located.get("operation_success")):
        return _failure("未找到 dock wand。", "object_not_found", query=label)
    samples = located.get("samples")
    if not isinstance(samples, list) or not samples or not isinstance(samples[0], list):
        return _failure("dock wand 缺少 3D sample。", "no_3d_samples")
    try:
        target = [float(value) for value in samples[0][:3]]
    except (TypeError, ValueError):
        return _failure("dock wand 3D sample 无效。", "no_3d_samples")
    if len(target) != 3:
        return _failure("dock wand 3D sample 无效。", "no_3d_samples")
    pre = [target[0], target[1], target[2] + 0.08]
    pre_ik = await _call(ctx, "arm_solve_position_ik", {"target_xyz": pre})
    if isinstance(pre_ik, SkillResult):
        return pre_ik
    if not bool(pre_ik.get("operation_success")):
        return _failure("dock pre-grasp IK 不可达。", "ik_unreachable")
    grasp_ik = await _call(
        ctx,
        "arm_solve_position_ik",
        {
            "target_xyz": target,
            "current_joints": pre_ik.get("joint_positions"),
            "target_axis": located.get("grasp_axis"),
        },
    )
    if isinstance(grasp_ik, SkillResult):
        return grasp_ik
    if not bool(grasp_ik.get("operation_success")):
        return _failure("dock grasp IK 不可达。", "ik_unreachable")
    for joints, duration in (
        (pre_ik.get("joint_positions"), 2.5),
        (grasp_ik.get("joint_positions"), 1.5),
    ):
        moved = await _move_joints(ctx, joints, duration)
        if moved is not None:
            return moved
    closed = await _call(ctx, "set_gripper", {"action": "close"})
    if isinstance(closed, SkillResult):
        return closed
    moved = await _move_joints(ctx, pre_ik.get("joint_positions"), 1.5)
    if moved is not None:
        return moved
    reset = await _call(ctx, "reset_posture", {})
    if isinstance(reset, SkillResult):
        return reset
    state = await _call(ctx, "sim_get_object_state", {})
    if isinstance(state, SkillResult):
        return state
    weld_active = bool(dict(state.get("welds") or {}).get("wand"))
    held = state.get("held_object")
    if held != "wand" or not weld_active:
        return _failure("dock wand 抓取未确认。", "grasp_not_confirmed", held=held)
    return SkillResult(
        True,
        "wand 已从 dock 抓取。",
        "completed",
        data={"held_object": held, "weld_active": weld_active, "target_xyz": target},
    )


async def place_wand_to_dock(
    ctx: SkillContext, _arguments: dict[str, Any]
) -> SkillResult:
    """将当前持有的 wand 放回 simulation dock。"""
    state = await _call(ctx, "sim_get_object_state", {})
    if isinstance(state, SkillResult):
        return state
    if state.get("held_object") != "wand":
        return _failure("当前夹爪未持有 wand。", "gripper_empty")
    target = state.get("dock_target")
    if not isinstance(target, list | tuple) or len(target) != 3:
        return _failure("dock target 不可用。", "dock_not_found")
    try:
        dock = [float(value) for value in target]
    except (TypeError, ValueError):
        return _failure("dock target 无效。", "dock_not_found")
    approach = [dock[0], dock[1], dock[2] + 0.13]
    approach_ik = await _call(ctx, "arm_solve_position_ik", {"target_xyz": approach})
    if isinstance(approach_ik, SkillResult):
        return approach_ik
    if not bool(approach_ik.get("operation_success")):
        return _failure("dock approach IK 不可达。", "ik_unreachable")
    insert_ik = await _call(
        ctx,
        "arm_solve_position_ik",
        {
            "target_xyz": [dock[0], dock[1], dock[2] + 0.02],
            "current_joints": approach_ik.get("joint_positions"),
        },
    )
    if isinstance(insert_ik, SkillResult):
        return insert_ik
    if not bool(insert_ik.get("operation_success")):
        return _failure("dock insert IK 不可达。", "ik_unreachable")
    for joints, duration in (
        (approach_ik.get("joint_positions"), 2.5),
        (insert_ik.get("joint_positions"), 1.5),
    ):
        moved = await _move_joints(ctx, joints, duration)
        if moved is not None:
            return moved
    released = await _call(ctx, "set_gripper", {"action": "open"})
    if isinstance(released, SkillResult):
        return released
    moved = await _move_joints(ctx, approach_ik.get("joint_positions"), 1.5)
    if moved is not None:
        return moved
    reset = await _call(ctx, "reset_posture", {})
    if isinstance(reset, SkillResult):
        return reset
    state = await _call(ctx, "sim_get_object_state", {})
    if isinstance(state, SkillResult):
        return state
    if bool(dict(state.get("welds") or {}).get("wand")):
        return _failure("dock wand 未释放。", "place_not_confirmed")
    return SkillResult(True, "wand 已放回 dock。", "completed", data={"released": True})


async def _call(
    ctx: SkillContext, action: str, arguments: dict[str, Any]
) -> dict[str, Any] | SkillResult:
    if ctx.robot is None:
        return _failure("RobotClient 不可用。", "robot_client_unavailable")
    result = await ctx.robot.execute(ctx.robot_id, action, arguments, run_id=ctx.run_id)
    if not result.success:
        return _failure(
            result.summary,
            result.failure_mode or "primitive_execution_failed",
            error=result.error,
        )
    return dict(result.data)


async def _move_joints(
    ctx: SkillContext, positions: Any, duration: float
) -> SkillResult | None:
    if not isinstance(positions, list) or len(positions) != len(_JOINT_NAMES):
        return _failure("IK joint_positions 无效。", "ik_unreachable")
    result = await _call(
        ctx,
        "move_arm_joints",
        {
            "joints": dict(zip(_JOINT_NAMES, positions, strict=True)),
            "duration": duration,
        },
    )
    return result if isinstance(result, SkillResult) else None


def _failure(summary: str, failure_mode: str, **data: Any) -> SkillResult:
    return SkillResult(
        False,
        summary,
        "failed",
        data=data,
        failure_mode=failure_mode,
        error=data.get("error"),
    )


PICK_WAND_FROM_DOCK = Skill(
    "pick_wand_from_dock",
    "Pick the wand from the simulation dock.",
    {
        "type": "object",
        "properties": {
            "object_label": {"type": "string"},
            "mode": {"type": "string", "enum": ["oracle", "perception"]},
        },
        "additionalProperties": False,
    },
    pick_wand_from_dock,
    resources=("arm", "gripper"),
    timeout_sec=60.0,
    supported_robots=("xlerobot", "so101_mobile"),
    required_actions=(
        "sim_locate_object",
        "arm_solve_position_ik",
        "move_arm_joints",
        "set_gripper",
        "sim_get_object_state",
    ),
)
PLACE_WAND_TO_DOCK = Skill(
    "place_wand_to_dock",
    "Place the wand into the simulation dock.",
    {"type": "object", "properties": {}, "additionalProperties": False},
    place_wand_to_dock,
    resources=("arm", "gripper"),
    timeout_sec=60.0,
    supported_robots=("xlerobot", "so101_mobile"),
    required_actions=(
        "sim_get_object_state",
        "arm_solve_position_ik",
        "move_arm_joints",
        "set_gripper",
    ),
)


def register(registry: SkillRegistry) -> None:
    registry.register(PICK_WAND_FROM_DOCK)
    registry.register(PLACE_WAND_TO_DOCK)
