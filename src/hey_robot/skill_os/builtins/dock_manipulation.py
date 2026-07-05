# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
# Modified for Xbotics Hey Robot: dock wand pick and place.
from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from hey_robot.motion.calibration import camera_to_base, load_transform
from hey_robot.motion.density_cluster import density_cluster_mean
from hey_robot.skill_os.base import BaseSkill, SkillResult
from hey_robot.skill_os.builtins.common import spec

JOINT_NAMES = (
    "Rotation_2",
    "Pitch_2",
    "Elbow_2",
    "Wrist_Pitch_2",
    "Wrist_Roll_2",
)
WORKSPACE_MIN_XY = 0.05
WORKSPACE_MAX_XY = 0.50
WORKSPACE_MIN_Z = 0.05
WORKSPACE_MAX_Z = 1.20
PRE_GRASP_HEIGHT = 0.08
PLACE_APPROACH_HEIGHT = 0.13


def _failure(summary: str, failure_mode: str, **data: Any) -> SkillResult:
    return SkillResult(
        success=False,
        summary=summary,
        status="failed",
        failure_mode=failure_mode,
        error=summary,
        data=data,
    )


async def _primitive(ctx: Any, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if ctx.robot is None:
        raise RuntimeError("robot runtime is unavailable")
    result = await ctx.robot.run(name, arguments)
    if not isinstance(result, dict):
        raise RuntimeError(f"{name} returned a non-object response")
    return result


def _joint_payload(positions: list[float]) -> dict[str, float]:
    return {
        name: float(value) for name, value in zip(JOINT_NAMES, positions, strict=True)
    }


class PickWandSkill(BaseSkill):
    """Pick the cat wand from the dock."""

    spec = spec(
        "pick_wand_from_dock",
        "Pick the cat wand from the dock using the right arm.",
        category="arm",
        input_schema={
            "type": "object",
            "properties": {
                "object_label": {"type": "string"},
                "max_retries": {"type": "integer"},
            },
        },
        required_resources=("arm", "gripper"),
        supported_robots=("so101_mobile",),
        success_criteria=("wand is held and lifted from dock",),
        failure_modes=(
            "no_arm",
            "no_gripper",
            "object_not_found",
            "no_3d_samples",
            "out_of_workspace",
            "ik_unreachable",
            "grasp_not_confirmed",
            "interrupted",
            "simulation_unstable",
        ),
        driver_primitives=(
            "sim_locate_object",
            "sim_get_object_state",
            "arm_get_state",
            "arm_solve_position_ik",
            "move_arm_joints",
            "set_gripper",
        ),
        safety_level="motion",
        timeout_sec=60.0,
        feedback_mode="action",
        goal_effects=("wand_held",),
        evidence_outputs=("pick_wand_result",),
    )

    async def execute(self, ctx, arguments):
        label = str(arguments.get("object_label") or "wand")
        max_retries = min(3, max(1, int(arguments.get("max_retries") or 1)))
        transform = load_transform(None)

        for attempt in range(1, max_retries + 1):
            result = await self._attempt(ctx, label, transform)
            if result.success:
                return result
            if attempt < max_retries and result.failure_mode != "object_not_found":
                await _primitive(ctx, "reset_posture", {})
                await asyncio.sleep(0.5)
        return result

    async def _attempt(self, ctx, label, transform):
        # 1. Open gripper
        await _primitive(ctx, "set_gripper", {"action": "open"})
        await asyncio.sleep(0.2)

        # 2. Locate wand
        located = await _primitive(
            ctx,
            "sim_locate_object",
            {"query": label, "sample_count": 10, "sample_interval": 0.03},
        )
        if not located.get("operation_success"):
            return _failure(
                f"wand not found: {label}",
                "object_not_found",
                query=label,
            )
        samples = located.get("samples", [])
        if not samples:
            return _failure("no 3D samples obtained", "no_3d_samples")

        # 3. Cluster samples
        try:
            point = density_cluster_mean(np.asarray(samples, dtype=float), 0.015)
        except (ValueError, IndexError):
            return _failure("failed to cluster 3D samples", "no_3d_samples")
        target_xyz = camera_to_base(point, transform).tolist()

        # 4. Validate workspace
        xy = float(np.linalg.norm(target_xyz[:2]))
        z = float(target_xyz[2])
        if xy < WORKSPACE_MIN_XY or xy > WORKSPACE_MAX_XY:
            return _failure(f"wand out of XY workspace xy={xy:.3f}", "out_of_workspace")
        if z < WORKSPACE_MIN_Z or z > WORKSPACE_MAX_Z:
            return _failure(f"wand out of Z workspace z={z:.3f}", "out_of_workspace")

        # 5. IK for pre-grasp (above wand)
        pre_grasp = [target_xyz[0], target_xyz[1], target_xyz[2] + PRE_GRASP_HEIGHT]
        pre_ik = await _primitive(
            ctx,
            "arm_solve_position_ik",
            {"target_xyz": pre_grasp},
        )
        if not pre_ik.get("operation_success"):
            return _failure("IK unreachable for pre-grasp", "ik_unreachable")

        # 6. IK for grasp (at wand grasp point)
        grasp_ik = await _primitive(
            ctx,
            "arm_solve_position_ik",
            {"target_xyz": target_xyz, "current_joints": pre_ik["joint_positions"]},
        )
        if not grasp_ik.get("operation_success"):
            return _failure("IK unreachable for grasp", "ik_unreachable")

        # 7. Move to pre-grasp
        await _primitive(
            ctx,
            "move_arm_joints",
            {
                "joints": _joint_payload(pre_ik["joint_positions"]),
                "duration": 2.5,
            },
        )

        # 8. Move to grasp
        await _primitive(
            ctx,
            "move_arm_joints",
            {
                "joints": _joint_payload(grasp_ik["joint_positions"]),
                "duration": 1.5,
            },
        )

        # 9. Close gripper (weld activates)
        await _primitive(ctx, "set_gripper", {"action": "close"})
        await asyncio.sleep(0.3)

        # 10. Lift to pre-grasp
        await _primitive(
            ctx,
            "move_arm_joints",
            {
                "joints": _joint_payload(pre_ik["joint_positions"]),
                "duration": 1.5,
            },
        )

        # 11. Return home
        await _primitive(ctx, "reset_posture", {})

        # 12. Verify grasp
        state = await _primitive(ctx, "sim_get_object_state", {})
        held = state.get("held_object")
        welds = state.get("welds", {})
        wand_weld = welds.get("cat_wand", False)
        if not wand_weld or held != "cat_wand":
            return _failure("grasp not confirmed", "grasp_not_confirmed", held=held)

        return SkillResult(
            success=True,
            summary="wand picked from dock",
            status="completed",
            data={
                "held_object": held,
                "weld_active": wand_weld,
                "target_xyz": target_xyz,
                "attempts": 1,
            },
        )


class PlaceWandSkill(BaseSkill):
    """Place the cat wand back into the dock."""

    spec = spec(
        "place_wand_to_dock",
        "Place the cat wand back into the dock.",
        category="arm",
        input_schema={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
            },
        },
        required_resources=("arm", "gripper"),
        supported_robots=("so101_mobile",),
        success_criteria=("wand is released in dock",),
        failure_modes=(
            "gripper_empty",
            "dock_not_found",
            "ik_unreachable",
            "place_not_confirmed",
            "interrupted",
        ),
        driver_primitives=(
            "sim_get_object_state",
            "arm_solve_position_ik",
            "move_arm_joints",
            "set_gripper",
        ),
        safety_level="motion",
        timeout_sec=60.0,
        feedback_mode="action",
        goal_effects=("wand_placed",),
        evidence_outputs=("place_wand_result",),
    )

    async def execute(self, ctx, arguments=None):  # noqa: ARG002
        # 1. Verify holding wand
        state = await _primitive(ctx, "sim_get_object_state", {})
        held = state.get("held_object")
        if held != "cat_wand":
            return _failure("gripper is empty", "gripper_empty")

        # 2. Get wand current position and dock position
        object_positions = state.get("objects", {})
        wand_pos = object_positions.get("cat_wand")
        if wand_pos is None:
            return _failure("wand position unknown", "place_not_confirmed")

        # Dock position from scene: wand_dock body at (0.35, 0.05, 0.45) base_link
        # dock_insertion site is at z=0.105 above dock body, so insertion is at:
        dock_x, dock_y, dock_z = 0.35, 0.05, 0.64  # wand world pos when in dock

        # 3. Compute approach above dock
        approach_xyz = [dock_x, dock_y, dock_z + PLACE_APPROACH_HEIGHT]
        approach_ik = await _primitive(
            ctx,
            "arm_solve_position_ik",
            {"target_xyz": approach_xyz},
        )
        if not approach_ik.get("operation_success"):
            return _failure("IK unreachable for dock approach", "ik_unreachable")

        # 4. Compute insert at dock
        insert_xyz = [dock_x, dock_y, dock_z + 0.02]  # slightly above insertion point
        insert_ik = await _primitive(
            ctx,
            "arm_solve_position_ik",
            {
                "target_xyz": insert_xyz,
                "current_joints": approach_ik["joint_positions"],
            },
        )
        if not insert_ik.get("operation_success"):
            return _failure("IK unreachable for dock insert", "ik_unreachable")

        # 5. Move to approach
        await _primitive(
            ctx,
            "move_arm_joints",
            {
                "joints": _joint_payload(approach_ik["joint_positions"]),
                "duration": 2.5,
            },
        )

        # 6. Move to insert
        await _primitive(
            ctx,
            "move_arm_joints",
            {
                "joints": _joint_payload(insert_ik["joint_positions"]),
                "duration": 1.5,
            },
        )

        # 7. Release
        await _primitive(ctx, "set_gripper", {"action": "open"})
        await asyncio.sleep(0.5)

        # 8. Retract
        await _primitive(
            ctx,
            "move_arm_joints",
            {
                "joints": _joint_payload(approach_ik["joint_positions"]),
                "duration": 1.5,
            },
        )

        # 9. Return home
        await _primitive(ctx, "reset_posture", {})

        # 10. Verify release
        state2 = await _primitive(ctx, "sim_get_object_state", {})
        welds = state2.get("welds", {})
        wand_weld = welds.get("cat_wand", False)
        if wand_weld:
            return _failure("wand not released", "place_not_confirmed")

        return SkillResult(
            success=True,
            summary="wand placed in dock",
            status="completed",
            data={"released": True, "weld_active": False},
        )
