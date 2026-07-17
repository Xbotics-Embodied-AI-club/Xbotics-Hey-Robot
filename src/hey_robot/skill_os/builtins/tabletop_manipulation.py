# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
# 为 Xbotics Hey Robot 修改。
from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from hey_robot.motion.calibration import camera_to_base, load_transform
from hey_robot.motion.density_cluster import density_cluster_mean
from hey_robot.skill_os.base import BaseSkill, SkillResult
from hey_robot.skill_os.builtins.common import spec

HOME_JOINTS = [-0.014, -1.238, 0.562, 0.858, 0.311]
JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
LOCATION_MAP: dict[str, tuple[float, float]] = {
    "front": (0.30, 0.00),
    "front_left": (0.30, 0.12),
    "front_right": (0.30, -0.12),
    "center": (0.22, 0.00),
    "left": (0.22, 0.12),
    "right": (0.22, -0.12),
    "back": (0.12, 0.00),
    "back_left": (0.12, 0.12),
    "back_right": (0.12, -0.12),
}


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


class PickSkill(BaseSkill):
    spec = spec(
        "pick",
        "Pick a named object from the SO101 tabletop.",
        category="arm",
        input_schema={
            "type": "object",
            "properties": {
                "object_label": {"type": "string"},
                "object_id": {"type": "string"},
                "mode": {"type": "string", "enum": ["hold", "drop"]},
                "max_retries": {"type": "integer"},
            },
        },
        required_resources=("arm", "gripper"),
        success_criteria=("target object is held and lifted, or dropped on request",),
        failure_modes=(
            "no_arm",
            "no_gripper",
            "object_not_found",
            "no_detections",
            "no_3d_samples",
            "out_of_workspace",
            "ik_unreachable",
            "move_failed",
            "wrong_object_grasped",
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
        supported_robots=("so101",),
        safety_level="motion",
        timeout_sec=45.0,
        agent_visible=True,
        goal_effects=("gripper_holds_target",),
        evidence_outputs=("mujoco_weld_state", "object_lift_height"),
    )

    async def execute(self, ctx: Any, arguments: dict[str, Any]) -> SkillResult:
        retries = max(1, int(arguments.get("max_retries", 2)))
        last = _failure("pick did not run", "move_failed")
        for attempt in range(1, retries + 1):
            try:
                last = await self._attempt(ctx, arguments)
            except asyncio.CancelledError:
                with np.errstate(all="ignore"):
                    await _primitive(ctx, "stop_motion", {})
                return _failure("pick interrupted", "interrupted")
            except Exception as exc:
                last = _failure(str(exc), "move_failed")
            if last.success:
                return last
            if last.failure_mode in {
                "object_not_found",
                "no_detections",
                "out_of_workspace",
            }:
                break
            if attempt < retries:
                try:
                    await _primitive(
                        ctx,
                        "move_arm_joints",
                        {"joints": _joint_payload(HOME_JOINTS), "duration": 3.0},
                    )
                except Exception:
                    break
        return SkillResult(
            success=False,
            summary=f"pick failed after {attempt} attempt(s): {last.summary}",
            status="failed",
            failure_mode=last.failure_mode,
            error=last.error,
            data={**last.data, "attempts": attempt},
        )

    async def _attempt(self, ctx: Any, arguments: dict[str, Any]) -> SkillResult:
        await _primitive(ctx, "set_gripper", {"action": "open"})
        query = str(
            arguments.get("object_label")
            or arguments.get("object_id")
            or arguments.get("query")
            or ""
        )
        if not query.strip():
            return _failure(
                "no object_label provided; LLM planner must include "
                "'object_label' in the skill slots when calling pick",
                "object_not_found",
            )
        sample_count = max(1, int(arguments.get("sample_count", 20)))
        located = await _primitive(
            ctx,
            "sim_locate_object",
            {
                "query": query,
                "sample_count": sample_count,
                "sample_interval": float(arguments.get("sample_interval", 0.05)),
            },
        )
        if not located.get("operation_success", True):
            return _failure(
                str(located.get("message") or "object not found"),
                str(located.get("failure_mode") or "object_not_found"),
                query=query,
            )
        object_name = str(located.get("object_name") or "")
        samples = np.asarray(located.get("samples") or [], dtype=float)
        if samples.ndim != 2 or samples.shape[1:] != (3,) or len(samples) == 0:
            return _failure("no valid 3D samples", "no_3d_samples")
        threshold = float(arguments.get("cluster_threshold", 0.015))
        target = density_cluster_mean(samples, threshold)
        target = camera_to_base(target, load_transform(None))
        target[2] += float(arguments.get("z_offset", 0.0))
        if bool(arguments.get("hardware_offsets", False)):
            target[0] += float(arguments.get("x_offset", 0.0)) + 0.02
            target[1] += 0.02

        distance_xy = float(np.linalg.norm(target[:2]))
        if distance_xy < 0.05 or distance_xy > 0.35:
            return _failure(
                f"target is outside workspace: {distance_xy:.3f}m",
                "out_of_workspace",
                target_xyz=target.tolist(),
            )

        initial = await _primitive(
            ctx, "sim_get_object_state", {"object_name": object_name}
        )
        initial_position = initial.get("position")
        arm_state = await _primitive(ctx, "arm_get_state", {})
        current = [float(value) for value in arm_state["joint_positions"]]
        pre_grasp = target.copy()
        pre_grasp[2] += float(arguments.get("pre_grasp_height", 0.06))

        pre_solution = await _primitive(
            ctx,
            "arm_solve_position_ik",
            {"target_xyz": pre_grasp.tolist(), "current_joints": current},
        )
        if not pre_solution.get("operation_success", True):
            return _failure("pre-grasp IK unreachable", "ik_unreachable")
        q_pre = [float(value) for value in pre_solution["joint_positions"]]

        grasp_solution = await _primitive(
            ctx,
            "arm_solve_position_ik",
            {"target_xyz": target.tolist(), "current_joints": q_pre},
        )
        if not grasp_solution.get("operation_success", True):
            return _failure("grasp IK unreachable", "ik_unreachable")
        q_grasp = [float(value) for value in grasp_solution["joint_positions"]]

        wrist_offset = float(arguments.get("wrist_roll_offset", 0.0))
        q_pre[4] += wrist_offset
        q_grasp[4] += wrist_offset

        await _primitive(ctx, "set_gripper", {"action": "open"})
        await _primitive(
            ctx,
            "move_arm_joints",
            {"joints": _joint_payload(q_pre), "duration": 3.0},
        )
        await _primitive(
            ctx,
            "move_arm_joints",
            {"joints": _joint_payload(q_grasp), "duration": 1.0},
        )
        await _primitive(ctx, "set_gripper", {"action": "open"})
        await asyncio.sleep(0.3)
        for _ in range(3):
            await _primitive(ctx, "set_gripper", {"action": "close"})
            await asyncio.sleep(0.2)
        await _primitive(
            ctx,
            "move_arm_joints",
            {"joints": _joint_payload(q_pre), "duration": 1.0},
        )
        await _primitive(
            ctx,
            "move_arm_joints",
            {"joints": _joint_payload(HOME_JOINTS), "duration": 3.0},
        )

        mode = str(arguments.get("mode") or "drop")
        if mode == "drop":
            drop = list(HOME_JOINTS)
            drop[0] += 1.57
            await _primitive(
                ctx,
                "move_arm_joints",
                {"joints": _joint_payload(drop), "duration": 3.0},
            )
            await _primitive(ctx, "set_gripper", {"action": "open"})
            await asyncio.sleep(0.5)
            await _primitive(ctx, "set_gripper", {"action": "close"})
            await _primitive(
                ctx,
                "move_arm_joints",
                {"joints": _joint_payload(HOME_JOINTS), "duration": 3.0},
            )
            final = await _primitive(
                ctx, "sim_get_object_state", {"object_name": object_name}
            )
            if final.get("held_object") is not None:
                return _failure("object did not release", "grasp_not_confirmed")
            return SkillResult(
                success=True,
                summary=f"picked and dropped {object_name}",
                data={"object_name": object_name, "mode": mode},
            )

        final = await _primitive(
            ctx, "sim_get_object_state", {"object_name": object_name}
        )
        held = final.get("held_object")
        if held != object_name:
            failure = (
                "wrong_object_grasped" if held is not None else "grasp_not_confirmed"
            )
            return _failure(f"expected to hold {object_name}, holding={held}", failure)
        final_position = final.get("position")
        lift = (
            float(final_position[2]) - float(initial_position[2])
            if initial_position is not None and final_position is not None
            else 0.0
        )
        if lift <= float(arguments.get("min_lift_m", 0.05)):
            return _failure(
                f"{object_name} lift {lift:.3f}m did not exceed threshold",
                "grasp_not_confirmed",
                lift_m=lift,
            )
        return SkillResult(
            success=True,
            summary=f"picked and held {object_name}",
            data={
                "object_name": object_name,
                "mode": mode,
                "lift_m": lift,
                "target_xyz": target.tolist(),
            },
        )


class PlaceSkill(BaseSkill):
    spec = spec(
        "place",
        "Place the object held by the SO101 at a named tabletop location.",
        category="arm",
        input_schema={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "enum": list(LOCATION_MAP),
                },
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
        },
        required_resources=("arm", "gripper"),
        success_criteria=("held object weld is released near the target",),
        failure_modes=(
            "no_arm",
            "no_gripper",
            "gripper_empty",
            "ik_unreachable",
            "move_failed",
            "place_not_confirmed",
            "interrupted",
        ),
        driver_primitives=(
            "sim_get_object_state",
            "arm_get_state",
            "arm_solve_position_ik",
            "move_arm_joints",
            "set_gripper",
        ),
        supported_robots=("so101",),
        safety_level="motion",
        timeout_sec=45.0,
        agent_visible=True,
        goal_effects=("gripper_empty", "object_placed"),
        evidence_outputs=("mujoco_weld_state", "object_final_position"),
    )

    async def execute(self, ctx: Any, arguments: dict[str, Any]) -> SkillResult:
        try:
            held_state = await _primitive(ctx, "sim_get_object_state", {})
            object_name = str(held_state.get("held_object") or "")
            if not object_name:
                return _failure("gripper is empty", "gripper_empty")

            if "x" in arguments and "y" in arguments:
                target_x = float(arguments["x"])
                target_y = float(arguments["y"])
            else:
                location = str(arguments.get("location") or "front")
                target_x, target_y = LOCATION_MAP.get(location, LOCATION_MAP["front"])
            target_z = float(arguments.get("z", 0.04))
            target = np.asarray([target_x, target_y, target_z], dtype=float)
            above = target.copy()
            above[2] += float(arguments.get("pre_grasp_height", 0.06))

            arm_state = await _primitive(ctx, "arm_get_state", {})
            current = [float(value) for value in arm_state["joint_positions"]]
            above_solution = await _primitive(
                ctx,
                "arm_solve_position_ik",
                {"target_xyz": above.tolist(), "current_joints": current},
            )
            if not above_solution.get("operation_success", True):
                return _failure("above-place IK unreachable", "ik_unreachable")
            q_above = [float(value) for value in above_solution["joint_positions"]]
            place_solution = await _primitive(
                ctx,
                "arm_solve_position_ik",
                {"target_xyz": target.tolist(), "current_joints": q_above},
            )
            if not place_solution.get("operation_success", True):
                return _failure("place IK unreachable", "ik_unreachable")
            q_place = [float(value) for value in place_solution["joint_positions"]]

            await _primitive(
                ctx,
                "move_arm_joints",
                {"joints": _joint_payload(q_above), "duration": 3.0},
            )
            await _primitive(
                ctx,
                "move_arm_joints",
                {"joints": _joint_payload(q_place), "duration": 2.0},
            )
            await _primitive(ctx, "set_gripper", {"action": "open"})
            await _primitive(
                ctx,
                "move_arm_joints",
                {"joints": _joint_payload(q_above), "duration": 2.0},
            )
            await _primitive(ctx, "set_gripper", {"action": "close"})
            await _primitive(
                ctx,
                "move_arm_joints",
                {"joints": _joint_payload(HOME_JOINTS), "duration": 3.0},
            )

            final = await _primitive(
                ctx, "sim_get_object_state", {"object_name": object_name}
            )
            position = final.get("position")
            if final.get("held_object") is not None or position is None:
                return _failure(
                    "place release was not confirmed", "place_not_confirmed"
                )
            xy_error = float(
                np.linalg.norm(np.asarray(position[:2], dtype=float) - target[:2])
            )
            tolerance = float(arguments.get("place_tolerance_m", 0.06))
            if xy_error > tolerance or not np.all(np.isfinite(position)):
                return _failure(
                    f"placed object is {xy_error:.3f}m from target",
                    "place_not_confirmed",
                    final_position=position,
                    target_xyz=target.tolist(),
                )
            return SkillResult(
                success=True,
                summary=f"placed {object_name}",
                data={
                    "object_name": object_name,
                    "placed_at": position,
                    "target_xyz": target.tolist(),
                    "xy_error_m": xy_error,
                },
            )
        except asyncio.CancelledError:
            with np.errstate(all="ignore"):
                await _primitive(ctx, "stop_motion", {})
            return _failure("place interrupted", "interrupted")
        except Exception as exc:
            return _failure(str(exc), "move_failed")


class DriverPrimitiveSkill(BaseSkill):
    """机器人驱动实现的 primitive 在 Skill OS 内部的 contract 代理。"""

    def __init__(self, name: str, *, category: str, resources: tuple[str, ...]) -> None:
        self.spec = spec(
            name,
            f"Internal SO101 tabletop primitive: {name}.",
            category=category,
            input_schema={"type": "object", "properties": {}},
            required_resources=resources,
            supported_robots=("so101", "so101_mobile", "xlerobot"),
            safety_level="motion" if resources else "normal",
            agent_visible=False,
            refresh_observation=False,
        )

    async def execute(self, ctx: Any, arguments: dict[str, Any]) -> SkillResult:
        response = await _primitive(ctx, self.spec.name, arguments)
        return SkillResult(
            success=bool(response.get("success", True)),
            summary=str(response.get("message") or self.spec.name),
            data=response,
        )


INTERNAL_PRIMITIVES = (
    DriverPrimitiveSkill("arm_get_state", category="arm", resources=("arm",)),
    DriverPrimitiveSkill("arm_solve_position_ik", category="arm", resources=("arm",)),
    DriverPrimitiveSkill(
        "sim_locate_object", category="perception", resources=("camera",)
    ),
    DriverPrimitiveSkill("sim_get_object_state", category="perception", resources=()),
)
