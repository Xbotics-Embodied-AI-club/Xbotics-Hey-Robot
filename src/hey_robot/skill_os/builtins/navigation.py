from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any

from hey_robot.protocol import RobotObservation
from hey_robot.skill_os.base import BaseSkill, SkillResult
from hey_robot.skill_os.builtins.common import spec
from hey_robot.skill_os.builtins.navigation_adapter import (
    PrimitiveCommand,
    planner_output_to_primitive,
)
from hey_robot.skill_os.perception.human_follow import (
    HumanFollowRunner,
    VelocityCommand,
    load_detector,
)


class MoveBaseSkill(BaseSkill):
    spec = spec(
        "move_base",
        "Move the base forward, backward, left, or right by a short distance in centimeters.",
        category="base",
        input_schema={
            "type": "object",
            "properties": {
                "direction": {"type": "string"},
                "distance_cm": {"type": "number"},
            },
            "required": ["direction", "distance_cm"],
        },
        required_resources=("base",),
        driver_primitives=("move_base",),
        safety_level="motion",
        timeout_sec=8.0,
        agent_visible=False,
        goal_effects=("changes_base_position",),
        evidence_outputs=("base_move_action_result",),
        cannot_satisfy=("weak_scene_observation", "base_turn_action_result"),
    )

    async def execute(self, ctx, arguments):
        await ctx.robot.move_base(**arguments)
        return SkillResult(success=True, summary="Base motion completed.")


class TurnBaseSkill(BaseSkill):
    spec = spec(
        "turn_base",
        "Turn the base left or right by a bounded angle in degrees.",
        category="base",
        input_schema={
            "type": "object",
            "properties": {
                "direction": {"type": "string"},
                "angle_deg": {"type": "number"},
            },
            "required": ["direction", "angle_deg"],
        },
        required_resources=("base",),
        driver_primitives=("turn_base",),
        safety_level="motion",
        timeout_sec=8.0,
        agent_visible=False,
        goal_effects=("changes_base_orientation",),
        evidence_outputs=("base_turn_action_result",),
        cannot_satisfy=("weak_scene_observation", "base_move_action_result"),
    )

    async def execute(self, ctx, arguments):
        await ctx.robot.turn_base(**arguments)
        return SkillResult(success=True, summary="Base turn completed.")


class BaseVelocityStepSkill(BaseSkill):
    spec = spec(
        "base_velocity_step",
        "Apply a short bounded base velocity command for supervised following.",
        category="base",
        input_schema={
            "type": "object",
            "properties": {
                "vx": {"type": "number"},
                "vy": {"type": "number"},
                "wz": {"type": "number"},
                "duration_ms": {"type": "integer"},
            },
            "required": ["vx", "vy", "wz", "duration_ms"],
        },
        required_resources=("base",),
        driver_primitives=("base_velocity_step",),
        safety_level="motion",
        timeout_sec=3.0,
        agent_visible=False,
        goal_effects=("changes_base_velocity",),
        evidence_outputs=("base_velocity_action_result",),
    )

    async def execute(self, ctx, arguments):
        await ctx.robot.base_velocity_step(**arguments)
        return SkillResult(success=True, summary="Base velocity step completed.")


class _VLNNavigationSkill(BaseSkill):
    async def execute(self, ctx, arguments):
        if ctx.model_services is None:
            return SkillResult(
                success=False,
                summary=f"{self.spec.name} requires a VLN model service",
                status="failed",
                failure_mode="model_service_unavailable",
                error="model service port is unavailable",
            )
        max_steps = max(1, int(arguments.get("max_steps") or 1))
        execute_primitives = bool(arguments.get("execute_primitives", True))
        steps: list[dict[str, Any]] = []
        planner_data: dict[str, Any] = {}
        result = None
        look_down_requested = bool(arguments.get("look_down", False))

        for step_index in range(max_steps):
            payload = _vln_payload(ctx, arguments)
            if step_index > 0:
                payload["reset_policy"] = False
            if look_down_requested:
                payload["look_down"] = True
            result = await ctx.model_services.call(
                self.spec.required_model_service, payload
            )
            planner_data = _extract_vln_planner(result)
            command: PrimitiveCommand | None = None
            try:
                command = planner_output_to_primitive(planner_data)
            except ValueError:
                command = None
            await _emit_vln_progress(
                ctx,
                step="planning",
                summary=getattr(result, "summary", None) or "VLN planner returned",
                progress=min(0.9, 0.2 + step_index * 0.3),
                planner=planner_data,
                command=command,
            )
            if not bool(getattr(result, "success", False)):
                return SkillResult(
                    success=False,
                    summary=str(getattr(result, "summary", "") or "VLN planner failed"),
                    status=str(getattr(result, "status", "") or "failed"),
                    failure_mode=getattr(result, "failure_mode", None)
                    or "vln_planner_failed",
                    error=getattr(result, "error", None),
                    data=dict(getattr(result, "metrics", {}) or {}),
                )
            if _requires_secondary_observation(planner_data):
                look_down_requested = True
                await _emit_vln_progress(
                    ctx,
                    step="secondary_observation",
                    summary="VLN planner requested look-down observation",
                    progress=min(0.95, 0.3 + step_index * 0.3),
                    planner=planner_data,
                    command=None,
                )
                if step_index + 1 < max_steps and ctx.invoke is not None:
                    await ctx.invoke(
                        "inspect_scene",
                        {
                            "camera": arguments.get("camera", "front"),
                            "look_down": True,
                        },
                    )
                    continue
                return SkillResult(
                    success=False,
                    summary="VLN planner requires a secondary look-down observation",
                    status="failed",
                    failure_mode="vln_secondary_observation_required",
                    error="planner requested look_down but no planning step remains",
                    data=dict(getattr(result, "metrics", {}) or {}),
                )
            if not execute_primitives:
                return SkillResult(
                    success=True,
                    summary=str(
                        getattr(result, "summary", "") or "VLN planner completed"
                    ),
                    status=str(getattr(result, "status", "") or "completed"),
                    data=dict(getattr(result, "metrics", {}) or {}),
                )
            if ctx.robot is None:
                return SkillResult(
                    success=False,
                    summary="robot runtime is required to execute VLN primitives",
                    status="failed",
                    failure_mode="robot_runtime_unavailable",
                    error="robot runtime port is unavailable",
                    data=dict(getattr(result, "metrics", {}) or {}),
                )
            if command is None:
                return SkillResult(
                    success=False,
                    summary="VLN planner did not return an executable primitive",
                    status="failed",
                    failure_mode="vln_no_valid_goal",
                    error="planner output lacks stop, heading_deg, or pixel_goal",
                    data=dict(getattr(result, "metrics", {}) or {}),
                )
            step = await _execute_vln_primitive(ctx, command)
            steps.append(step)
            await _emit_vln_progress(
                ctx,
                step="executed",
                summary=step["message"],
                progress=min(0.95, 0.4 + step_index * 0.3),
                planner=planner_data,
                command=command,
                primitive_result=step.get("primitive_result"),
            )
            if not step["success"]:
                return SkillResult(
                    success=False,
                    summary=step["message"],
                    status="failed",
                    failure_mode="primitive_execution_failed",
                    error=step.get("error"),
                    data={"vln": planner_data, "steps": steps},
                )
            if command.primitive == "stop_motion":
                break
            if step_index + 1 < max_steps and ctx.invoke is not None:
                await ctx.invoke(
                    "inspect_scene", {"camera": arguments.get("camera", "front")}
                )

        summary = (
            str(getattr(result, "summary", "") or "VLN navigation completed")
            if result is not None
            else "VLN navigation completed"
        )
        return SkillResult(
            success=True,
            summary=summary,
            data={"vln": planner_data, "steps": steps},
        )


class NavigateToSkill(_VLNNavigationSkill):
    spec = spec(
        "navigate_to",
        "Navigate toward a semantic target using a foundation VLN planner.",
        category="navigation",
        input_schema={
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "instruction": {"type": "string"},
                "camera": {"type": "string"},
                "image_path": {"type": "string"},
                "execute_primitives": {"type": "boolean"},
                "max_steps": {"type": "integer"},
            },
            "required": ["target"],
        },
        required_resources=("camera",),
        dependencies=("inspect_scene",),
        driver_primitives=("move_base", "turn_base", "stop_motion"),
        required_model_service="navigate_to",
        safety_level="motion",
        timeout_sec=60.0,
        agent_visible=True,
        feedback_mode="vision",
        goal_effects=("approaches_named_place",),
        evidence_outputs=("vln_planner_result", "base_motion_action_result"),
        cannot_satisfy=("weak_scene_observation",),
    )


class ApproachObjectSkill(_VLNNavigationSkill):
    spec = spec(
        "approach_object",
        "Approach a visible or named object using a foundation VLN planner.",
        category="navigation",
        input_schema={
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "instruction": {"type": "string"},
                "camera": {"type": "string"},
                "image_path": {"type": "string"},
                "execute_primitives": {"type": "boolean"},
                "max_steps": {"type": "integer"},
            },
            "required": ["target"],
        },
        required_resources=("camera",),
        dependencies=("inspect_scene",),
        driver_primitives=("move_base", "turn_base", "stop_motion"),
        required_model_service="approach_object",
        safety_level="motion",
        timeout_sec=60.0,
        agent_visible=True,
        feedback_mode="vision",
        goal_effects=("approaches_object",),
        evidence_outputs=("vln_planner_result", "base_motion_action_result"),
        cannot_satisfy=("weak_scene_observation",),
    )


class HumanFollowSkill(BaseSkill):
    spec = spec(
        "human_follow",
        "Continuously follow a visible person using the base only until cancelled or completed.",
        category="interaction",
        input_schema={
            "type": "object",
            "properties": {
                "duration_sec": {"type": "number"},
                "max_steps": {"type": "integer"},
                "target_distance_m": {"type": "number"},
                "target_height_ratio": {"type": "number"},
            },
        },
        required_resources=("camera", "base"),
        dependencies=("inspect_scene",),
        driver_primitives=("base_velocity_step", "stop_motion"),
        safety_level="motion",
        timeout_sec=300.0,
        agent_visible=True,
        feedback_mode="vision",
        goal_effects=("tracks_and_follows_person",),
        evidence_outputs=("human_follow_action_result",),
        cannot_satisfy=("weak_scene_observation",),
    )

    async def execute(self, ctx, arguments):
        service = getattr(ctx, "human_follow", None)
        skill_id = getattr(ctx, "skill_id", None)
        robot_id = getattr(ctx, "robot_id", None)
        if service is not None and skill_id and robot_id:
            return await service.run(
                robot_id=robot_id,
                skill_id=skill_id,
                arguments=dict(arguments),
                progress=getattr(ctx, "progress", None),
            )

        # Local mode: use shared HumanFollowRunner with camera frames from bus.
        if ctx.get_camera_frame is None:
            return SkillResult(
                success=False,
                summary="human follow requires camera frame access",
                failure_mode="camera_unavailable",
                error="camera frame stream is unavailable",
            )

        load_detector(str(arguments.get("model_path") or "models/yolo26n.pt"))
        steps: list[dict] = []
        finished = asyncio.Event()

        async def get_frame():
            return ctx.get_camera_frame()

        async def apply_velocity(vx, vy, wz):
            if abs(vx) < 0.002 and abs(wz) < 0.0002:
                return
            await ctx.robot.base_velocity_step(
                vx=vx,
                vy=vy,
                wz=wz,
                duration_ms=400,
            )
            steps.append(
                {
                    "success": True,
                    "skill": "base_velocity_step",
                    "message": "follow velocity step",
                    "command": {"vx": vx, "vy": vy, "wz": wz},
                }
            )

        async def emit_progress(**payload):
            phase = payload.get("phase", "following")
            summary = payload.get("summary", "")
            command_raw = payload.get("command")
            if isinstance(command_raw, dict):
                command = VelocityCommand(
                    vx=float(command_raw.get("vx") or 0),
                    vy=float(command_raw.get("vy") or 0),
                    vz=float(command_raw.get("wz") or 0),
                )
            elif isinstance(command_raw, VelocityCommand):
                command = command_raw
            else:
                command = None
            await _emit_follow_progress(
                ctx,
                phase=phase,
                summary=summary,
                progress=0.5 if phase == "following" else 0.3,
                observation=ctx.current_observation()
                if ctx.current_observation
                else None,
                target=payload.get("target"),
                detections=payload.get("detections"),
                command=command,
                mode=phase,
            )

        async def on_stop():
            await ctx.robot.stop_motion()

        runner = HumanFollowRunner(
            arguments,
            get_frame=get_frame,
            apply_velocity=apply_velocity,
            emit_progress=emit_progress,
            is_stopped=lambda: finished.is_set(),
            on_stop=on_stop,
        )

        try:
            result = await runner.run()
        except asyncio.CancelledError:
            await ctx.robot.stop_motion()
            raise

        await ctx.robot.stop_motion()
        await _emit_follow_progress(
            ctx,
            phase="completed" if result.get("success") else "lost",
            summary=result.get("summary", "human follow stopped"),
            progress=1.0 if result.get("success") else 0.0,
            observation=ctx.current_observation() if ctx.current_observation else None,
            command=VelocityCommand(0.0, 0.0, 0.0),
            mode="completed" if result.get("success") else "failed",
        )
        return SkillResult(
            success=result.get("success", False),
            summary=result.get("summary", "human follow stopped"),
            failure_mode=result.get("failure_mode"),
            error=result.get("error"),
            data={
                "steps": steps,
                "mode": "completed" if result.get("success") else "failed",
            },
        )


def _vln_payload(_ctx: Any, arguments: dict[str, Any]) -> dict[str, Any]:
    """Build a stable VLN policy payload from explicit args and current observation."""
    payload = {
        key: value
        for key, value in dict(arguments).items()
        if key not in {"execute_primitives", "max_steps"}
    }
    if "observation" not in payload and "image_path" not in payload:
        observation = _ctx.current_observation() if _ctx.current_observation else None
        observation_payload = _observation_payload(
            observation,
            camera=payload.get("camera"),
        )
        if observation_payload is not None:
            payload["observation"] = observation_payload
    skill_id = getattr(_ctx, "skill_id", None)
    if skill_id and not payload.get("policy_session_id"):
        payload["policy_session_id"] = skill_id
    payload.setdefault("reset_policy", True)
    return payload


def _observation_payload(
    observation: RobotObservation | None, *, camera: object | None = None
) -> dict[str, Any] | None:
    if observation is None:
        return None
    images = observation.images
    if camera:
        preferred = [image for image in images if image.camera == str(camera)]
        if preferred:
            images = preferred
    return {
        "frame_id": observation.frame_id,
        "timestamp": observation.envelope.timestamp,
        "images": [asdict(image) for image in images],
        "proprioception": list(observation.proprioception),
        "raw": dict(observation.raw),
    }


def _extract_vln_planner(result) -> dict[str, Any]:
    metrics = dict(getattr(result, "metrics", {}) or {})
    planner = metrics.get("vln")
    if isinstance(planner, dict):
        return dict(planner)
    for key in ("mode", "pixel_goal", "heading_deg", "stop"):
        if key in metrics:
            return metrics
    return {}


def _requires_secondary_observation(planner: dict[str, Any]) -> bool:
    return (
        bool(planner.get("requires_secondary_observation"))
        or str(planner.get("mode") or "") == "look_down_required"
    )


async def _execute_vln_primitive(ctx, command: PrimitiveCommand) -> dict[str, Any]:
    try:
        method = getattr(ctx.robot, command.primitive)
        primitive_result = await method(**command.arguments)
    except Exception as exc:
        return {
            "success": False,
            "primitive": command.primitive,
            "arguments": dict(command.arguments),
            "reason": command.reason,
            "message": str(exc),
            "error": str(exc),
        }
    return {
        "success": True,
        "primitive": command.primitive,
        "arguments": dict(command.arguments),
        "reason": command.reason,
        "message": f"{command.primitive} completed",
        "primitive_result": primitive_result,
    }


async def _emit_vln_progress(
    ctx,
    *,
    step: str,
    summary: str,
    progress: float,
    planner: dict[str, Any],
    command: PrimitiveCommand | None,
    primitive_result: Any | None = None,
) -> None:
    progress_fn = getattr(ctx, "progress", None)
    if progress_fn is None:
        return
    await progress_fn(
        phase="executing",
        step=step,
        summary=summary,
        progress=progress,
        metadata={
            "ux": {
                "skill": "vln_navigation",
                "planner": dict(planner),
                "primitive": command.primitive if command is not None else None,
                "arguments": dict(command.arguments) if command is not None else None,
                "primitive_result": primitive_result,
            }
        },
    )


async def _emit_follow_progress(
    ctx,
    *,
    phase: str,
    summary: str,
    progress: float,
    observation: RobotObservation | None,
    target=None,
    detections=None,
    command: VelocityCommand | None = None,
    mode: str | None = None,
    reason: str | None = None,
) -> None:
    progress_fn = getattr(ctx, "progress", None)
    if progress_fn is None:
        return
    detections = list(detections or [])
    target_bbox = getattr(target, "bbox", None)
    target_center = getattr(target, "center", None)
    target_area = getattr(target, "area", None)
    metadata = {
        "ux": {
            "skill": "human_follow",
            "phase": phase,
            "mode": mode or phase,
            "target_id": getattr(target, "id", None),
            "bbox": list(target_bbox) if target_bbox else None,
            "center": list(target_center) if target_center else None,
            "area": target_area,
            "confidence": getattr(target, "confidence", None),
            "detections": len(detections),
            "command": {
                "vx": command.vx,
                "vy": command.vy,
                "wz": command.vz,
            }
            if command is not None
            else None,
            "frame_id": observation.frame_id if observation else None,
            "camera": _first_camera(observation),
            "reason": reason,
        }
    }
    await progress_fn(
        phase="executing",
        step=phase,
        summary=summary,
        progress=progress,
        metadata=metadata,
    )


def _first_camera(observation: RobotObservation | None) -> str | None:
    if observation is None or not observation.images:
        return None
    return observation.images[0].camera
