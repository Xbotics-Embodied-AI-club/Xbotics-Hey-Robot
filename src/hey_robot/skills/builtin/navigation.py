from __future__ import annotations

import asyncio
import time

from hey_robot.perception.human_follow import (
    FollowController,
    TargetTracker,
    VelocityCommand,
    detect_people,
    load_detector,
)
from hey_robot.protocol import RobotObservation
from hey_robot.skills.base import BaseSkill, SkillResult
from hey_robot.skills.builtin.common import spec


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
    )

    async def execute(self, ctx, arguments):
        await ctx.robot.base_velocity_step(**arguments)
        return SkillResult(success=True, summary="Base velocity step completed.")


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
        observation = ctx.observation
        if observation is None:
            return SkillResult(
                success=False,
                summary="human follow requires a current observation",
                failure_mode="observation_unavailable",
                error="current observation is unavailable",
            )
        if ctx.resolve_images is None:
            return SkillResult(
                success=False,
                summary="human follow image resolver is unavailable",
                failure_mode="image_unavailable",
                error="image resolver is unavailable",
            )
        load_detector(str(arguments.get("model_path") or "models/yolo26n.pt"))
        await _emit_follow_progress(
            ctx,
            phase="starting",
            summary="preparing human follow",
            progress=0.05,
            observation=observation,
        )
        duration_raw = arguments.get("duration_sec")
        duration = (
            min(3600.0, max(1.0, float(duration_raw)))
            if duration_raw is not None
            else None
        )
        max_steps = int(arguments.get("max_steps") or 0)
        if duration is None and max_steps <= 0:
            max_steps = 300
        deadline = time.monotonic() + duration if duration is not None else None
        tracker = TargetTracker(
            max_age=int(arguments.get("max_tracking_age") or 30),
            min_iou=float(arguments.get("min_iou_threshold") or 0.3),
        )
        controller = FollowController(
            target_distance=float(arguments.get("target_distance_m") or 0.7),
            target_width_ratio=float(arguments.get("target_width_ratio") or 0.35),
            target_height_ratio=float(arguments.get("target_height_ratio") or 1.0),
            kp_linear=float(arguments.get("kp_linear") or 0.35),
            kp_angular=float(arguments.get("kp_angular") or 1.0),
            max_linear_speed=float(arguments.get("max_linear_speed") or 0.3),
            max_backward_speed=float(arguments.get("max_backward_speed") or 0.2),
            allow_backward=bool(arguments.get("allow_backward", True)),
            max_angular_speed=float(arguments.get("max_angular_speed") or 1.0),
            dead_zone_x=float(arguments.get("dead_zone_x") or 0.15),
            dead_zone_area=float(arguments.get("dead_zone_area") or 0.1),
        )
        current_velocity = VelocityCommand(0.0, 0.0, 0.0)
        steps: list[dict] = []
        last_processed_frame_id: int | None = None
        try:
            while (deadline is None or time.monotonic() < deadline) and (
                max_steps <= 0 or len(steps) < max_steps
            ):
                observation = await _refresh_observation(ctx, observation)
                frame_id = getattr(observation, "frame_id", None)
                if frame_id is not None and frame_id == last_processed_frame_id:
                    await asyncio.sleep(0.01)
                    continue
                last_processed_frame_id = frame_id
                image = _resolve_follow_image(
                    ctx,
                    observation,
                    camera=arguments.get("camera"),
                )
                detections = detect_people(image)
                target = tracker.update(detections)
                width, height = _frame_size(image)
                command = controller.compute_velocity(
                    target, frame_width=width, frame_height=height
                )
                mode = "following"
                if command is None:
                    if controller.is_searching():
                        command = controller.compute_search_velocity()
                        mode = "searching"
                        await _emit_follow_progress(
                            ctx,
                            phase="searching",
                            summary="target temporarily lost; searching",
                            progress=0.35,
                            observation=observation,
                            detections=detections,
                            command=command,
                            mode=mode,
                        )
                    else:
                        await _emit_follow_progress(
                            ctx,
                            phase="acquiring",
                            summary="looking for a person to follow",
                            progress=0.2,
                            observation=observation,
                            detections=detections,
                            mode="acquiring",
                        )
                        await asyncio.sleep(0.02)
                        continue
                if controller.is_target_lost():
                    await ctx.robot.stop_motion()
                    await _emit_follow_progress(
                        ctx,
                        phase="lost",
                        summary="person lost during human follow",
                        progress=0.0,
                        observation=observation,
                        detections=detections,
                        command=VelocityCommand(0.0, 0.0, 0.0),
                        mode="lost",
                        reason="person_lost",
                    )
                    return SkillResult(
                        success=False,
                        summary="person lost during human follow",
                        failure_mode="person_lost",
                        error="person lost during human follow",
                        data={"steps": steps, "mode": "lost"},
                    )
                current_velocity = controller.smooth_velocity(
                    current_velocity, command, alpha=0.3
                )
                await _emit_follow_progress(
                    ctx,
                    phase=mode,
                    summary="following target" if mode == "following" else mode,
                    progress=0.6 if mode == "following" else 0.4,
                    observation=observation,
                    target=target,
                    detections=detections,
                    command=current_velocity,
                    mode=mode,
                )
                applied_steps = await _apply_follow_velocity(ctx, current_velocity)
                for step in applied_steps:
                    step["mode"] = mode
                steps.extend(applied_steps)
                if (
                    target is not None
                    and abs(current_velocity.vx) < 0.02
                    and abs(current_velocity.vz) < 0.02
                ):
                    steps.append(
                        {
                            "success": True,
                            "skill": "human_follow",
                            "message": "target already within follow window",
                            "mode": mode,
                        }
                    )
                    if duration is not None:
                        break
                if max_steps > 0 and len(steps) >= max_steps:
                    break
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            await ctx.robot.stop_motion()
            await _emit_follow_progress(
                ctx,
                phase="interrupted",
                summary="human follow interrupted; motion stopped",
                progress=0.0,
                observation=observation,
                command=VelocityCommand(0.0, 0.0, 0.0),
                mode="interrupted",
                reason="cancelled",
            )
            raise
        await ctx.robot.stop_motion()
        await _emit_follow_progress(
            ctx,
            phase="completed" if duration is not None or max_steps > 0 else "stopped",
            summary="human follow completed"
            if duration is not None or max_steps > 0
            else "human follow stopped",
            progress=1.0,
            observation=observation,
            command=VelocityCommand(0.0, 0.0, 0.0),
            mode="completed" if duration is not None or max_steps > 0 else "running",
        )
        return SkillResult(
            success=True,
            summary="human follow completed"
            if duration is not None or max_steps > 0
            else "human follow stopped",
            data={
                "steps": steps,
                "mode": "completed"
                if duration is not None or max_steps > 0
                else "running",
            },
        )


async def _refresh_observation(ctx, observation: RobotObservation) -> RobotObservation:
    # Fast path: read latest frame directly from the driver's continuous stream.
    # The driver observation loop publishes frames via the message bus;
    # _on_observation keeps state.latest_observation current. No need for
    # an expensive sub-skill call just to get a fresh image for detection.
    if ctx.current_observation is not None:
        latest = ctx.current_observation()
        if isinstance(latest, RobotObservation):
            return latest
    # Slow fallback: only when the driver stream has not started yet.
    if ctx.invoke is not None:
        await ctx.invoke("inspect_scene", {})
        if ctx.current_observation is not None:
            latest = ctx.current_observation()
            if isinstance(latest, RobotObservation):
                return latest
    return observation


def _resolve_follow_image(ctx, observation: RobotObservation, *, camera: object | None):
    refs = observation.images
    if camera is not None:
        filtered = [ref for ref in refs if ref.camera == str(camera)]
        if filtered:
            refs = filtered
    if not refs or ctx.resolve_images is None:
        return None
    images = ctx.resolve_images(refs[:1])
    if not images:
        return None
    return images[0]


def _frame_size(image) -> tuple[int, int]:
    if image is None:
        return (640, 480)
    height, width = image.shape[:2]
    return width, height


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


async def _apply_follow_velocity(ctx, command: VelocityCommand) -> list[dict]:
    steps: list[dict] = []
    if abs(command.vx) < 0.002 and abs(command.vz) < 0.0002:
        return steps
    await ctx.robot.base_velocity_step(
        vx=command.vx,
        vy=command.vy,
        wz=command.vz,
        duration_ms=400,
    )
    steps.append(
        {
            "success": True,
            "skill": "base_velocity_step",
            "message": "follow velocity step",
            "command": {"vx": command.vx, "vy": command.vy, "wz": command.vz},
        }
    )
    return steps
