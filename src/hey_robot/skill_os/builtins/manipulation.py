from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from hey_robot.protocol import RobotObservation
from hey_robot.skill_os.base import BaseSkill, SkillResult
from hey_robot.skill_os.builtins.common import spec
from hey_robot.skill_os.builtins.manipulation_adapter import vla_output_to_primitives
from hey_robot.vla.so101_schema import (
    SO101_STATE_SCHEMA,
    state_from_arm_status,
)

_SO101_STATE_SCHEMA = "so101_single_arm_rad_gripper01"


class SetArmPoseSkill(BaseSkill):
    spec = spec(
        "set_arm_pose",
        "Move the arm to a named verified pose.",
        category="arm",
        input_schema={
            "type": "object",
            "properties": {"pose_name": {"type": "string"}},
            "required": ["pose_name"],
        },
        required_resources=("arm",),
        supported_robots=("xlerobot", "so101", "so101_mobile"),
        driver_primitives=("set_arm_pose",),
        safety_level="motion",
        timeout_sec=12.0,
        agent_visible=False,
        goal_effects=("sets_arm_named_pose",),
        evidence_outputs=("arm_pose_action_result",),
        cannot_satisfy=("weak_scene_observation",),
    )

    async def execute(self, ctx, arguments):
        await ctx.robot.set_arm_pose(**arguments)
        return SkillResult(success=True, summary="Arm pose set.")


class MoveArmJointsSkill(BaseSkill):
    spec = spec(
        "move_arm_joints",
        "Set multiple arm joints. Use mode=delta for relative movement.",
        category="arm",
        input_schema={
            "type": "object",
            "properties": {
                "joints": {"type": "object"},
                "mode": {"type": "string"},
            },
            "required": ["joints"],
        },
        required_resources=("arm",),
        supported_robots=("xlerobot", "so101", "so101_mobile"),
        driver_primitives=("move_arm_joints",),
        safety_level="motion",
        timeout_sec=10.0,
        agent_visible=False,
        goal_effects=("changes_arm_joint_positions",),
        evidence_outputs=("arm_joint_action_result",),
        cannot_satisfy=("weak_scene_observation",),
    )

    async def execute(self, ctx, arguments):
        await ctx.robot.move_arm_joints(**arguments)
        return SkillResult(success=True, summary="Arm joints moved.")


class SetGripperSkill(BaseSkill):
    spec = spec(
        "set_gripper",
        "Set gripper opening. Use opening_pct or action=open/close.",
        category="gripper",
        input_schema={
            "type": "object",
            "properties": {
                "opening_pct": {"type": "number"},
                "action": {"type": "string"},
            },
        },
        required_resources=("gripper",),
        supported_robots=("xlerobot", "so101", "so101_mobile"),
        driver_primitives=("set_gripper",),
        safety_level="motion",
        timeout_sec=10.0,
        agent_visible=False,
        goal_effects=("changes_gripper_opening",),
        evidence_outputs=("gripper_action_result",),
        cannot_satisfy=("weak_scene_observation",),
    )

    async def execute(self, ctx, arguments):
        await ctx.robot.set_gripper(**arguments)
        return SkillResult(success=True, summary="Gripper command completed.")


class _ManipulateSkillBase(BaseSkill):
    """Base class for VLA-driven manipulation skills.

    Runs a control loop in Skill OS:
      1. Capture current observation
      2. Call VLA model service (stateless, single-frame inference)
      3. Parse VLA output → arm primitives
      4. Execute primitives on robot
      5. Repeat until task_done or max_steps reached
    """

    async def execute(self, ctx, arguments):
        if ctx.model_services is None:
            return SkillResult(
                success=False,
                summary=f"{self.spec.name} requires a VLA model service",
                status="failed",
                failure_mode="model_service_unavailable",
                error="model service port is unavailable",
            )
        if ctx.robot is None:
            return SkillResult(
                success=False,
                summary="robot runtime is required for VLA manipulation",
                status="failed",
                failure_mode="robot_runtime_unavailable",
                error="robot runtime port is unavailable",
            )

        max_steps = max(1, int(arguments.get("max_steps") or 30))
        task_prompt = str(
            arguments.get("task_prompt") or arguments.get("objective") or self.spec.name
        )
        steps: list[dict[str, Any]] = []
        service_name = self.spec.required_model_service

        for step_index in range(max_steps):
            payload = _vla_payload(ctx, arguments)
            payload.update(
                {
                    "skill_name": self.spec.name,
                    "task_prompt": task_prompt,
                    "vla_step": step_index,
                    "policy_session_id": payload.get("policy_session_id")
                    or getattr(ctx, "skill_id", None),
                }
            )
            result = await ctx.model_services.call(
                service_name,
                payload,
            )
            if not bool(getattr(result, "success", False)):
                return SkillResult(
                    success=False,
                    summary=str(
                        getattr(result, "summary", "") or "VLA inference failed"
                    ),
                    status=str(getattr(result, "status", "") or "failed"),
                    failure_mode=getattr(result, "failure_mode", None)
                    or "vla_inference_failed",
                    error=getattr(result, "error", None),
                    data={"steps": steps},
                )

            vla_data = _extract_vla_policy_data(result)

            primitives = vla_output_to_primitives(vla_data)

            await self._emit_progress(
                ctx,
                step="inference",
                summary=f"VLA step {step_index + 1}/{max_steps}",
                progress=min(0.9, 0.2 + step_index * (0.7 / max(1, max_steps))),
                vla=vla_data,
                primitives=primitives,
            )

            for prim in primitives:
                step = await self._execute_primitive(ctx, prim)
                steps.append(step)
                if not step["success"]:
                    return SkillResult(
                        success=False,
                        summary=step["message"],
                        status="failed",
                        failure_mode="primitive_execution_failed",
                        error=step.get("error"),
                        data={"vla": vla_data, "steps": steps},
                    )

            if _vla_task_done(vla_data):
                return SkillResult(
                    success=True,
                    summary=f"{self.spec.name} completed in {step_index + 1} steps",
                    data={"vla": vla_data, "steps": steps},
                )

        return SkillResult(
            success=False,
            status="failed",
            failure_mode="vla_max_steps_exhausted",
            summary=f"{self.spec.name} reached max steps without task_done",
            data={"steps": steps},
        )

    async def _execute_primitive(self, ctx, prim) -> dict[str, Any]:
        try:
            method = getattr(ctx.robot, prim.primitive)
            result = await method(**prim.arguments)
        except Exception as exc:
            return {
                "success": False,
                "primitive": prim.primitive,
                "arguments": dict(prim.arguments),
                "reason": prim.reason,
                "message": str(exc),
                "error": str(exc),
            }
        return {
            "success": True,
            "primitive": prim.primitive,
            "arguments": dict(prim.arguments),
            "reason": prim.reason,
            "message": f"{prim.primitive} completed",
            "result": result,
        }

    async def _emit_progress(
        self, ctx, *, step, summary, progress, vla, primitives
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
                    "skill": self.spec.name,
                    "vla": vla,
                    "primitives": [
                        {"primitive": p.primitive, "arguments": dict(p.arguments)}
                        for p in primitives
                    ],
                }
            },
        )


def _vla_payload(ctx: Any, arguments: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in dict(arguments).items()
        if key not in {"max_steps", "execute_primitives"}
    }
    if "observation" not in payload and "image_path" not in payload:
        observation = ctx.current_observation() if ctx.current_observation else None
        resolve_images = getattr(ctx, "resolve_images", None)
        observation_payload = _observation_payload(
            observation,
            camera=None,  # Send ALL cameras — VLA models need multiple views
            resolve_images=resolve_images,
        )
        if observation_payload is not None:
            payload["observation"] = observation_payload
    return payload


def _observation_payload(
    observation: RobotObservation | None,
    *,
    camera: object | None = None,
    resolve_images: Any = None,
) -> dict[str, Any] | None:
    if observation is None:
        return None
    images = observation.images
    if camera:
        preferred = [image for image in images if image.camera == str(camera)]
        if preferred:
            images = preferred
    image_dicts = _encode_images(images, resolve_images)
    raw = dict(observation.raw)
    payload = {
        "frame_id": observation.frame_id,
        "timestamp": observation.envelope.timestamp,
        "images": image_dicts,
        "proprioception": list(observation.proprioception),
        "raw": raw,
    }
    vla_state = state_from_arm_status(raw)
    if vla_state is not None:
        payload["state"] = vla_state
        payload["state_schema"] = str(raw.get("vla_state_schema") or SO101_STATE_SCHEMA)
        payload["active_arm"] = str(raw.get("active_arm") or "right")
    return payload


def _encode_images(
    images: list[Any], resolve_images: Any = None
) -> list[dict[str, Any]]:
    import base64
    import io

    import numpy as np
    from PIL import Image

    if resolve_images is not None:
        try:
            arrays = resolve_images(images)
            if len(arrays) == len(images):
                result: list[dict[str, Any]] = []
                for ref, arr in zip(images, arrays, strict=False):
                    if not isinstance(arr, np.ndarray):
                        result.append(asdict(ref))
                        continue
                    pil = Image.fromarray(arr)
                    buf = io.BytesIO()
                    pil.save(buf, format="JPEG", quality=85)
                    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                    entry = asdict(ref)
                    entry["data"] = b64
                    entry["format"] = "jpeg"
                    result.append(entry)
                return result
        except Exception:
            import logging

            logging.getLogger(__name__).debug(
                "Failed to resolve images via resolve_images", exc_info=True
            )
    # Fallback: try to read from file URIs
    result = []
    for ref in images:
        entry = asdict(ref)
        uri = str(getattr(ref, "uri", "") or "")
        if uri.startswith("media://local/"):
            rel = uri[len("media://local/") :]
            candidate = Path(rel)
            if candidate.is_file():
                try:
                    import base64 as _b64

                    entry["data"] = _b64.b64encode(candidate.read_bytes()).decode(
                        "ascii"
                    )
                    entry["format"] = "jpeg"
                except Exception:
                    import logging

                    logging.getLogger(__name__).debug(
                        "Failed to read media file from %s", candidate, exc_info=True
                    )
        result.append(entry)
    return result


def _extract_vla_policy_data(result: Any) -> dict[str, Any]:
    metrics = dict(getattr(result, "metrics", {}) or {})
    vla = metrics.get("vla")
    data = dict(vla) if isinstance(vla, dict) else {}
    for key in ("policy_result", "action_chunk"):
        value = metrics.get(key)
        if isinstance(value, dict):
            data[key] = dict(value)
    policy_result = data.get("policy_result")
    if isinstance(policy_result, dict) and policy_result.get("kind") == "action_chunk":
        data.setdefault("task_done", bool(policy_result.get("done", False)))
    return data


def _vla_task_done(vla_data: dict[str, Any]) -> bool:
    if bool(vla_data.get("task_done")):
        return True
    policy_result = vla_data.get("policy_result")
    if isinstance(policy_result, dict):
        return bool(policy_result.get("done", False))
    action_chunk = vla_data.get("action_chunk")
    if isinstance(action_chunk, dict):
        return bool(action_chunk.get("done", False))
    return False


class ManipulateSkill(_ManipulateSkillBase):
    spec = spec(
        "manipulate",
        "Run the deployed manipulation policy for a natural-language arm task.",
        category="manipulation",
        input_schema={
            "type": "object",
            "properties": {
                "task_prompt": {"type": "string"},
                "objective": {"type": "string"},
                "arm": {"type": "string"},
                "camera": {"type": "string"},
                "execution_time": {"type": "number"},
                "max_steps": {"type": "integer"},
            },
            "required": [],
        },
        required_resources=("arm", "gripper", "camera"),
        dependencies=("inspect_scene",),
        driver_primitives=("move_arm_joints", "set_gripper", "stop_motion"),
        required_model_service="manipulate",
        safety_level="motion",
        timeout_sec=60.0,
        agent_visible=True,
        feedback_mode="vision",
        goal_effects=("manipulates_object",),
        evidence_outputs=("vla_policy_result", "arm_action_result"),
        cannot_satisfy=("weak_scene_observation",),
    )
