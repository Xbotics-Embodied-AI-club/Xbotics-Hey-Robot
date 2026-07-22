from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from hey_robot.protocol import RobotObservation
from hey_robot.skill_os.base import BaseSkill, SkillResult
from hey_robot.skill_os.builtins.common import spec
from hey_robot.skill_os.builtins.manipulation_adapter import vla_output_to_primitives
from hey_robot.skill_os.termination import (
    FixedHorizonTerminationEvaluator,
    TerminationState,
)
from hey_robot.vla.so101_schema import (
    SO101_STATE_SCHEMA,
    state_from_arm_status,
)


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
    """VLA 驱动操作技能的基类。

    在 Skill OS 中运行控制循环：
      1. 获取当前观测
      2. 调用 VLA 模型服务（无状态、单帧推理）
      3. 将 VLA 输出解析为机械臂 primitive
      4. 在机器人上执行 primitive
      5. 重复直到 task_done 或达到 max_steps
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

        model_settings = dict(getattr(ctx, "model_settings", None) or {})
        max_steps = max(
            1,
            int(
                arguments.get("max_steps") or model_settings.get("option_horizon") or 30
            ),
        )
        task_prompt = str(
            arguments.get("task_prompt") or arguments.get("objective") or self.spec.name
        )
        steps: list[dict[str, Any]] = []
        service_name = self.spec.required_model_service
        last_vla: dict[str, Any] = {}
        before_frame_id: int | None = None
        after_frame_id: int | None = None
        executed_action = False
        fresh_observation_timeout_sec = max(
            0.0, float(arguments.get("fresh_observation_timeout_sec") or 2.0)
        )

        for step_index in range(max_steps):
            observation = _current_observation(ctx)
            before_frame_id = _frame_id(observation)
            payload = _vla_payload(ctx, arguments, observation=observation)
            policy_session_id = _policy_session_id(
                observation, fallback=getattr(ctx, "skill_id", None)
            )
            payload.update(
                {
                    "skill_name": self.spec.name,
                    "task_prompt": task_prompt,
                    "agent_subgoal": task_prompt,
                    "vla_step": step_index,
                    "policy_session_id": payload.get("policy_session_id")
                    or policy_session_id,
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
            last_vla = vla_data

            native_action = _native_action(vla_data)
            if native_action is not None:
                action_result = await ctx.robot.apply_policy_action(
                    native_action["values"],
                    expected_frame_id=native_action["expected_frame_id"],
                    raw_values=native_action.get("raw_values"),
                )
                steps.append(
                    {
                        "success": bool(action_result.get("success", False)),
                        "primitive": "embodiment_native_action",
                        "arguments": {
                            "dimensions": len(native_action["values"]),
                            "expected_frame_id": native_action["expected_frame_id"],
                        },
                        "result": dict(action_result),
                    }
                )
                executed_action = True
                after_frame_id = int(
                    action_result.get("frame_id") or native_action["expected_frame_id"]
                )
                if not bool(action_result.get("done", False)):
                    fresh_observation = await _wait_for_fresh_observation(
                        ctx,
                        after_frame_id=before_frame_id,
                        timeout_sec=fresh_observation_timeout_sec,
                    )
                    if (
                        ctx.current_observation is not None
                        and before_frame_id is not None
                    ):
                        if fresh_observation is None:
                            return _termination_result(
                                success=False,
                                state=TerminationState.FAILED,
                                reason="observation_stale",
                                steps=steps,
                                last_vla=last_vla,
                                before_frame_id=before_frame_id,
                                after_frame_id=after_frame_id,
                                episode_done=False,
                            )
                        after_frame_id = int(fresh_observation.frame_id)
                decision = FixedHorizonTerminationEvaluator(max_steps).evaluate(
                    steps_executed=step_index + 1,
                    policy_result=native_action,
                    action_result=action_result,
                )
                if decision.state is TerminationState.FAILED:
                    return _termination_result(
                        success=False,
                        state=decision.state,
                        reason=decision.reason,
                        steps=steps,
                        last_vla=last_vla,
                        before_frame_id=before_frame_id,
                        after_frame_id=after_frame_id,
                        episode_done=bool(action_result.get("done", False)),
                    )
                if decision.state in {
                    TerminationState.SUCCESS,
                    TerminationState.UNKNOWN,
                }:
                    return _termination_result(
                        success=not bool(action_result.get("done", False)),
                        state=decision.state,
                        reason=decision.reason,
                        steps=steps,
                        last_vla=last_vla,
                        before_frame_id=before_frame_id,
                        after_frame_id=after_frame_id,
                        episode_done=bool(action_result.get("done", False)),
                    )
                continue

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
                        data={
                            "vla": vla_data,
                            "steps": steps,
                            "option_state": "failed",
                            "termination_reason": "primitive_execution_failed",
                            "root_task_success": None,
                            "episode_done": None,
                            "requires_reobservation": True,
                            "before_frame_id": before_frame_id,
                            "after_frame_id": after_frame_id,
                        },
                    )
                executed_action = True

            if primitives:
                fresh_observation = await _wait_for_fresh_observation(
                    ctx,
                    after_frame_id=before_frame_id,
                    timeout_sec=fresh_observation_timeout_sec,
                )
                if ctx.current_observation is not None and before_frame_id is not None:
                    if fresh_observation is None:
                        return SkillResult(
                            success=False,
                            summary=(
                                f"{self.spec.name} executed an action but no fresh "
                                "observation arrived"
                            ),
                            status="failed",
                            failure_mode="observation_stale",
                            error=(
                                "current observation did not advance beyond frame "
                                f"{before_frame_id} within "
                                f"{fresh_observation_timeout_sec:.2f}s"
                            ),
                            data={
                                "vla": vla_data,
                                "steps": steps,
                                "option_state": "failed",
                                "termination_reason": "observation_stale",
                                "root_task_success": None,
                                "episode_done": None,
                                "requires_reobservation": True,
                                "before_frame_id": before_frame_id,
                                "after_frame_id": None,
                            },
                        )
                    after_frame_id = _frame_id(fresh_observation)

            if _vla_task_done(vla_data):
                return SkillResult(
                    success=True,
                    summary=f"{self.spec.name} completed in {step_index + 1} steps",
                    data={
                        "vla": vla_data,
                        "steps": steps,
                        "option_state": "succeeded",
                        "termination_reason": "vla_done",
                        "root_task_success": None,
                        "episode_done": None,
                        "requires_reobservation": executed_action,
                        "before_frame_id": before_frame_id,
                        "after_frame_id": after_frame_id,
                    },
                )

        return SkillResult(
            success=True,
            status="completed",
            summary=(
                f"{self.spec.name} reached its bounded execution limit; "
                "root task completion is not established"
            ),
            data={
                "vla": last_vla,
                "steps": steps,
                "option_state": "boundary_reached",
                "termination_reason": "max_steps",
                "root_task_success": None,
                "episode_done": None,
                "requires_reobservation": executed_action,
                "before_frame_id": before_frame_id,
                "after_frame_id": after_frame_id,
            },
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


def _vla_payload(
    ctx: Any,
    arguments: dict[str, Any],
    *,
    observation: RobotObservation | None,
) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in dict(arguments).items()
        if key
        not in {
            "max_steps",
            "execute_primitives",
            "fresh_observation_timeout_sec",
        }
    }
    if "observation" not in payload and "image_path" not in payload:
        resolve_images = getattr(ctx, "resolve_images", None)
        observation_payload = _observation_payload(
            observation,
            camera=None,  # 发送所有相机；VLA 模型需要多视角
            resolve_images=resolve_images,
        )
        if observation_payload is not None:
            payload["observation"] = observation_payload
    return payload


def _policy_session_id(
    observation: RobotObservation | None,
    *,
    fallback: object | None,
) -> str | None:
    if observation is not None:
        raw = dict(observation.raw)
        trial_id = str(raw.get("trial_id") or "").strip()
        if trial_id:
            return trial_id
        episode_id = str(observation.envelope.episode_id or "").strip()
        if episode_id:
            return episode_id
    value = str(fallback or "").strip()
    return value or None


def _native_action(vla_data: dict[str, Any]) -> dict[str, Any] | None:
    policy_result = vla_data.get("policy_result")
    if (
        not isinstance(policy_result, dict)
        or policy_result.get("kind") != "native_action"
    ):
        return None
    values = policy_result.get("values")
    if not isinstance(values, list) or not values:
        raise ValueError("native policy result must contain a non-empty values list")
    return {
        **policy_result,
        "values": [float(value) for value in values],
        "expected_frame_id": int(policy_result.get("expected_frame_id", 0)),
    }


def _termination_result(
    *,
    success: bool,
    state: TerminationState,
    reason: str,
    steps: list[dict[str, Any]],
    last_vla: dict[str, Any],
    before_frame_id: int | None,
    after_frame_id: int | None,
    episode_done: bool,
) -> SkillResult:
    boundary = state is TerminationState.UNKNOWN and not episode_done
    return SkillResult(
        success=success,
        status="completed" if success else "failed",
        summary=(
            f"manipulate reached a bounded option boundary after {len(steps)} actions"
            if boundary
            else f"manipulate terminated with {state.value}: {reason}"
        ),
        failure_mode=None if success else reason,
        data={
            "vla": last_vla,
            "steps": steps,
            "option_state": "boundary_reached" if boundary else state.value,
            "termination_state": state.value,
            "termination_reason": reason,
            "root_task_success": None,
            "episode_done": episode_done,
            "requires_reobservation": True,
            "before_frame_id": before_frame_id,
            "after_frame_id": after_frame_id,
        },
    )


def _current_observation(ctx: Any) -> RobotObservation | None:
    if ctx.current_observation is not None:
        return cast(RobotObservation | None, ctx.current_observation())
    return cast(RobotObservation | None, getattr(ctx, "observation", None))


def _frame_id(observation: RobotObservation | None) -> int | None:
    if observation is None:
        return None
    return int(observation.frame_id)


async def _wait_for_fresh_observation(
    ctx: Any,
    *,
    after_frame_id: int | None,
    timeout_sec: float,
) -> RobotObservation | None:
    """Wait until feedback is causally newer than the action input frame.

    Runtimes without an observation callback are kept compatible: they cannot
    provide a live freshness guarantee, so the caller skips the barrier.
    """
    if ctx.current_observation is None or after_frame_id is None:
        return None
    deadline = time.monotonic() + timeout_sec
    while True:
        observation = cast(RobotObservation | None, ctx.current_observation())
        frame_id = _frame_id(observation)
        if frame_id is not None and frame_id > after_frame_id:
            return observation
        if time.monotonic() >= deadline:
            return None
        await asyncio.sleep(min(0.02, max(0.001, deadline - time.monotonic())))


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
                    pil.save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                    entry = asdict(ref)
                    entry["data"] = b64
                    entry["format"] = "png"
                    result.append(entry)
                return result
        except Exception:
            import logging

            logging.getLogger(__name__).debug(
                "Failed to resolve images via resolve_images", exc_info=True
            )
    # 回退：尝试从 file URI 读取
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
                "fresh_observation_timeout_sec": {
                    "type": "number",
                    "default": 2.0,
                },
            },
            "required": [],
        },
        required_resources=("robot_control", "camera"),
        dependencies=("inspect_scene",),
        # Model output adapters select either named driver primitives or the
        # embodiment-native action port. These are alternatives, not a list of
        # primitives every embodiment must implement.
        driver_primitives=(),
        required_model_service="manipulate",
        supported_robots=("xlerobot", "robocasa"),
        safety_level="motion",
        timeout_sec=1800.0,
        agent_visible=True,
        feedback_mode="vision",
        goal_effects=("manipulates_object",),
        evidence_outputs=("vla_policy_result", "arm_action_result"),
        cannot_satisfy=("weak_scene_observation",),
        failure_modes=(
            "model_service_unavailable",
            "robot_runtime_unavailable",
            "vla_inference_failed",
            "primitive_execution_failed",
            "observation_stale",
        ),
    )
