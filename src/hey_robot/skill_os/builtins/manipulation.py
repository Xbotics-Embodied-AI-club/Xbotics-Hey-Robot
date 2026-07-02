from __future__ import annotations

from typing import Any

from hey_robot.skill_os.base import BaseSkill, SkillResult
from hey_robot.skill_os.builtins.common import spec
from hey_robot.skill_os.builtins.manipulation_adapter import vla_output_to_primitives


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
        driver_primitives=("set_arm_pose",),
        safety_level="motion",
        timeout_sec=12.0,
        agent_visible=False,
        capability_type="arm_pose",
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
        driver_primitives=("move_arm_joints",),
        safety_level="motion",
        timeout_sec=10.0,
        agent_visible=False,
        capability_type="arm_joint_delta",
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
        driver_primitives=("set_gripper",),
        safety_level="motion",
        timeout_sec=10.0,
        agent_visible=False,
        capability_type="gripper_control",
        goal_effects=("changes_gripper_opening",),
        evidence_outputs=("gripper_action_result",),
        cannot_satisfy=("weak_scene_observation",),
    )

    async def execute(self, ctx, arguments):
        await ctx.robot.set_gripper(**arguments)
        return SkillResult(success=True, summary="Gripper command completed.")


class _VLAManipulationSkill(BaseSkill):
    """Base class for VLA-driven manipulation skills.

    Runs a control loop in Skill OS:
      1. Capture current observation
      2. Call VLA model service (stateless, single-frame inference)
      3. Parse VLA output → arm primitives
      4. Execute primitives on robot
      5. Repeat until task_done or max_steps reached
    """

    capability_name: str = ""

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

        for step_index in range(max_steps):
            result = await ctx.model_services.call(
                self.capability_name,
                {
                    "skill_name": self.capability_name,
                    "task_prompt": task_prompt,
                    "vla_step": step_index,
                },
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

            vla_data = dict(getattr(result, "metrics", {}) or {}).get("vla", {})
            if not isinstance(vla_data, dict):
                vla_data = {}

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

            if vla_data.get("task_done"):
                return SkillResult(
                    success=True,
                    summary=f"{self.spec.name} completed in {step_index + 1} steps",
                    data={"vla": vla_data, "steps": steps},
                )

        return SkillResult(
            success=True,
            summary=f"{self.spec.name} reached max steps ({max_steps})",
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
                    "skill": self.capability_name,
                    "vla": vla,
                    "primitives": [
                        {"primitive": p.primitive, "arguments": dict(p.arguments)}
                        for p in primitives
                    ],
                }
            },
        )


class PickObjectSkill(_VLAManipulationSkill):
    capability_name = "pick_object"
    spec = spec(
        "pick_object",
        "Pick up an object using the VLA manipulation policy.",
        category="manipulation",
        input_schema={
            "type": "object",
            "properties": {
                "task_prompt": {"type": "string"},
                "objective": {"type": "string"},
                "camera": {"type": "string"},
                "max_steps": {"type": "integer"},
            },
            "required": ["task_prompt"],
        },
        required_resources=("arm", "gripper", "camera"),
        dependencies=("inspect_scene",),
        driver_primitives=("move_arm_joints", "set_gripper", "stop_motion"),
        required_model_service="vla_manipulation",
        safety_level="motion",
        timeout_sec=60.0,
        agent_visible=True,
        feedback_mode="vision",
        capability_type="object_pick",
        goal_effects=("grasps_object",),
        evidence_outputs=("vla_policy_result", "arm_action_result"),
        cannot_satisfy=("weak_scene_observation",),
    )


class PlaceObjectSkill(_VLAManipulationSkill):
    capability_name = "place_object"
    spec = spec(
        "place_object",
        "Place a held object at a target location using the VLA manipulation policy.",
        category="manipulation",
        input_schema={
            "type": "object",
            "properties": {
                "task_prompt": {"type": "string"},
                "objective": {"type": "string"},
                "camera": {"type": "string"},
                "max_steps": {"type": "integer"},
            },
            "required": ["task_prompt"],
        },
        required_resources=("arm", "gripper", "camera"),
        dependencies=("inspect_scene",),
        driver_primitives=("move_arm_joints", "set_gripper", "stop_motion"),
        required_model_service="vla_manipulation",
        safety_level="motion",
        timeout_sec=60.0,
        agent_visible=True,
        feedback_mode="vision",
        capability_type="object_place",
        goal_effects=("places_object",),
        evidence_outputs=("vla_policy_result", "arm_action_result"),
        cannot_satisfy=("weak_scene_observation",),
    )


class VLAManipulationSkill(_VLAManipulationSkill):
    """Backward-compatible generic VLA manipulation skill.

    Accepts any task_prompt and delegates to the VLA policy service.
    Use pick_object / place_object for semantic skill names.
    """

    capability_name = "vla_manipulation"
    spec = spec(
        "vla_manipulation",
        "Run the deployed VLA policy for a natural-language arm manipulation task.",
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
            "required": ["task_prompt"],
        },
        required_resources=("arm", "gripper", "camera"),
        dependencies=("inspect_scene",),
        driver_primitives=("move_arm_joints", "set_gripper", "stop_motion"),
        required_model_service="vla_manipulation",
        safety_level="motion",
        timeout_sec=60.0,
        agent_visible=True,
        feedback_mode="vision",
        capability_type="vla_manipulation",
        goal_effects=("manipulates_object",),
        evidence_outputs=("vla_policy_result", "arm_action_result"),
        cannot_satisfy=("weak_scene_observation",),
    )
