from __future__ import annotations

from hey_robot.skills.base import BaseSkill, SkillResult
from hey_robot.skills.builtin.common import spec


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
        # The timeout includes controller scheduling, bus delivery, simulator
        # settling, and the success status round-trip. Five seconds is too
        # tight under normal MuJoCo load and can race a completed jaw motion.
        timeout_sec=10.0,
        agent_visible=False,
    )

    async def execute(self, ctx, arguments):
        await ctx.robot.set_gripper(**arguments)
        return SkillResult(success=True, summary="Gripper command completed.")
