from __future__ import annotations

from hey_robot.skills.base import BaseSkill, SkillResult
from hey_robot.skills.builtin.common import spec


class StopMotionSkill(BaseSkill):
    spec = spec(
        "stop_motion",
        "Stop robot motion. Set emergency=true for emergency stop.",
        category="safety",
        input_schema={
            "type": "object",
            "properties": {"emergency": {"type": "boolean"}},
        },
        required_resources=("base", "arm"),
        driver_primitives=("stop_motion",),
        safety_level="stop",
        timeout_sec=8.0,
        feedback_mode="none",
        capability_type="stop_motion",
        goal_effects=("stops_robot_motion",),
        evidence_outputs=("stop_motion_result",),
    )

    async def execute(self, ctx, arguments):
        await ctx.robot.stop_motion(**arguments)
        return SkillResult(success=True, summary="Motion stopped.")


class ResetPostureSkill(BaseSkill):
    spec = spec(
        "reset_posture",
        "Stop motion and return the robot to a safe default posture.",
        category="safety",
        input_schema={"type": "object", "properties": {}},
        required_resources=("base", "arm", "gripper"),
        driver_primitives=("reset_posture",),
        safety_level="stop",
        timeout_sec=15.0,
        feedback_mode="none",
        capability_type="reset_posture",
        goal_effects=("restores_safe_posture",),
        evidence_outputs=("reset_posture_result",),
    )

    async def execute(self, ctx, arguments):
        await ctx.robot.reset_posture(**arguments)
        return SkillResult(success=True, summary="Robot posture reset.")
