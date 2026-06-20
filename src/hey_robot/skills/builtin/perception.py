from __future__ import annotations

from hey_robot.skills.base import BaseSkill, SkillResult
from hey_robot.skills.builtin.common import spec


class InspectSceneSkill(BaseSkill):
    spec = spec(
        "inspect_scene",
        "Inspect the current scene and return grounded visual evidence.",
        category="perception",
        input_schema={
            "type": "object",
            "properties": {"question": {"type": "string"}},
        },
        required_resources=("camera",),
        driver_primitives=("inspect_scene",),
        safety_level="observe",
        timeout_sec=8.0,
        feedback_mode="vision",
    )

    async def execute(self, ctx, arguments):
        result = await ctx.perception.inspect_scene(**arguments)
        summary = str(
            result.get("summary") or result.get("message") or "scene inspected"
        )
        return SkillResult(
            success=bool(result.get("success", True)),
            summary=summary,
            failure_mode=result.get("failure_mode"),
            error=None
            if result.get("success", True)
            else str(result.get("message") or summary),
            data=dict(result),
        )


class LookAroundSkill(BaseSkill):
    spec = spec(
        "look_around",
        "Collect visual evidence from multiple short viewing directions.",
        category="perception",
        input_schema={
            "type": "object",
            "properties": {"question": {"type": "string"}},
        },
        required_resources=("camera", "base"),
        driver_primitives=("look_around", "turn_base"),
        safety_level="observe",
        timeout_sec=30.0,
        feedback_mode="vision",
    )

    async def execute(self, ctx, arguments):
        result = await ctx.perception.look_around(**arguments)
        summary = str(
            result.get("summary") or result.get("message") or "look around completed"
        )
        return SkillResult(
            success=bool(result.get("success", True)),
            summary=summary,
            failure_mode=result.get("failure_mode"),
            error=None
            if result.get("success", True)
            else str(result.get("message") or summary),
            data=dict(result),
        )


class DetectMarkerSkill(BaseSkill):
    spec = spec(
        "detect_marker",
        "Detect a workspace marker in the current camera frame.",
        category="perception",
        input_schema={
            "type": "object",
            "properties": {"marker_id": {"type": "integer"}},
        },
        required_resources=("camera",),
        driver_primitives=("detect_marker",),
        safety_level="observe",
        timeout_sec=6.0,
        agent_visible=False,
        feedback_mode="vision",
    )

    async def execute(self, ctx, arguments):
        result = await ctx.perception.detect_marker(**arguments)
        summary = str(
            result.get("summary")
            or result.get("message")
            or "marker detection completed"
        )
        return SkillResult(
            success=bool(result.get("success", True)),
            summary=summary,
            failure_mode=result.get("failure_mode"),
            error=None
            if result.get("success", True)
            else str(result.get("message") or summary),
            data=dict(result),
        )
