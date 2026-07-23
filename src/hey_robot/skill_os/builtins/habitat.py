from __future__ import annotations

from typing import Any

from hey_robot.skill_os.base import BaseSkill, SkillResult
from hey_robot.skill_os.builtins.common import spec


class _HabitatSkill(BaseSkill):
    async def execute(self, ctx: Any, arguments: dict[str, Any]) -> SkillResult:
        if ctx.robot is None:
            return SkillResult(
                success=False,
                summary="Habitat runtime is unavailable",
                status="failed",
                failure_mode="runtime_unavailable",
            )
        result = await ctx.robot.run(self.spec.name, arguments)
        success = bool(result.get("success", False))
        return SkillResult(
            success=success,
            summary=str(
                result.get("summary") or result.get("message") or self.spec.name
            ),
            status="completed" if success else "failed",
            failure_mode=result.get("failure_mode"),
            error=result.get("error"),
            data=dict(result),
        )


class HabitatNavigateToSkill(_HabitatSkill):
    spec = spec(
        "habitat_navigate_to",
        "Navigate the controlled Habitat agent to a profile-allowed entity or position.",
        category="navigation",
        input_schema={
            "type": "object",
            "properties": {
                "entity_id": {"type": "string"},
                "position": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "max_steps": {"type": "integer", "minimum": 1},
            },
        },
        required_resources=("remote_runtime", "base"),
        supported_robots=("habitat3",),
        driver_primitives=("habitat_navigate_to",),
        safety_level="motion",
        timeout_sec=180.0,
        feedback_mode="status",
        goal_effects=("navigates_to_habitat_target",),
        evidence_outputs=("habitat_skill_trace", "habitat_metrics"),
    )


class HabitatFollowHumanSkill(_HabitatSkill):
    spec = spec(
        "habitat_follow_human",
        "Follow a Habitat human agent at a requested social distance.",
        category="interaction",
        input_schema={
            "type": "object",
            "properties": {
                "human_id": {"type": "string", "default": "agent_1"},
                "distance_m": {"type": "number", "minimum": 0.1},
                "max_steps": {"type": "integer", "minimum": 1},
            },
        },
        required_resources=("remote_runtime", "base", "camera"),
        supported_robots=("habitat3",),
        driver_primitives=("habitat_follow_human",),
        safety_level="motion",
        timeout_sec=300.0,
        feedback_mode="status",
        goal_effects=("follows_habitat_human",),
        evidence_outputs=("social_nav_measure", "habitat_skill_trace"),
    )


class HabitatSymbolicPickSkill(_HabitatSkill):
    spec = spec(
        "habitat_symbolic_pick",
        "Apply a privileged Habitat PDDL pick transition; it is not physical arm execution.",
        category="simulation",
        input_schema={
            "type": "object",
            "properties": {"object_id": {"type": "string"}},
            "required": ["object_id"],
        },
        required_resources=("remote_runtime",),
        supported_robots=("habitat3",),
        driver_primitives=("habitat_symbolic_pick",),
        timeout_sec=30.0,
        feedback_mode="status",
        goal_effects=("symbolically_holds_habitat_target",),
        evidence_outputs=("pddl_action", "habitat_metrics"),
    )


class HabitatSymbolicPlaceSkill(_HabitatSkill):
    spec = spec(
        "habitat_symbolic_place",
        "Apply a privileged Habitat PDDL place transition; it is not physical arm execution.",
        category="simulation",
        input_schema={
            "type": "object",
            "properties": {
                "object_id": {"type": "string"},
                "receptacle_id": {"type": "string"},
            },
            "required": ["object_id", "receptacle_id"],
        },
        required_resources=("remote_runtime",),
        supported_robots=("habitat3",),
        driver_primitives=("habitat_symbolic_place",),
        timeout_sec=30.0,
        feedback_mode="status",
        goal_effects=("symbolically_places_habitat_target",),
        evidence_outputs=("pddl_action", "habitat_metrics"),
    )


class HabitatPickSkill(_HabitatSkill):
    spec = spec(
        "habitat_pick",
        "Physically pick an object in a checkpoint or physical-controller Habitat profile.",
        category="arm",
        input_schema={
            "type": "object",
            "properties": {
                "object_id": {"type": "string"},
                "max_steps": {"type": "integer"},
            },
            "required": ["object_id"],
        },
        required_resources=("remote_runtime", "arm", "gripper"),
        supported_robots=("habitat3",),
        driver_primitives=("habitat_pick",),
        safety_level="motion",
        timeout_sec=180.0,
        feedback_mode="status",
        goal_effects=("holds_habitat_target",),
        evidence_outputs=("grasp_state", "habitat_skill_trace"),
    )


class HabitatPlaceSkill(_HabitatSkill):
    spec = spec(
        "habitat_place",
        "Physically place an object in a checkpoint or physical-controller Habitat profile.",
        category="arm",
        input_schema={
            "type": "object",
            "properties": {
                "object_id": {"type": "string"},
                "receptacle_id": {"type": "string"},
                "max_steps": {"type": "integer"},
            },
            "required": ["object_id", "receptacle_id"],
        },
        required_resources=("remote_runtime", "arm", "gripper"),
        supported_robots=("habitat3",),
        driver_primitives=("habitat_place",),
        safety_level="motion",
        timeout_sec=180.0,
        feedback_mode="status",
        goal_effects=("places_habitat_target",),
        evidence_outputs=("object_pose", "habitat_skill_trace"),
    )


class HabitatWaitSkill(_HabitatSkill):
    spec = spec(
        "habitat_wait",
        "Advance a Habitat episode with safe zero actions for a bounded number of steps.",
        category="simulation",
        input_schema={
            "type": "object",
            "properties": {"steps": {"type": "integer", "minimum": 1}},
            "required": ["steps"],
        },
        required_resources=("remote_runtime",),
        supported_robots=("habitat3",),
        driver_primitives=("habitat_wait",),
        timeout_sec=120.0,
        feedback_mode="status",
        goal_effects=("advances_habitat_time",),
        evidence_outputs=("habitat_skill_trace",),
    )


class HabitatStopSkill(_HabitatSkill):
    spec = spec(
        "habitat_stop",
        "Cooperatively cancel the active Habitat skill and submit a zero-motion stop.",
        category="safety",
        input_schema={"type": "object", "properties": {}},
        required_resources=(),
        supported_robots=("habitat3",),
        driver_primitives=("habitat_stop",),
        safety_level="stop",
        timeout_sec=15.0,
        feedback_mode="none",
        goal_effects=("stops_habitat_motion",),
        evidence_outputs=("habitat_skill_trace",),
    )
