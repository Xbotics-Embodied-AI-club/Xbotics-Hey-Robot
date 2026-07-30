"""Native VLA Skill adapter over the reusable bounded-option runner."""

from __future__ import annotations

from typing import Any

from hey_robot.skills.context import SkillContext
from hey_robot.skills.models import Skill, SkillResult
from hey_robot.skills.registry import SkillRegistry
from hey_robot.skills.vla import VLAOptionRequest, VLAOptionRunner

# Large remote policies can delay simulation observations, so observation
# freshness must tolerate inference scheduling jitter.
_DEFAULT_FRESH_OBSERVATION_TIMEOUT_SEC = 10.0
_MAX_PUBLIC_STEPS = 600
# Long-horizon RoboCasa365 options can take more than half an hour.
_MAX_MANIPULATE_TIMEOUT_SEC = 3600.0

MANIPULATE_PARAMETERS = {
    "type": "object",
    "properties": {
        "task_prompt": {"type": "string", "minLength": 1},
        "max_steps": {
            "type": "integer",
            "minimum": 1,
            "maximum": _MAX_PUBLIC_STEPS,
            "default": 1,
        },
    },
    "required": ["task_prompt"],
    "additionalProperties": False,
}


async def manipulate(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    request = VLAOptionRequest(
        task_prompt=str(arguments["task_prompt"]),
        max_steps=int(arguments.get("max_steps", 1)),
        fresh_observation_timeout_sec=_DEFAULT_FRESH_OBSERVATION_TIMEOUT_SEC,
    )
    result = await VLAOptionRunner().run(ctx, request)
    return result.to_skill_result()


MANIPULATE = Skill(
    name="manipulate",
    description="Execute one bounded VLA manipulation option.",
    parameters=MANIPULATE_PARAMETERS,
    handler=manipulate,
    resources=("robot_control", "camera"),
    timeout_sec=_MAX_MANIPULATE_TIMEOUT_SEC,
    supported_robots=("xlerobot", "so101", "so101_mobile", "robocasa"),
    required_actions=("embodiment_native_action",),
    required_models=("manipulate",),
)


def register(registry: SkillRegistry) -> None:
    registry.register(MANIPULATE)
