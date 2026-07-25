from __future__ import annotations

from hey_robot.robot_api import RobotActionSpec, RobotClientCapabilities
from hey_robot.skills import Skill, SkillResult, validate_skill_surface


async def _handler(_ctx, _arguments):
    return SkillResult(True, "done", "completed")


def test_validate_skill_surface_checks_static_requirements() -> None:
    skill = Skill(
        "pick",
        "Pick an object.",
        {},
        _handler,
        supported_robots=("xlerobot",),
        required_actions=("move_arm",),
        required_models=("manipulate",),
    )
    capabilities = RobotClientCapabilities(
        robot_id="mock0",
        actions=(RobotActionSpec("stop_motion", {}),),
    )

    issues = validate_skill_surface(
        (skill,),
        robot_family="so101",
        robot_capabilities=capabilities,
        model_capabilities=("caption",),
    )

    assert [issue.message for issue in issues] == [
        "robot family 'so101' is not supported",
        "missing robot actions: move_arm",
        "missing model capabilities: manipulate",
    ]


def test_validate_skill_surface_accepts_available_requirements() -> None:
    skill = Skill(
        "inspect",
        "Inspect.",
        {},
        _handler,
        supported_robots=("xlerobot",),
        required_actions=("observe",),
        required_models=("caption",),
    )
    capabilities = RobotClientCapabilities(
        robot_id="mock0",
        actions=(RobotActionSpec("observe", {}),),
    )

    assert (
        validate_skill_surface(
            (skill,),
            robot_family="xlerobot",
            robot_capabilities=capabilities,
            model_capabilities=("caption",),
        )
        == ()
    )
