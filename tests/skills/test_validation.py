from __future__ import annotations

import pytest

from hey_robot.robot_runtime.clients import RobotActionSpec, RobotClientCapabilities
from hey_robot.skills import Skill, SkillRegistry, SkillResult, validate_skill_surface


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


def test_registry_resolves_transitive_dependencies_in_execution_order() -> None:
    registry = SkillRegistry()
    registry.register(Skill("leaf", "Leaf.", {}, _handler))
    registry.register(Skill("middle", "Middle.", {}, _handler, dependencies=("leaf",)))
    registry.register(Skill("root", "Root.", {}, _handler, dependencies=("middle",)))

    assert [skill.name for skill in registry.resolve_dependencies(("root",))] == [
        "leaf",
        "middle",
        "root",
    ]


def test_registry_rejects_missing_and_cyclic_dependencies() -> None:
    missing = SkillRegistry()
    missing.register(Skill("root", "Root.", {}, _handler, dependencies=("missing",)))
    with pytest.raises(ValueError, match="depends on unknown skill"):
        missing.resolve_dependencies(("root",))

    cyclic = SkillRegistry()
    cyclic.register(Skill("a", "A.", {}, _handler, dependencies=("b",)))
    cyclic.register(Skill("b", "B.", {}, _handler, dependencies=("a",)))
    with pytest.raises(ValueError, match="dependency cycle: a -> b -> a"):
        cyclic.resolve_dependencies(("a",))
