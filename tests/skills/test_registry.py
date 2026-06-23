from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from hey_robot.config import DeploymentConfig
from hey_robot.skills.base import BaseSkill, SkillResult, SkillSpec
from hey_robot.skills.context import SkillContext
from hey_robot.skills.registry import load_skill_registry, registry_from_config
from hey_robot.skills.runtime import SkillRuntime


def test_registry_loads_builtin_module_and_defaults_to_agent_visible_surface() -> None:
    registry = load_skill_registry()

    assert "inspect_scene" in registry.names()
    assert "reset_posture" in registry.names()
    assert "move_base" not in registry.names()


class _RobotAPI:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def move_base(self, **arguments) -> None:
        self.calls.append(("move_base", dict(arguments)))


def test_runtime_runs_plugin_backed_builtin_skill() -> None:
    robot = _RobotAPI()
    registry = load_skill_registry(enabled=("move_base",))
    runtime = SkillRuntime(registry)

    result = __import__("asyncio").run(
        runtime.execute(
            "move_base",
            {"direction": "forward", "distance_cm": 10},
            context_factory=lambda invoke: SkillContext(robot=robot, invoke=invoke),
        )
    )

    assert result.success is True
    assert robot.calls == [("move_base", {"direction": "forward", "distance_cm": 10})]


def test_runtime_returns_failed_result_for_invalid_arguments() -> None:
    robot = _RobotAPI()
    registry = load_skill_registry(enabled=("move_base",))
    runtime = SkillRuntime(registry)

    result = __import__("asyncio").run(
        runtime.execute(
            "move_base",
            {"direction": "forward"},
            context_factory=lambda invoke: SkillContext(robot=robot, invoke=invoke),
        )
    )

    assert result.success is False
    assert result.status == "failed"
    assert result.failure_mode == "invalid_arguments"
    assert "distance_cm" in (result.summary or "")
    assert robot.calls == []


def test_registry_from_config_loads_custom_module_and_filters_enabled_surface(
    monkeypatch,
) -> None:
    module_name = "tests.fake_registry_plugin"
    module = types.ModuleType(module_name)

    class VisibleSkill(BaseSkill):
        spec = SkillSpec(
            name="visible_plugin_skill",
            description="Visible test plugin.",
            agent_visible=True,
        )

        async def execute(self, ctx, arguments):
            del ctx, arguments
            return SkillResult(success=True, summary="visible")

    class HiddenSkill(BaseSkill):
        spec = SkillSpec(
            name="hidden_plugin_skill",
            description="Hidden test plugin.",
            agent_visible=False,
        )

        async def execute(self, ctx, arguments):
            del ctx, arguments
            return SkillResult(success=True, summary="hidden")

    def register_skills(registry) -> None:
        registry.register(VisibleSkill())
        registry.register(HiddenSkill())

    setattr(module, "register_skills", register_skills)
    monkeypatch.setitem(sys.modules, module_name, module)

    config = DeploymentConfig.from_dict(
        {
            "skills": {
                "modules": [module_name],
                "enabled": ["visible_plugin_skill"],
            }
        }
    )

    registry = registry_from_config(config)

    assert registry.names() == ("visible_plugin_skill",)
    assert registry.names(enabled_only=False) == (
        "visible_plugin_skill",
        "hidden_plugin_skill",
    )


def test_runtime_uses_context_factory() -> None:
    class EchoRobot:
        def __init__(self, label: str) -> None:
            self.label = label

        async def move_base(self, **arguments) -> str:
            return f"{self.label}:{arguments['distance_cm']}"

    class EchoMoveSkill(BaseSkill):
        spec = SkillSpec(
            name="echo_move",
            description="Echo context-specific robot result.",
            input_schema={
                "type": "object",
                "properties": {"distance_cm": {"type": "number"}},
                "required": ["distance_cm"],
            },
            agent_visible=True,
        )

        async def execute(self, ctx, arguments):
            summary = await ctx.robot.move_base(**arguments)
            return SkillResult(success=True, summary=summary)

    registry = load_skill_registry(enabled=())
    registry.register(EchoMoveSkill())
    registry = registry.configure(enabled=("echo_move",))
    runtime = SkillRuntime(registry)

    result = __import__("asyncio").run(
        runtime.execute(
            "echo_move",
            {"distance_cm": 12},
            context_factory=lambda invoke: SkillContext(
                robot=EchoRobot("override"),
                invoke=invoke,
            ),
        )
    )

    assert result.success is True
    assert result.summary == "override:12"


def test_runtime_wraps_plugin_exception_as_internal_error() -> None:
    class BrokenSkill(BaseSkill):
        spec = SkillSpec(
            name="broken_skill",
            description="Raise during execution.",
            agent_visible=True,
        )

        async def execute(self, ctx, arguments):
            del ctx, arguments
            raise RuntimeError("plugin exploded")

    registry = load_skill_registry(enabled=())
    registry.register(BrokenSkill())
    registry = registry.configure(enabled=("broken_skill",))
    runtime = SkillRuntime(registry)

    result = __import__("asyncio").run(
        runtime.execute(
            "broken_skill",
            context_factory=lambda invoke: SkillContext(invoke=invoke),
        )
    )

    assert result.success is False
    assert result.failure_mode == "internal_error"
    assert result.error == "plugin exploded"


def test_runtime_executes_vla_manipulation_skill() -> None:
    class CapabilityAPI:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def call(self, name: str, arguments: dict):
            self.calls.append((name, dict(arguments)))
            return SimpleNamespace(
                success=True,
                summary="object picked",
                status="completed",
                failure_mode=None,
                error=None,
                metrics={"verified": True},
            )

    capabilities = CapabilityAPI()
    registry = load_skill_registry(enabled=("vla_manipulation",))
    runtime = SkillRuntime(registry)

    result = __import__("asyncio").run(
        runtime.execute(
            "vla_manipulation",
            {"task_prompt": "Pick up the red cup."},
            context_factory=lambda invoke: SkillContext(
                capabilities=capabilities,
                invoke=invoke,
            ),
        )
    )

    assert result.success is True
    assert result.data == {"verified": True}
    assert capabilities.calls == [
        ("vla_manipulation", {"task_prompt": "Pick up the red cup."})
    ]


def test_registry_rejects_duplicate_skill_names() -> None:
    registry = load_skill_registry(enabled=())
    duplicate = registry.get("inspect_scene", enabled_only=False).skill

    assert duplicate is not None
    with pytest.raises(ValueError, match="duplicate skill: inspect_scene"):
        registry.register(duplicate)


def test_robot_skill_catalog_exposes_capability_semantics() -> None:
    catalog = load_skill_registry().robot_skill_catalog()

    turn_base = catalog.get("turn_base")
    inspect_scene = catalog.get("inspect_scene")

    assert turn_base.capability_type == "base_turn"
    assert turn_base.evidence_outputs == ("base_turn_action_result",)
    assert inspect_scene.capability_type == "scene_observation"
    assert "base_turn_action_result" in inspect_scene.cannot_satisfy
    assert catalog.get("detect_marker").evidence_outputs == ("marker_detection_result",)
