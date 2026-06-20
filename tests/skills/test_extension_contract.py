from __future__ import annotations

import sys
import types

from hey_robot.config import DeploymentConfig
from hey_robot.skills.base import BaseSkill, SkillResult, SkillSpec
from hey_robot.skills.context import SkillContext
from hey_robot.skills.registry import registry_from_config
from hey_robot.skills.runtime import SkillRuntime


class _InnerSkill(BaseSkill):
    spec = SkillSpec(
        name="inner_skill",
        description="Inner skill.",
        agent_visible=True,
    )

    async def execute(self, ctx, arguments):
        del ctx, arguments
        return SkillResult(success=True, summary="inner-ok")


class _OuterSkill(BaseSkill):
    spec = SkillSpec(
        name="outer_skill",
        description="Outer skill.",
        agent_visible=True,
        dependencies=("inner_skill",),
    )

    async def execute(self, ctx, arguments):
        result = await ctx.invoke("inner_skill", dict(arguments))
        return SkillResult(success=True, summary=f"outer:{result.summary}")


def test_third_party_skill_can_compose_without_agent_or_hardware_access(
    monkeypatch,
) -> None:
    module_name = "tests.fake_extension_module"
    module = types.ModuleType(module_name)

    def register_skills(registry) -> None:
        registry.register(_InnerSkill())
        registry.register(_OuterSkill())

    setattr(module, "register_skills", register_skills)
    monkeypatch.setitem(sys.modules, module_name, module)

    config = DeploymentConfig.from_dict(
        {
            "skills": {
                "modules": [module_name],
                "enabled": ["outer_skill", "inner_skill"],
            }
        }
    )
    registry = registry_from_config(config)
    runtime = SkillRuntime(registry)
    result = __import__("asyncio").run(
        runtime.execute(
            "outer_skill",
            {"x": 1},
            context_factory=lambda invoke: SkillContext(invoke=invoke),
        )
    )

    assert result.success is True
    assert result.summary == "outer:inner-ok"
