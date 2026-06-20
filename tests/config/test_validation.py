from __future__ import annotations

import sys
import types

from hey_robot.config import DeploymentConfig
from hey_robot.config.validation import validate_deployment


def test_validate_deployment_reports_missing_robot_and_policy(tmp_path) -> None:
    config = DeploymentConfig.from_dict(
        {
            "resources": {
                "runtime_dir": str(tmp_path / "runtime"),
                "media": {"root": str(tmp_path / "media")},
                "episodes": {"root": str(tmp_path / "episodes")},
            },
            "agents": {
                "main": {
                    "type": "robot_agent",
                    "robot_id": "missing-robot",
                    "policy_id": "missing-policy",
                }
            },
            "policies": {"p1": {"type": "mock", "robot_id": "missing-robot"}},
            "channels": {"web": {"type": "web", "enabled": True}},
        }
    )

    issues = validate_deployment(config)
    messages = {issue.message for issue in issues}

    assert "agent main references missing robot missing-robot" in messages
    assert "agent main references missing policy missing-policy" in messages
    assert "policy p1 references missing robot missing-robot" in messages


def test_validate_deployment_creates_resource_paths(tmp_path) -> None:
    runtime_dir = tmp_path / "runtime"
    media_root = tmp_path / "media"
    episodes_root = tmp_path / "episodes"
    config = DeploymentConfig.from_dict(
        {
            "resources": {
                "runtime_dir": str(runtime_dir),
                "media": {"root": str(media_root)},
                "episodes": {"root": str(episodes_root)},
            },
            "robots": {"mock0": {"type": "mock"}},
            "skills": {"enabled": ["inspect_scene"]},
        }
    )

    issues = validate_deployment(config)

    assert issues == []
    assert runtime_dir.exists()
    assert media_root.exists()
    assert episodes_root.exists()


def test_validate_deployment_rejects_implementation_skill_in_production(
    tmp_path,
) -> None:
    config = DeploymentConfig.from_dict(
        {
            "resources": {
                "runtime_dir": str(tmp_path / "runtime"),
                "media": {"root": str(tmp_path / "media")},
                "episodes": {"root": str(tmp_path / "episodes")},
            },
            "robots": {"mock0": {"type": "mock"}},
            "skills": {
                "mode": "production",
                "enabled": ["move_base"],
            },
        }
    )

    issues = validate_deployment(config)

    assert any("implementation-level" in issue.message for issue in issues)


def test_validate_deployment_reports_transitive_unknown_skill_dependency(
    tmp_path, monkeypatch
) -> None:
    from hey_robot.skills.base import BaseSkill, SkillResult, SkillSpec

    module_name = "tests.fake_validation_plugin"
    module = types.ModuleType(module_name)

    class RootSkill(BaseSkill):
        spec = SkillSpec(
            name="root_skill",
            description="Root skill.",
            dependencies=("child_skill",),
        )

        async def execute(self, ctx, arguments):
            del ctx, arguments
            return SkillResult(success=True, summary="root")

    class ChildSkill(BaseSkill):
        spec = SkillSpec(
            name="child_skill",
            description="Child skill.",
            dependencies=("missing_leaf_skill",),
            agent_visible=False,
        )

        async def execute(self, ctx, arguments):
            del ctx, arguments
            return SkillResult(success=True, summary="child")

    def register_skills(registry) -> None:
        registry.register(RootSkill())
        registry.register(ChildSkill())

    setattr(module, "register_skills", register_skills)
    monkeypatch.setitem(sys.modules, module_name, module)

    config = DeploymentConfig.from_dict(
        {
            "resources": {
                "runtime_dir": str(tmp_path / "runtime"),
                "media": {"root": str(tmp_path / "media")},
                "episodes": {"root": str(tmp_path / "episodes")},
            },
            "robots": {"mock0": {"type": "mock"}},
            "skills": {
                "modules": [module_name],
                "enabled": ["root_skill"],
            },
        }
    )

    issues = validate_deployment(config)

    assert any(
        issue.message
        == "skill root_skill references unknown dependency missing_leaf_skill"
        for issue in issues
    )


def test_validate_deployment_rejects_unsupported_robot_family(
    tmp_path, monkeypatch
) -> None:
    from hey_robot.skills.base import BaseSkill, SkillResult, SkillSpec

    module_name = "tests.fake_robot_specific_skill"
    module = types.ModuleType(module_name)

    class RobotSpecificSkill(BaseSkill):
        spec = SkillSpec(
            name="robot_specific_skill",
            description="Only supports another robot family.",
            supported_robots=("other_robot",),
        )

        async def execute(self, ctx, arguments):
            del ctx, arguments
            return SkillResult(success=True, summary="done")

    def register_skills(registry) -> None:
        registry.register(RobotSpecificSkill())

    setattr(module, "register_skills", register_skills)
    monkeypatch.setitem(sys.modules, module_name, module)
    config = DeploymentConfig.from_dict(
        {
            "resources": {
                "runtime_dir": str(tmp_path / "runtime"),
                "media": {"root": str(tmp_path / "media")},
                "episodes": {"root": str(tmp_path / "episodes")},
            },
            "robots": {"robot0": {"type": "xlerobot"}},
            "skills": {
                "modules": [module_name],
                "enabled": ["robot_specific_skill"],
            },
        }
    )

    issues = validate_deployment(config)

    assert any("supports robots other_robot" in issue.message for issue in issues)


def test_validate_deployment_rejects_unavailable_external_capability(
    tmp_path, monkeypatch
) -> None:
    from hey_robot.skills.base import BaseSkill, SkillResult, SkillSpec

    module_name = "tests.fake_external_capability_skill"
    module = types.ModuleType(module_name)

    class ExternalCapabilitySkill(BaseSkill):
        spec = SkillSpec(
            name="external_capability_skill",
            description="Requires an external service.",
            external_capability="special_service",
        )

        async def execute(self, ctx, arguments):
            del ctx, arguments
            return SkillResult(success=True, summary="done")

    def register_skills(registry) -> None:
        registry.register(ExternalCapabilitySkill())

    setattr(module, "register_skills", register_skills)
    monkeypatch.setitem(sys.modules, module_name, module)
    config = DeploymentConfig.from_dict(
        {
            "resources": {
                "runtime_dir": str(tmp_path / "runtime"),
                "media": {"root": str(tmp_path / "media")},
                "episodes": {"root": str(tmp_path / "episodes")},
            },
            "robots": {"robot0": {"type": "xlerobot"}},
            "skills": {
                "modules": [module_name],
                "enabled": ["external_capability_skill"],
            },
        }
    )

    issues = validate_deployment(config)

    assert any(
        issue.message
        == "skill external_capability_skill requires unavailable capability "
        "special_service"
        for issue in issues
    )


def test_validate_deployment_rejects_missing_driver_primitive(
    tmp_path, monkeypatch
) -> None:
    from hey_robot.skills.base import BaseSkill, SkillResult, SkillSpec

    module_name = "tests.fake_driver_primitive_skill"
    module = types.ModuleType(module_name)

    class RootSkill(BaseSkill):
        spec = SkillSpec(
            name="so101_root_skill",
            description="Root skill for SO101.",
            dependencies=("canonical_arm_primitive",),
            supported_robots=("so101",),
        )

        async def execute(self, ctx, arguments):
            del ctx, arguments
            return SkillResult(success=True, summary="root")

    class CanonicalArmPrimitive(BaseSkill):
        spec = SkillSpec(
            name="canonical_arm_primitive",
            description="Requires a canonical arm primitive.",
            driver_primitives=("set_arm_pose",),
            supported_robots=("so101",),
            agent_visible=False,
        )

        async def execute(self, ctx, arguments):
            del ctx, arguments
            return SkillResult(success=True, summary="primitive")

    def register_skills(registry) -> None:
        registry.register(RootSkill())
        registry.register(CanonicalArmPrimitive())

    setattr(module, "register_skills", register_skills)
    monkeypatch.setitem(sys.modules, module_name, module)
    config = DeploymentConfig.from_dict(
        {
            "resources": {
                "runtime_dir": str(tmp_path / "runtime"),
                "media": {"root": str(tmp_path / "media")},
                "episodes": {"root": str(tmp_path / "episodes")},
            },
            "robots": {"robot0": {"type": "so101"}},
            "skills": {
                "modules": [module_name],
                "enabled": ["so101_root_skill"],
            },
        }
    )

    issues = validate_deployment(config)

    assert any(
        "requires driver primitives set_arm_pose via canonical_arm_primitive"
        in issue.message
        for issue in issues
    )


def test_validate_deployment_allows_configured_driver_primitive(
    tmp_path, monkeypatch
) -> None:
    from hey_robot.skills.base import BaseSkill, SkillResult, SkillSpec

    module_name = "tests.fake_configured_driver_primitive_skill"
    module = types.ModuleType(module_name)

    class ConfiguredPrimitiveSkill(BaseSkill):
        spec = SkillSpec(
            name="configured_primitive_skill",
            description="Uses a deployment-declared primitive.",
            driver_primitives=("custom_drive",),
            supported_robots=("custombot",),
        )

        async def execute(self, ctx, arguments):
            del ctx, arguments
            return SkillResult(success=True, summary="done")

    def register_skills(registry) -> None:
        registry.register(ConfiguredPrimitiveSkill())

    setattr(module, "register_skills", register_skills)
    monkeypatch.setitem(sys.modules, module_name, module)
    config = DeploymentConfig.from_dict(
        {
            "resources": {
                "runtime_dir": str(tmp_path / "runtime"),
                "media": {"root": str(tmp_path / "media")},
                "episodes": {"root": str(tmp_path / "episodes")},
            },
            "robots": {
                "robot0": {
                    "type": "custombot",
                    "settings": {"supported_driver_primitives": ["custom_drive"]},
                }
            },
            "skills": {
                "modules": [module_name],
                "enabled": ["configured_primitive_skill"],
            },
        }
    )

    assert validate_deployment(config) == []
