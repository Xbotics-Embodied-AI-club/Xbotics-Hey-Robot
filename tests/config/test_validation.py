from __future__ import annotations

import sys
import types

import pytest

from hey_robot.config import DeploymentConfig
from hey_robot.config.validation import validate_deployment
from hey_robot.skills import Skill, SkillResult


async def _noop_skill(_ctx, _arguments) -> SkillResult:
    return SkillResult(True, "done", "completed")


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
            "skills": {"tools": ["inspect_scene"]},
        }
    )

    issues = validate_deployment(config)

    assert issues == []
    assert runtime_dir.exists()
    assert media_root.exists()
    assert episodes_root.exists()


def test_validate_deployment_requires_explicit_lerobot_policy_contract(
    tmp_path,
) -> None:
    config = DeploymentConfig.from_dict(
        {
            "resources": {"runtime_dir": str(tmp_path / "runtime")},
            "model_services": {
                "policy": {
                    "type": "robot_policy",
                    "robot_id": "robot",
                    "settings": {
                        "runtime": "other",
                        "action_dimensions": 0,
                    },
                }
            },
        }
    )

    messages = {issue.message for issue in validate_deployment(config)}

    assert (
        "model service policy has unsupported robot policy runtime 'other'" in messages
    )
    assert "model service policy requires setting policy_path" in messages
    assert "model service policy requires setting policy_device" in messages
    assert "model service policy requires setting action_space" in messages
    assert "model service policy requires positive action_dimensions" in messages


def test_validate_deployment_requires_explicit_vln_backend_contract(tmp_path) -> None:
    config = DeploymentConfig.from_dict(
        {
            "resources": {"runtime_dir": str(tmp_path / "runtime")},
            "model_services": {
                "planner": {
                    "type": "vln_planner",
                    "robot_id": "robot",
                    "settings": {
                        "backend": "unknown",
                        "control_mode": "direct_velocity",
                    },
                }
            },
        }
    )

    messages = {issue.message for issue in validate_deployment(config)}

    assert "model service planner has unsupported VLN backend 'unknown'" in messages
    assert (
        "model service planner has unsupported VLN control_mode 'direct_velocity'"
        in messages
    )
    assert "model service planner requires setting model_path" in messages
    assert "model service planner requires setting internnav_repo" in messages
    assert "model service planner requires setting media_root" in messages


def test_validate_deployment_rejects_unsafe_dual_vln_control_limits(
    tmp_path,
) -> None:
    config = DeploymentConfig.from_dict(
        {
            "resources": {"runtime_dir": str(tmp_path / "runtime")},
            "model_services": {
                "planner": {
                    "type": "vln_planner",
                    "robot_id": "robot",
                    "settings": {
                        "control_mode": "base_action_chunk",
                        "base_linear_speed": 1.0,
                        "base_angular_speed": 2.0,
                        "max_action_chunk_steps": 20,
                        "system1_replans_per_waypoint": 20,
                        "discrete_forward_cm": 30,
                        "discrete_turn_deg": 45,
                        "mock_mode": True,
                    },
                }
            },
        }
    )

    messages = {issue.message for issue in validate_deployment(config)}

    assert any("base_linear_speed" in message for message in messages)
    assert any("base_angular_speed" in message for message in messages)
    assert any("max_action_chunk_steps" in message for message in messages)
    assert any("system1_replans_per_waypoint" in message for message in messages)


def test_deployment_config_rejects_unknown_skill_surface_field(
    tmp_path,
) -> None:
    del tmp_path
    with pytest.raises(ValueError, match="skills uses unknown fields"):
        DeploymentConfig.from_dict({"skills": {"legacy": ["move_base"]}})


def test_validate_deployment_rejects_unsupported_robot_family(
    tmp_path, monkeypatch
) -> None:
    module_name = "hey_robot.skills.fake_robot_specific_skill"
    module = types.ModuleType(module_name)

    def register(registry) -> None:
        registry.register(
            Skill(
                name="robot_specific_skill",
                description="Only supports another robot family.",
                parameters={"type": "object", "additionalProperties": True},
                handler=_noop_skill,
                supported_robots=("other_robot",),
            )
        )

    setattr(module, "register", register)
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
                "tools": ["robot_specific_skill"],
            },
        }
    )

    issues = validate_deployment(config)

    assert any("supports robots other_robot" in issue.message for issue in issues)


def test_validate_deployment_rejects_unavailable_required_model_service(
    tmp_path, monkeypatch
) -> None:
    module_name = "hey_robot.skills.fake_required_model_service_skill"
    module = types.ModuleType(module_name)

    def register(registry) -> None:
        registry.register(
            Skill(
                name="required_model_service_skill",
                description="Requires an external service.",
                parameters={"type": "object", "additionalProperties": True},
                handler=_noop_skill,
                required_models=("special_service",),
            )
        )

    setattr(module, "register", register)
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
                "tools": ["required_model_service_skill"],
            },
        }
    )

    issues = validate_deployment(config)

    assert any(
        issue.message
        == "skill required_model_service_skill requires unavailable model service "
        "special_service"
        for issue in issues
    )


def test_validate_deployment_rejects_missing_driver_primitive(
    tmp_path, monkeypatch
) -> None:
    module_name = "hey_robot.skills.fake_driver_primitive_skill"
    module = types.ModuleType(module_name)

    def register(registry) -> None:
        registry.register(
            Skill(
                name="so101_root_skill",
                description="Root skill for SO101.",
                parameters={"type": "object", "additionalProperties": True},
                handler=_noop_skill,
                required_actions=("set_arm_pose",),
                supported_robots=("so101",),
            )
        )

    setattr(module, "register", register)
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
                "tools": ["so101_root_skill"],
            },
        }
    )

    issues = validate_deployment(config)

    assert any(
        "requires driver primitives set_arm_pose via so101_root_skill" in issue.message
        for issue in issues
    )


def test_validate_deployment_allows_configured_driver_primitive(
    tmp_path, monkeypatch
) -> None:
    module_name = "hey_robot.skills.fake_configured_driver_primitive_skill"
    module = types.ModuleType(module_name)

    def register(registry) -> None:
        registry.register(
            Skill(
                name="configured_primitive_skill",
                description="Uses a deployment-declared primitive.",
                parameters={"type": "object", "additionalProperties": True},
                handler=_noop_skill,
                required_actions=("custom_drive",),
                supported_robots=("custombot",),
            )
        )

    setattr(module, "register", register)
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
                "tools": ["configured_primitive_skill"],
            },
        }
    )

    assert validate_deployment(config) == []


def test_validate_deployment_rejects_multiple_enabled_agents(tmp_path) -> None:
    config = DeploymentConfig.from_dict(
        {
            "resources": {"runtime_dir": str(tmp_path / "runtime")},
            "robots": {"mock0": {"type": "mock"}},
            "agents": {
                "first": {"robot_id": "mock0"},
                "second": {"robot_id": "mock0"},
            },
            "skills": {"tools": ["inspect_scene"]},
        }
    )

    messages = [issue.message for issue in validate_deployment(config)]

    assert any("exactly one enabled autonomous agent" in item for item in messages)
