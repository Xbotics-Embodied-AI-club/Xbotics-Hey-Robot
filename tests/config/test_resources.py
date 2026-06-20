from __future__ import annotations

from pathlib import Path

from hey_robot.config import DeploymentConfig
from hey_robot.config.validation import validate_deployment


def test_deployment_config_resources_and_validation(tmp_path: Path) -> None:
    config = DeploymentConfig.from_dict(
        {
            "deployment": {"id": "d1"},
            "resources": {
                "runtime_dir": str(tmp_path / "runtime"),
                "media": {"root": str(tmp_path / "media"), "max_items": 4},
                "episodes": {"root": str(tmp_path / "episodes")},
                "events": {"retain": 8},
            },
            "robots": {"mock0": {"type": "mock"}},
            "agents": {"main": {"type": "robot_agent", "robot_id": "mock0"}},
            "policies": {"mock_policy": {"type": "mock", "robot_id": "mock0"}},
            "skills": {"enabled": ["inspect_scene"]},
        }
    )

    assert config.resources.media_root.endswith("media")
    assert config.resources.media_image_save_every_n == 1
    assert not validate_deployment(config)


def test_deployment_config_identity_defaults_and_bindings(tmp_path: Path) -> None:
    config = DeploymentConfig.from_dict(
        {
            "resources": {
                "runtime_dir": str(tmp_path / "runtime"),
                "episodes": {"root": str(tmp_path / "episodes")},
            },
            "identity": {
                "enabled": True,
                "unified_user_episodes": True,
                "default_user_id": "fallback-user",
                "bindings": {
                    "web:sender:web-user": "owner",
                    "voice:sender:voice-user": "owner",
                },
            },
        }
    )

    assert config.identity.enabled is True
    assert config.identity.unified_user_episodes is True
    assert config.identity.default_user_id == "fallback-user"
    assert config.identity.bindings["web:sender:web-user"] == "owner"
    assert config.identity.bindings["voice:sender:voice-user"] == "owner"
