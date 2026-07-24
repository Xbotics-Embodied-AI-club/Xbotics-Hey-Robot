from __future__ import annotations

from pathlib import Path

from hey_robot.app import DeploymentRunner
from hey_robot.config import DeploymentConfig


def test_deployment_runner_inspect(tmp_path: Path) -> None:
    config = DeploymentConfig.from_dict(
        {
            "deployment": {"id": "d1"},
            "monitoring": {"enabled": True, "port": 18081},
            "resources": {
                "runtime_dir": str(tmp_path / "runtime"),
                "media": {"root": str(tmp_path / "media")},
                "episodes": {"root": str(tmp_path / "episodes")},
            },
            "skills": {"tools": ["inspect_scene", "stop_motion"]},
            "robots": {"mock0": {"type": "mock"}},
            "agents": {
                "main": {
                    "type": "robot_agent",
                    "robot_id": "mock0",
                    "settings": {
                        "mode": "agent",
                        "models": {
                            "planner": {
                                "model": "mock-planner",
                                "api_key": "test-key",
                                "base_url": "http://127.0.0.1:9/v1",
                            }
                        },
                    },
                }
            },
        }
    )
    runner = DeploymentRunner(config, episode_dir=tmp_path / "episodes")
    info = runner.inspect()

    assert info["deployment"] == "d1"
    assert "mock0" in info["robots"]
    assert info["issues"] == []
    assert "agent:main" in info["services"]


def test_deployment_runner_composes_native_local_agent_without_controller(
    tmp_path: Path,
) -> None:
    config = DeploymentConfig.from_dict(
        {
            "deployment": {"id": "native-local"},
            "resources": {
                "runtime_dir": str(tmp_path / "runtime"),
                "media": {"root": str(tmp_path / "media")},
                "episodes": {"root": str(tmp_path / "episodes")},
            },
            "skills": {
                "modules": ["hey_robot.skills.builtins"],
                "tools": ["inspect_scene"],
                "execution_mode": "local",
            },
            "robots": {"mock0": {"type": "mock"}},
            "policies": {"embodied_skills": {"robot_id": "mock0"}},
            "agents": {
                "main": {
                    "robot_id": "mock0",
                    "settings": {
                        "models": {
                            "planner": {
                                "model": "mock-planner",
                                "api_key": "test-key",
                                "base_url": "http://127.0.0.1:9/v1",
                            }
                        }
                    },
                }
            },
        }
    )

    info = DeploymentRunner(config, episode_dir=tmp_path / "episodes").inspect()

    assert info["issues"] == []
    assert "robot" in info["services"]
    assert "skills" in info["services"]
    assert "agent:main" in info["services"]
    assert "skill-worker:local" not in info["services"]
    assert "skill-controller" not in info["services"]
