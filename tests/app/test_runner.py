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
            "skills": {"enabled": ["inspect_scene", "stop_motion"]},
            "robots": {"mock0": {"type": "mock"}},
            "agents": {
                "main": {
                    "type": "robot_agent",
                    "robot_id": "mock0",
                    "settings": {"mode": "direct"},
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
