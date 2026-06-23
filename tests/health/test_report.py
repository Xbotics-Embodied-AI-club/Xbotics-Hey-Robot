from __future__ import annotations

import sys

from hey_robot.cli.doctor import main as doctor_main
from hey_robot.config import DeploymentConfig
from hey_robot.health import HealthReportService


def _config(tmp_path) -> DeploymentConfig:
    return DeploymentConfig.from_dict(
        {
            "resources": {
                "runtime_dir": str(tmp_path / "runtime"),
                "media": {"root": str(tmp_path / "media")},
                "episodes": {"root": str(tmp_path / "episodes")},
            },
            "robots": {"mock0": {"type": "mock"}},
            "policies": {
                "skills": {
                    "type": "skill",
                    "robot_id": "mock0",
                    "settings": {"codec": "skill"},
                }
            },
            "skills": {"enabled": ["inspect_scene", "human_follow"]},
        }
    )


def test_health_report_describes_skill_resource_readiness(tmp_path) -> None:
    payload = HealthReportService(_config(tmp_path)).payload(robot_id="mock0")

    assert payload["status"] == "ok"
    reports = payload["reports"]
    human_follow = next(
        report for report in reports if report["component"] == "skill.human_follow"
    )
    assert human_follow["status"] == "ready_check_required"
    assert human_follow["impacted_skills"] == ["human_follow"]
    assert "camera" in human_follow["metadata"]["resources"]
    assert "base" in human_follow["metadata"]["resources"]
    assert "verify camera scan" in human_follow["fix_hint"]


def test_full_health_report_aggregates_platform_and_script_inventory(tmp_path) -> None:
    config = _config(tmp_path)
    payload = HealthReportService(
        config,
        config_path=tmp_path / "deployment.yaml",
    ).payload(robot_id="mock0", full=True)

    components = {report["component"] for report in payload["reports"]}
    assert "platform.python" in components
    assert "diagnostics.check_platform" in components
    assert "diagnostics.xlerobot.camera" in components


def test_doctor_cli_outputs_json_report(tmp_path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "deployment.yaml"
    runtime_dir = (tmp_path / "runtime").as_posix()
    media_root = (tmp_path / "media").as_posix()
    episodes_root = (tmp_path / "episodes").as_posix()
    config_path.write_text(
        f"""
resources:
  runtime_dir: "{runtime_dir}"
  media:
    root: "{media_root}"
  episodes:
    root: "{episodes_root}"
robots:
  mock0:
    type: mock
policies:
  skills:
    robot_id: mock0
    freq_hz: 10.0
skills:
  enabled:
    - inspect_scene
    - human_follow
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hey-robot doctor",
            "--config",
            str(config_path),
            "--robot",
            "mock0",
            "--json",
        ],
    )

    doctor_main()

    output = capsys.readouterr().out
    assert '"status": "ok"' in output
    assert '"component": "skill.human_follow"' in output
