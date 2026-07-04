from __future__ import annotations

import sys

from hey_robot.cli.doctor import main as doctor_main
from hey_robot.config import DeploymentConfig
from hey_robot.health import HealthReportService
from hey_robot.health.report import (
    HealthReport,
    _component_from_issue,
    _fix_hint,
    _overall_status,
    _platform_fix_hint,
    _skills_for_resources,
    _skills_from_text,
    _task_fix_hint,
)


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


def test_full_health_report_describes_structured_robot_components(tmp_path) -> None:
    config = DeploymentConfig.from_dict(
        {
            "resources": {
                "runtime_dir": str(tmp_path / "runtime"),
                "media": {"root": str(tmp_path / "media")},
                "episodes": {"root": str(tmp_path / "episodes")},
            },
            "robots": {
                "mock0": {
                    "type": "mock",
                    "components": {
                        "camera": {"enabled": True, "backend": "opencv"},
                        "base": {"enabled": True, "type": "sim"},
                        "arm": {"enabled": True, "type": "sim"},
                    },
                }
            },
            "policies": {
                "skills": {
                    "type": "skill",
                    "robot_id": "mock0",
                    "settings": {"codec": "skill"},
                }
            },
            "skills": {"enabled": ["inspect_scene", "move_base", "set_gripper"]},
        }
    )

    payload = HealthReportService(config).payload(robot_id="mock0", full=True)
    by_component = {report["component"]: report for report in payload["reports"]}

    assert by_component["robot.mock0.camera"]["status"] == "missing"
    assert "inspect_scene" in by_component["robot.mock0.camera"]["impacted_skills"]
    assert by_component["robot.mock0.base"]["status"] == "configured"
    assert "move_base" in by_component["robot.mock0.base"]["impacted_skills"]
    assert by_component["robot.mock0.arm"]["status"] == "missing"
    assert "set_gripper" in by_component["robot.mock0.arm"]["impacted_skills"]


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


def test_health_report_helper_branches_describe_actionable_failures() -> None:
    reports = [
        HealthReport(
            component="camera",
            status="failed",
            severity="warning",
            evidence="camera missing",
        ),
        HealthReport(
            component="skill.inspect_scene",
            status="ready_check_required",
            severity="error",
            evidence="needs camera",
        ),
    ]

    assert _overall_status(reports) == "degraded"
    assert _component_from_issue("skill inspect_scene missing", robot_id=None) == (
        "skill.config"
    )
    assert _component_from_issue("robot mock0 camera missing", robot_id="mock0") == (
        "robot.mock0"
    )
    assert _component_from_issue("robot other camera missing", robot_id="mock0") is None
    assert _skills_from_text("enabled skill inspect_scene is not available") == (
        "inspect_scene",
    )
    assert _skills_from_text("skill") == ()
    assert "camera scan" in (_fix_hint("camera is missing") or "")
    assert "model service" in (_fix_hint("required capability is unavailable") or "")
    assert "PATH" in (_platform_fix_hint("nats_server") or "")
    assert "Python 3.12" in (_platform_fix_hint("python") or "")
    assert _platform_fix_hint("unknown") is None
    assert "camera availability" in (_task_fix_hint("image observation failed") or "")
    assert "target back into view" in (_task_fix_hint("target lost") or "")
    assert "inspect" in (_task_fix_hint(None) or "")
    assert _skills_for_resources(("camera", "base", "arm")) == (
        "inspect_scene",
        "human_follow",
        "move_base",
        "turn_base",
        "base_velocity_step",
        "set_arm_pose",
        "set_gripper",
    )
