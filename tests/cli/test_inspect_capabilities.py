from __future__ import annotations

import sys

from hey_robot.cli.inspect import main


def test_inspect_capabilities_prints_manifest(tmp_path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "deployment.yaml"
    config_path.write_text(
        """
deployment:
  id: d1
robots:
  mock0:
    type: mock
agents:
  main:
    robot_id: mock0
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys, "argv", ["hey-robot inspect", "capabilities", "--config", str(config_path)]
    )

    main()

    output = capsys.readouterr().out
    assert '"prompt_skills"' not in output
    assert '"robot_skill_actions"' in output
