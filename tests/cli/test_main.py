from __future__ import annotations

import sys

import pytest

from hey_robot.cli.main import CLI_ACTIONS, main


def test_main_cli_exposes_current_runtime_actions_only() -> None:
    assert "system1" not in CLI_ACTIONS
    assert "system2" not in CLI_ACTIONS
    assert "user" not in CLI_ACTIONS
    assert "stt" not in CLI_ACTIONS
    assert "tts" not in CLI_ACTIONS
    assert "policy" not in CLI_ACTIONS
    assert "task-supervisor" in CLI_ACTIONS


def test_main_cli_reports_unknown_action(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["hey-robot", "unknown-action"])

    with pytest.raises(SystemExit, match="unknown CLI action: unknown-action"):
        main()
