from __future__ import annotations

import sys

from hey_robot.cli.main import CLI_ACTIONS, main
from hey_robot.config import DeploymentConfig
from hey_robot.episode import EpisodeScope, JsonlEpisodeStore, RobotEpisodeStateStore
from hey_robot.protocol import Envelope, SkillEvent, UserTurn
from hey_robot.skills import SkillStore


def test_episode_cli_skills_are_new_deployment_skills() -> None:
    assert CLI_ACTIONS["inspect-episode"] == "hey_robot.cli.episode:main"
    assert CLI_ACTIONS["export-run"] == "hey_robot.cli.episode:main"


def test_inspect_episode_cli_prints_materialized_summary(
    tmp_path, monkeypatch, capsys
) -> None:
    runtime = tmp_path / "runtime"
    episodes = tmp_path / "episodes"
    config = DeploymentConfig.from_dict(
        {
            "resources": {
                "runtime_dir": str(runtime),
                "episodes": {"root": str(episodes)},
                "media": {"root": str(runtime / "media")},
            },
            "robots": {"mock0": {"type": "mock"}},
            "agents": {"main": {"type": "robot_agent", "robot_id": "mock0"}},
        }
    )
    config_path = tmp_path / "deployment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "resources:",
                f"  runtime_dir: {runtime.as_posix()}",
                "  episodes:",
                f"    root: {episodes.as_posix()}",
                "  media:",
                f"    root: {(runtime / 'media').as_posix()}",
                "robots:",
                "  mock0:",
                "    type: mock",
                "agents:",
                "  main:",
                "    type: robot_agent",
                "    robot_id: mock0",
            ]
        ),
        encoding="utf-8",
    )
    store = JsonlEpisodeStore(config.resources.episodes_root)
    store.ensure("s1", EpisodeScope(agent_id="main"), ["agent:main"])
    store.append_user_turn(
        "s1", UserTurn(envelope=Envelope(episode_id="s1"), text="move")
    )
    event = SkillEvent(
        envelope=Envelope(episode_id="s1", robot_id="mock0"),
        skill_id="cmd1",
        phase="issued",
    )
    SkillStore(runtime / "skills").append(event)
    RobotEpisodeStateStore(config.resources.episodes_root).apply_skill_event(event)
    monkeypatch.setattr(
        sys,
        "argv",
        ["hey-robot", "inspect-episode", "s1", "--config", str(config_path)],
    )

    main()
    out = capsys.readouterr().out

    assert "episode: s1" in out
    assert "history: 1 records" in out
    assert "skills: 1" in out
