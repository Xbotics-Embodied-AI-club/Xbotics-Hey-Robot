from __future__ import annotations

from pathlib import Path

from hey_robot.config import DeploymentConfig, validation
from hey_robot.config.validation import validate_deployment
from hey_robot.episode import JsonlEpisodeStore, allocate_episode
from hey_robot.episode.scope import DEFAULT_EPISODE_DIMENSIONS
from hey_robot.protocol import AgentReply, Envelope, UserTurn
from hey_robot.protocol.messages import from_payload, to_payload
from hey_robot.skills.base import SkillCatalog, SkillSpec
from hey_robot.skills.registry import SkillRegistry


def test_deployment_config_loads_new_topology() -> None:
    config = DeploymentConfig.from_yaml("configs/mock.test.yaml")

    assert config.deployment.id == "mock-test"
    assert config.channels["cli"].type == "cli"
    assert config.robots["mock0"].type == "mock"
    assert config.robots["mock0"].robot_family == "xlerobot"
    assert config.robots["mock0"].robot_environment == "mock"
    assert config.robots["mock0"].driver_kind == "mock"
    assert config.robots["mock0"].embodiment_profile == "xlerobot_mock"
    assert config.policies["embodied_skills"].robot_id == "mock0"
    assert config.agents["main"].robot_id == "mock0"


def test_policy_settings_load_from_current_mock_config() -> None:
    config = DeploymentConfig.from_yaml("configs/mock.test.yaml")

    policy = config.policies["embodied_skills"]
    assert policy.type == "skill"
    assert policy.robot_id == "mock0"
    assert policy.settings["codec"] == "skill"
    assert policy.settings["body"] == "xlerobot"
    assert config.skills.mode == "bringup"
    assert config.skills.enabled[0] == "inspect_scene"
    assert "human_follow" in config.skills.enabled
    assert "set_gripper" not in config.skills.enabled


def test_deployment_validation_requires_explicit_skill_surface() -> None:
    config = DeploymentConfig.from_dict({"robots": {"mock0": {"type": "mock"}}})

    issues = validate_deployment(config)

    assert any("skills.enabled must explicitly list" in item.message for item in issues)


def test_deployment_validation_rejects_unknown_enabled_skill() -> None:
    config = DeploymentConfig.from_dict({"skills": {"enabled": ["missing_skill"]}})

    issues = validate_deployment(config)

    assert any("unknown skill missing_skill" in item.message for item in issues)


def test_deployment_validation_rejects_implementation_skill_in_production() -> None:
    config = DeploymentConfig.from_dict(
        {"skills": {"mode": "production", "enabled": ["move_base"]}}
    )

    issues = validate_deployment(config)

    assert any("implementation-level" in item.message for item in issues)


def test_deployment_validation_checks_transitive_skill_dependencies(
    monkeypatch,
) -> None:
    catalog = SkillCatalog(
        (
            SkillSpec(
                name="public_nav",
                description="public navigation skill",
                category="navigation",
                agent_visible=True,
                dependencies=("mid_nav",),
            ),
            SkillSpec(
                name="mid_nav",
                description="intermediate implementation",
                category="navigation",
                agent_visible=False,
                dependencies=("missing_leaf",),
            ),
        )
    )
    registry = SkillRegistry()
    for contract in catalog.list():
        registry.register_spec(contract)
    registry = registry.configure(enabled=("public_nav",))
    monkeypatch.setattr(validation, "registry_from_config", lambda _config: registry)
    config = DeploymentConfig.from_dict(
        {"skills": {"mode": "production", "enabled": ["public_nav"]}}
    )

    issues = validate_deployment(config)

    assert any(
        "skill public_nav references unknown dependency missing_leaf" in item.message
        for item in issues
    )


def test_identity_settings_load_from_mock_test_config() -> None:
    config = DeploymentConfig.from_yaml("configs/mock.test.yaml")

    assert config.identity.enabled is True
    assert config.identity.unified_user_episodes is True
    assert config.identity.bindings["cli:sender:local-user"] == "owner"
    assert config.identity.bindings["voice:sender:voice-user"] == "owner"


def test_notification_settings_load_from_root_section() -> None:
    config = DeploymentConfig.from_dict(
        {
            "notifications": {
                "defaults": {"channels": ["web"]},
                "channels": {
                    "voice": {"chat_id": "voice-room", "sender_id": "voice-user"}
                },
                "kinds": {
                    "task_watchdog": {
                        "severity": "critical",
                        "channels": ["web", "voice"],
                    }
                },
            }
        }
    )

    assert config.notifications.defaults["channels"] == ["web"]
    assert config.notifications.channels["voice"]["chat_id"] == "voice-room"
    assert config.notifications.kinds["task_watchdog"]["severity"] == "critical"


def test_default_agent_robot_and_episode_allocation_are_stable(tmp_path: Path) -> None:
    config = DeploymentConfig.from_yaml("configs/mock.test.yaml")
    turn = UserTurn(
        envelope=Envelope(
            channel="cli",
            chat_id="chat-1",
            chat_type="direct",
            sender_id="user-1",
        ),
        text="pick up the block",
    )

    agent_id = config.default_agent_id()
    robot_id = config.default_robot_id(agent_id)
    assert agent_id == "main"
    assert robot_id == "mock0"

    allocation = allocate_episode(
        turn.envelope.child(agent_id=agent_id, robot_id=robot_id),
        agent_id=agent_id,
        dimensions=DEFAULT_EPISODE_DIMENSIONS,
    )
    again = allocate_episode(
        turn.envelope.child(agent_id=agent_id, robot_id=robot_id),
        agent_id=agent_id,
        dimensions=DEFAULT_EPISODE_DIMENSIONS,
    )
    assert allocation.episode_id == again.episode_id

    store = JsonlEpisodeStore(tmp_path)
    store.ensure(allocation.episode_id, allocation.scope, allocation.aliases)
    store.append_user_turn(allocation.episode_id, turn)
    reply = AgentReply(
        envelope=turn.envelope.child(episode_id=allocation.episode_id), text="ok"
    )
    store.append_agent_reply(allocation.episode_id, reply)

    history = store.history(allocation.episode_id)
    assert [item.role for item in history] == ["user", "assistant"]


def test_deployment_defaults_cover_single_robot_agent_edge_cases() -> None:
    secondary_only = DeploymentConfig.from_dict(
        {
            "agents": {"ops": {"type": "robot_agent", "robot_id": "arm0"}},
            "robots": {"arm0": {"type": "mock"}},
        }
    )
    no_agents = DeploymentConfig.from_dict({"robots": {"mock0": {"type": "mock"}}})
    no_robots = DeploymentConfig.from_dict({})

    assert secondary_only.default_agent_id() == "ops"
    assert secondary_only.default_robot_id("ops") == "arm0"
    assert secondary_only.default_robot_id("missing") == "arm0"
    assert no_agents.default_agent_id() == "main"
    assert no_agents.default_robot_id() == "mock0"
    assert no_robots.default_robot_id() is None


def test_robot_identity_fields_load_and_derive_from_config() -> None:
    config = DeploymentConfig.from_dict(
        {
            "robots": {
                "sim0": {
                    "type": "xlerobot_sim",
                    "family": "xlerobot",
                    "environment": "sim",
                    "driver": "mujoco",
                    "embodiment_profile": "xlerobot_sim",
                },
                "mock0": {
                    "type": "mock",
                    "body": "xlerobot",
                },
            }
        }
    )

    sim0 = config.robots["sim0"]
    mock0 = config.robots["mock0"]

    assert sim0.robot_family == "xlerobot"
    assert sim0.robot_environment == "sim"
    assert sim0.driver_kind == "mujoco"
    assert sim0.embodiment_profile == "xlerobot_sim"

    assert mock0.robot_family == "xlerobot"
    assert mock0.robot_environment == "mock"
    assert mock0.driver_kind == "mock"


def test_protocol_payload_roundtrip() -> None:
    turn = UserTurn(envelope=Envelope(channel="cli", sender_id="u"), text="hello")
    restored = from_payload(UserTurn, to_payload(turn))

    assert restored == turn
