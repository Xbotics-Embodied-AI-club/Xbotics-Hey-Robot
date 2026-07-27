from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

from hey_robot.config import DeploymentConfig
from hey_robot.config.validation import validate_deployment
from hey_robot.episode import JsonlEpisodeStore, allocate_episode
from hey_robot.episode.scope import DEFAULT_EPISODE_DIMENSIONS
from hey_robot.protocol import AgentReply, Envelope, UserTurn
from hey_robot.protocol.messages import from_payload, to_payload

XLEROBOT_DEV_CONFIGS = (
    "configs/xlerobot.real.s600.yaml",
    "configs/xlerobot.real.ubuntu.yaml",
    "configs/xlerobot.real.windows.yaml",
    "configs/xlerobot.sim.ubuntu.yaml",
    "configs/xlerobot.sim.windows.yaml",
)

MINIMAL_MOBILE_SKILLS = {"inspect_scene", "move_base", "turn_base"}
VLN_MOBILE_SKILLS = {"inspect_scene", "navigate_to", "approach_object"}


def test_runtime_dependencies_are_partitioned_by_container() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    core = tuple(project["dependencies"])
    extras = project["optional-dependencies"]

    assert not any(item.startswith(("torch", "ultralytics")) for item in core)
    assert not any(item.startswith(("fastapi", "openai", "opencv")) for item in core)
    assert any(item.startswith("fastapi") for item in extras["gateway"])
    assert any(item.startswith("openai") for item in extras["agent"])
    assert any(item.startswith("opencv-python") for item in extras["robot"])
    assert any(item.startswith("torch") for item in extras["human-follow"])


def test_lerobot_policy_has_a_generic_dependency_group_and_dockerfile() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    groups = project["dependency-groups"]

    assert "lerobot-policy" in groups
    assert "vla" not in groups
    assert Path("docker/Dockerfile.policy").is_file()
    assert not Path("docker/Dockerfile.vla").exists()
    policy_dockerfile = Path("docker/Dockerfile.policy").read_text(encoding="utf-8")
    assert "FROM python:${PYTHON_VERSION}-slim-bookworm" in policy_dockerfile
    assert 'ENTRYPOINT ["/app/.venv/bin/python"' in policy_dockerfile
    assert "pip install --break-system-packages \\\n    torch" not in policy_dockerfile


def test_vln_image_has_one_locked_cuda_runtime() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    vln_dependencies = "\n".join(project["dependency-groups"]["vln"])
    dockerfile = Path("docker/Dockerfile.vln").read_text(encoding="utf-8")
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))

    assert "nvidia-cuda-runtime-cu12" in vln_dependencies
    assert "nvidia-cudnn-cu12" in vln_dependencies
    assert "torch==2.6.0" in vln_dependencies
    assert "torchvision==0.21.0" in vln_dependencies
    assert "opencv-python==4.10.0.84" in vln_dependencies
    assert "FROM python:${PYTHON_VERSION}-slim-bookworm" in dockerfile
    assert "FROM nvidia/cuda" not in dockerfile
    assert "CUDA_VERSION" not in compose["services"]["vln"]["build"]["args"]


def test_robocasa_uses_generic_lerobot_policy_image() -> None:
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    robocasa_policy = services["robocasa-policy"]

    assert "policy" not in services
    assert robocasa_policy["build"]["dockerfile"] == "docker/Dockerfile.policy"
    assert robocasa_policy["image"] == (
        "${HEY_ROBOT_POLICY_IMAGE:-hey-robot-policy:latest}"
    )
    assert robocasa_policy["runtime"] == "nvidia"
    assert "deploy" not in robocasa_policy
    assert services["robocasa365"]["build"]["dockerfile"] == (
        "docker/Dockerfile.robocasa365"
    )
    assert "robocasa" in robocasa_policy["profiles"]


def test_configs_do_not_use_direct_agent_mode() -> None:
    offenders = [
        str(path)
        for path in Path("configs").rglob("*.yaml")
        if "mode: direct" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_agent_tool_surface_does_not_use_legacy_capability_proposal_tool() -> None:
    offenders = [
        str(path)
        for root in ("src", "configs")
        for path in Path(root).rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".md", ".yaml"}
        and "propose_capability" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_source_does_not_export_legacy_capability_catalog_names() -> None:
    banned = (
        "CapabilityLoader",
        "CapabilityManifest",
        "CapabilityPolicy",
        "CapabilityPolicyDecision",
        "CapabilityPolicySet",
        "CapabilityResolution",
        "CapabilityResolver",
        "RobotSkillCapability",
        "ToolCapability",
        "capability_policy",
        "capability_resolver",
    )
    offenders: dict[str, list[str]] = {}
    for path in Path("src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        found = [item for item in banned if item in text]
        if found:
            offenders[str(path)] = found

    assert offenders == {}


def test_inspect_cli_uses_skill_surface_command_name() -> None:
    text = Path("src/hey_robot/cli/inspect.py").read_text(encoding="utf-8")

    assert '"skill-surface"' in text
    assert '"capabilities"' not in text


def test_xlerobot_runtime_configs_use_minimal_mobile_skill_surface() -> None:
    offenders: dict[str, list[str]] = {}
    for path in XLEROBOT_DEV_CONFIGS:
        config = DeploymentConfig.from_yaml(path)
        visible = set(config.skills.tools)
        if config.skills.mode != "bringup" or visible != MINIMAL_MOBILE_SKILLS:
            offenders[path] = [
                f"mode={config.skills.mode}",
                f"visible={','.join(sorted(visible))}",
            ]

    assert offenders == {}


def test_xlerobot_vln_config_exposes_only_navigation_options() -> None:
    config = DeploymentConfig.from_yaml("configs/xlerobot.sim.vln.yaml")

    assert config.skills.mode == "bringup"
    assert set(config.skills.tools) == VLN_MOBILE_SKILLS


def test_protocol_does_not_export_skill_contract_runtime() -> None:
    import hey_robot.protocol as protocol
    import hey_robot.protocol.skills as protocol_skills

    assert not hasattr(protocol, "FeedbackMode")
    assert not hasattr(protocol, "RobotSkillCatalog")
    assert not hasattr(protocol, "RobotSkillSpec")
    assert not hasattr(protocol, "SkillContractDecision")
    assert not hasattr(protocol, "SkillContractRuntime")
    assert not hasattr(protocol_skills, "FeedbackMode")
    assert not hasattr(protocol_skills, "RobotSkillCatalog")
    assert not hasattr(protocol_skills, "RobotSkillSpec")
    assert not hasattr(protocol_skills, "SkillContractDecision")
    assert not hasattr(protocol_skills, "SkillContractRuntime")


def test_skill_os_does_not_keep_contract_forwarding_modules() -> None:
    assert not Path("src/hey_robot/skill_os/catalog.py").exists()
    assert not Path("src/hey_robot/skill_os/contracts.py").exists()

    offenders = [
        str(path)
        for path in Path("src").rglob("*.py")
        if (
            "hey_robot.skill_os.catalog" in path.read_text(encoding="utf-8")
            or "hey_robot.skill_os.contracts" in path.read_text(encoding="utf-8")
        )
    ]

    assert offenders == []


def test_single_agent_service_has_one_message_entrypoint() -> None:
    import importlib.util

    from hey_robot.cognition.autonomous_agent_service import AutonomousAgentService

    assert importlib.util.find_spec("hey_robot.cognition.core") is None
    assert importlib.util.find_spec("hey_robot.cognition.core_builder") is None
    assert hasattr(AutonomousAgentService, "_on_turn")
    assert not hasattr(AutonomousAgentService, "_on_deliberation")


def test_agent_runner_replaces_removed_model_loops() -> None:
    import importlib.util

    from hey_robot.cognition.runtime.agent_runner import AgentRunner

    assert importlib.util.find_spec("hey_robot.cognition.runtime.model_loop") is None
    assert importlib.util.find_spec("hey_robot.cognition.runtime.runner") is None
    assert hasattr(AgentRunner, "run")


def test_robot_runtime_does_not_depend_on_skill_os() -> None:
    offenders = [
        str(path)
        for path in Path("src/hey_robot/robot_runtime").rglob("*.py")
        if "hey_robot.skill_os" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_production_code_does_not_import_skill_os() -> None:
    offenders: list[str] = []
    for root in (Path("src/hey_robot"), Path("scripts")):
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            imports_skill_os = re.search(
                r"\bfrom\s+hey_robot\.skill_os(?:\s+|\.|$)", text
            ) or re.search(r"\bimport\s+hey_robot\.skill_os(?:\s+|\.|$)", text)
            if imports_skill_os:
                offenders.append(str(path))

    assert offenders == []
    assert not Path("src/hey_robot/skill_os").exists()


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


def test_mock_config_uses_native_local_skill_surface() -> None:
    config = DeploymentConfig.from_yaml("configs/mock.test.yaml")

    policy = config.policies["embodied_skills"]
    assert policy.robot_id == "mock0"
    assert policy.freq_hz == 10.0
    assert config.skills.mode == "bringup"
    assert config.skills.execution_mode == "local"
    assert config.skills.tools[0] == "inspect_scene"
    assert "human_follow" not in config.skills.tools
    assert "set_gripper" not in config.skills.tools


def test_runtime_configs_use_native_local_surface() -> None:
    migrated_configs = (
        "configs/mock.test.yaml",
        "configs/mock.dev.yaml",
        "configs/mock.compose.yaml",
        "configs/xlerobot.sim.ubuntu.yaml",
        "configs/xlerobot.sim.windows.yaml",
        "configs/xlerobot.sim.vln.yaml",
        "configs/xlerobot.real.ubuntu.yaml",
        "configs/xlerobot.real.windows.yaml",
        "configs/xlerobot.real.s600.yaml",
        "configs/evaluation/robocasa365.yaml",
    )

    for path in migrated_configs:
        config = DeploymentConfig.from_yaml(path)
        assert config.skills.modules == ("hey_robot.skills.builtins",), path
        assert config.skills.execution_mode == "local", path
        assert config.skills.tools, path
        assert "human_follow" not in config.skills.tools, path
        assert not [
            issue for issue in validate_deployment(config) if issue.level == "error"
        ], path


def test_compose_mock_config_uses_split_service_addresses() -> None:
    config = DeploymentConfig.from_yaml("configs/mock.compose.yaml")

    web = config.channels["web"]
    assert config.deployment.bus.url == "nats://nats:4222"
    assert web.settings["host"] == "0.0.0.0"  # noqa: S104
    assert web.settings["port"] == 8080
    assert web.settings["serve_frontend"] is False


def test_robocasa365_exposes_validated_native_vla_surface() -> None:
    config = DeploymentConfig.from_yaml("configs/evaluation/robocasa365.yaml")

    assert config.skills.modules == ("hey_robot.skills.builtins",)
    assert config.skills.execution_mode == "local"
    assert config.skills.tools == ("inspect_scene", "manipulate")


def test_deployment_validation_requires_explicit_skill_surface() -> None:
    config = DeploymentConfig.from_dict({"robots": {"mock0": {"type": "mock"}}})

    issues = validate_deployment(config)

    assert any("skills.tools must explicitly list" in item.message for item in issues)


def test_deployment_config_accepts_tools_and_rejects_unknown_skill_fields() -> None:
    config = DeploymentConfig.from_dict({"skills": {"tools": ["inspect_scene"]}})

    assert config.skills.tool_names == ("inspect_scene",)
    assert not [
        issue for issue in validate_deployment(config) if issue.level == "error"
    ]

    with pytest.raises(ValueError, match="skills uses unknown fields"):
        DeploymentConfig.from_dict(
            {"skills": {"tools": ["inspect_scene"], "legacy": ["manipulate"]}}
        )


def test_deployment_config_rejects_removed_event_driven_skill_mode() -> None:
    config = DeploymentConfig.from_dict(
        {"skills": {"tools": ["inspect_scene"], "execution_mode": "event_driven"}}
    )

    assert config.skills.execution_mode == "event_driven"
    assert any(
        "只支持 'local'" in issue.message
        for issue in validate_deployment(config)
        if issue.level == "error"
    )


def test_deployment_config_exposes_local_native_skill_mode() -> None:
    config = DeploymentConfig.from_dict(
        {
            "skills": {
                "modules": ["hey_robot.skills.builtins"],
                "tools": ["inspect_scene"],
                "execution_mode": "local",
            }
        }
    )

    assert config.skills.execution_mode == "local"
    assert not [
        issue for issue in validate_deployment(config) if issue.level == "error"
    ]


def test_deployment_validation_rejects_local_mode_with_legacy_modules() -> None:
    config = DeploymentConfig.from_dict(
        {
            "skills": {
                "modules": ["hey_robot.skills.legacy_builtins"],
                "tools": ["inspect_scene"],
                "execution_mode": "local",
            }
        }
    )

    assert any(
        "native hey_robot.skills.* modules" in issue.message
        for issue in validate_deployment(config)
        if issue.level == "error"
    )


def test_deployment_config_exposes_skill_implementations() -> None:
    config = DeploymentConfig.from_dict(
        {
            "skills": {
                "tools": ["inspect_scene"],
                "implementations": {"inspect_scene": "classic"},
            }
        }
    )

    assert config.skills.implementations == {"inspect_scene": "classic"}
    assert not [
        issue for issue in validate_deployment(config) if issue.level == "error"
    ]

    invalid = DeploymentConfig.from_dict(
        {
            "skills": {
                "tools": ["inspect_scene"],
                "implementations": {"manipulate": "vla"},
            }
        }
    )

    assert any(
        "non-surface skill manipulate" in issue.message
        for issue in validate_deployment(invalid)
    )


def test_deployment_validation_accepts_native_skill_modules() -> None:
    config = DeploymentConfig.from_dict(
        {
            "skills": {
                "modules": ["hey_robot.skills.builtins"],
                "tools": ["inspect_scene"],
            }
        }
    )

    assert config.skills.modules == ("hey_robot.skills.builtins",)
    assert not [
        issue for issue in validate_deployment(config) if issue.level == "error"
    ]


def test_deployment_validation_checks_native_skill_model_dependencies() -> None:
    config = DeploymentConfig.from_dict(
        {
            "skills": {
                "modules": ["hey_robot.skills.builtins"],
                "tools": ["manipulate"],
            }
        }
    )

    assert any(
        "requires unavailable model service manipulate" in issue.message
        for issue in validate_deployment(config)
    )


def test_identity_settings_load_from_mock_test_config() -> None:
    config = DeploymentConfig.from_yaml("configs/mock.test.yaml")

    assert config.identity.enabled is True
    assert config.identity.unified_user_episodes is True
    assert config.identity.bindings["cli:sender:local-user"] == "owner"
    assert config.identity.bindings["voice:sender:voice-user"] == "owner"


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
