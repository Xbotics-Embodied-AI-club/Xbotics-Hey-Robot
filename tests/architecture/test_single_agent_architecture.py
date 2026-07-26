from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "hey_robot"


def _read(relative: str) -> str:
    return (SRC / relative).read_text(encoding="utf-8")


def test_deployment_starts_only_the_unified_robot_agent() -> None:
    source = _read("app/runner.py")
    assert "AutonomousAgentService" in source
    assert "AutonomousAutonomousAgentService" not in source
    assert "build_conversation_agent" not in source
    assert 'ManagedService(f"agent:{agent_id}"' in source
    assert 'f"conversation:{agent_id}"' not in source


def test_cli_has_one_agent_entrypoint() -> None:
    main = _read("cli/main.py")
    agent = _read("cli/agent.py")
    assert '"conversation-agent"' not in main
    assert '"agent": "hey_robot.cli.agent:main"' in main
    assert "--episode-dir" not in agent


def test_robot_agent_owns_one_model_client_and_one_runner() -> None:
    source = _read("cognition/autonomous_agent_service.py")
    assert source.count("create_model_client(") == 1
    assert source.count("AgentRunner(") == 1
    assert source.count("ToolRegistry(") == 1


def test_only_one_tool_registry_exists() -> None:
    tools = SRC / "cognition" / "tools"
    registries = [
        path.name
        for path in tools.glob("*.py")
        if "class ToolRegistry" in path.read_text(encoding="utf-8")
    ]
    assert registries == ["registry.py"]


def test_agent_core_has_no_migration_only_surfaces() -> None:
    cognition = SRC / "cognition"
    assert not (cognition / "runtime" / "trace.py").exists()
    assert not (cognition / "runtime" / "completion_verifier.py").exists()
    assert not (cognition / "conversation_entities.py").exists()
    assert not (cognition / "tools" / "dispatcher.py").exists()
    assert not (cognition / "tools" / "robot.py").exists()
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in cognition.rglob("*.py")
    )
    for removed in (
        "hard_max_continuations",
        "continuation_count",
        "_duplicate_observation_gate",
        "_reobservation_gate",
        "requires_reobservation",
        "entity_context",
        "_resume_with_observation",
        "def proposal(",
    ):
        assert removed not in source


def test_shared_runner_has_no_io_or_robot_dependencies() -> None:
    source = _read("cognition/runtime/agent_runner.py")
    assert "create_bus_client" not in source
    assert "SkillIntent" not in source
    assert "RobotAction" not in source
    assert "robot_runtime" not in source


def test_agent_never_constructs_physical_protocol_messages() -> None:
    source = _read("cognition/autonomous_agent_service.py")
    assert "SkillIntent(" not in source
    assert "RobotAction(" not in source


def test_agent_prompt_has_no_removed_tool_vocabulary() -> None:
    templates = SRC / "templates" / "agent"
    prompt = "\n".join(
        path.read_text(encoding="utf-8") for path in templates.glob("*.md")
    )
    forbidden = {
        "tool_instructions",
        "可用 Skill 契约",
        "complete_task",
        "control_task",
        "task_state",
        "inspect_scene",
        "observation Skill",
        "request_perception",
        "get_robot_status",
        "get_task_context",
        "search_memory",
        "write_memory",
        "propose_skill",
        "wait_policy",
    }
    assert {name for name in forbidden if name in prompt} == set()
    assert not (templates / "TURN.md").exists()


def test_context_projection_does_not_describe_specific_tools() -> None:
    source = _read("cognition/runtime/agent_context.py")
    for name in ("complete_task", "control_task", "inspect_scene"):
        assert name not in source


def test_robot_agent_loads_packaged_prompts() -> None:
    source = _read("cognition/runtime/agent_context.py")
    assert '"agent/SYSTEM.md"' in source
    assert 'self._templates.render("agent/SOUL.md")' in source
    assert "task_context=" in source


def test_soul_contains_persona_not_runtime_policy() -> None:
    soul = _read("templates/agent/SOUL.md")
    assert "我是小白" in soul
    for policy_term in ("工具", "运行", "执行失败", "证据", "安全限制"):
        assert policy_term not in soul


def test_template_package_exposes_only_the_used_store() -> None:
    source = _read("templates/__init__.py")
    loader = _read("templates/loader.py")
    assert '__all__ = ["TemplateStore"]' in source
    for removed in ("load_template", "render_template", "render_text", "def read("):
        assert removed not in loader


def test_perception_templates_follow_their_code_owner() -> None:
    templates = SRC / "templates"
    assert not (templates / "robot").exists()
    scene = templates / "perception" / "scene_captioner"
    assert (scene / "SYSTEM.md").is_file()
    assert (scene / "USER.md").is_file()
    source = _read("cognition/perception/scene/captioner.py")
    assert '"perception/scene_captioner/SYSTEM.md"' in source
    assert '"perception/scene_captioner/USER.md"' in source


def test_unused_motion_and_notification_frameworks_are_removed() -> None:
    assert not tuple((SRC / "motion").glob("*.py"))
    assert not tuple((SRC / "notifications").glob("*.py"))
    config = _read("config/model.py")
    assert "NotificationSpec" not in config
    assert "notifications:" not in "\n".join(
        path.read_text(encoding="utf-8") for path in (ROOT / "configs").glob("*.yaml")
    )


def test_unused_policy_and_compatibility_frameworks_are_removed() -> None:
    catalog = SRC / "foundation" / "catalog"
    assert not (catalog / "policy.py").exists()
    assert not (catalog / "resolver.py").exists()
    assert "NullEventPublisher" not in _read("events/bus.py")
    assert "class EpisodeStore" not in _read("episode/store.py")

    task_store = _read("cognition/runtime/agent_task_store.py")
    assert "ALTER TABLE" not in task_store
    assert "_migrate_" not in task_store
    assert "task_envelopes" not in task_store
