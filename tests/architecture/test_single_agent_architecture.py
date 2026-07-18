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


def test_robot_agent_owns_one_provider_and_one_runner() -> None:
    source = _read("cognition/autonomous_agent_service.py")
    assert source.count("build_provider(") == 1
    assert source.count("AgentRunner(") == 1
    assert source.count("ToolRegistry(") == 1


def test_only_one_model_tool_registry_exists() -> None:
    tools = SRC / "cognition" / "tools"
    registries = [
        path.name
        for path in tools.glob("*.py")
        if "class ToolRegistry" in path.read_text(encoding="utf-8")
    ]
    assert registries == ["robot.py"]


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


def test_robot_agent_loads_packaged_prompts() -> None:
    source = _read("cognition/autonomous_agent_service.py")
    assert '"agent/SYSTEM.md"' in source
    assert 'self.templates.render("agent/SOUL.md")' in source
    assert "task_context=" in source


def test_template_package_exposes_only_the_used_store() -> None:
    source = _read("templates/__init__.py")
    loader = _read("templates/loader.py")
    assert '__all__ = ["TemplateStore"]' in source
    for removed in ("load_template", "render_template", "render_text", "def read("):
        assert removed not in loader
