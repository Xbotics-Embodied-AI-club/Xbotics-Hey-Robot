from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COGNITION = ROOT / "src" / "hey_robot" / "cognition"


def test_no_provider_framework_in_src() -> None:
    providers = ROOT / "src" / "hey_robot" / "providers"
    assert not providers.exists(), "model provider framework must be deleted"


def test_no_text_fallback_in_runtime() -> None:
    """AgentRunner must not contain text fallback, synthesis, or repair."""
    runner_path = COGNITION / "runtime" / "agent_runner.py"
    text = runner_path.read_text(encoding="utf-8")

    forbidden = (
        "_fallback",
        "_synthesize",
        "_repair",
        "final_answer",
        "safe_tool_result",
        "looks_like_final",
        "internal_protocol_retry",
        "orphan",
        "backfill",
    )
    for pattern in forbidden:
        assert pattern not in text, (
            f"agent_runner.py contains forbidden fallback pattern: {pattern!r}"
        )


def test_no_legacy_autonomy_in_cognition() -> None:
    """Legacy autonomy files must not exist."""
    forbidden_paths = (
        COGNITION / "autonomy.py",
        COGNITION / "container.py",
        COGNITION / "core.py",
        COGNITION / "core_builder.py",
        COGNITION / "loop.py",
        COGNITION / "robot_agent.py",
        COGNITION / "task_supervisor.py",
        COGNITION / "task_contract.py",
        COGNITION / "task_events.py",
        COGNITION / "task_safety.py",
        COGNITION / "task_run.py",
        COGNITION / "session.py",
        COGNITION / "skill_gateway.py",
        COGNITION / "command_router.py",
        COGNITION / "checkpoint.py",
        COGNITION / "interaction.py",
        COGNITION / "execution_feedback.py",
        COGNITION / "scene_evidence.py",
        COGNITION / "scene_runtime.py",
        COGNITION / "notification_runtime.py",
        COGNITION / "perception_query.py",
        COGNITION / "busy_turn.py",
        COGNITION / "turn_policy.py",
        COGNITION / "progress.py",
        COGNITION / "recovery_capabilities.py",
        COGNITION / "injection.py",
        COGNITION / "io.py",
        COGNITION / "context.py",
        COGNITION / "types.py",
        COGNITION / "tool_binding.py",
        COGNITION / "memory_context.py",
        COGNITION / "skill_state.py",
        COGNITION / "service" / "recovery_notifier.py",
        COGNITION / "service" / "skill_result_handler.py",
        COGNITION / "task" / "state.py",
    )
    offenders = [str(p.relative_to(ROOT)) for p in forbidden_paths if p.exists()]
    assert offenders == [], f"legacy autonomy files still exist: {offenders}"


def test_no_legacy_tool_files() -> None:
    """Legacy tool files must not exist."""
    tools_dir = COGNITION / "tools"
    forbidden_tools = (
        "base.py",
        "context.py",
        "dispatcher.py",
        "get_robot_status.py",
        "get_task_context.py",
        "loader.py",
        "propose_skill.py",
        "request_perception.py",
        "request_skill.py",
        "robot.py",
        "schema.py",
        "search_memory.py",
        "task_introspection.py",
        "wait.py",
        "write_memory.py",
    )
    offenders = [
        str(tools_dir / name) for name in forbidden_tools if (tools_dir / name).exists()
    ]
    assert offenders == [], f"legacy tool files still exist: {offenders}"


def test_no_agent_fallback_imports_in_autonomous_path() -> None:
    """No cognition/autonomous source may import old fallback-related modules."""
    forbidden_imports = (
        re.compile(r"\bhey_robot\.cognition\.autonomy\b"),
        re.compile(r"\bhey_robot\.cognition\.core\b"),
        re.compile(r"\bhey_robot\.cognition\.loop\b"),
        re.compile(r"\bhey_robot\.cognition\.checkpoint\b"),
        re.compile(r"\bhey_robot\.cognition\.command_router\b"),
        re.compile(r"\bhey_robot\.cognition\.container\b"),
        re.compile(r"\bhey_robot\.cognition\.context\b"),
        re.compile(r"\bhey_robot\.cognition\.injection\b"),
        re.compile(r"\bhey_robot\.cognition\.session\b"),
    )
    offenders: list[str] = []
    for path in (COGNITION / "autonomous").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        offenders.extend(
            f"{path.relative_to(ROOT)}: {pattern.pattern}"
            for pattern in forbidden_imports
            if pattern.search(text)
        )
    assert offenders == [], f"autonomous path imports legacy modules: {offenders}"
