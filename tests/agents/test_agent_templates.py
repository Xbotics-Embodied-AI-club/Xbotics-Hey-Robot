from __future__ import annotations

import pytest
from jinja2 import UndefinedError

from hey_robot.agents.runtime.prompts import (
    build_system_prompt,
    build_turn_prompt,
    load_agent_prompt_templates,
)
from hey_robot.templates import TemplateStore, load_template, render_template


def test_agent_templates_define_embodied_persona_and_message_channels() -> None:
    templates = load_agent_prompt_templates()
    system_prompt = build_system_prompt(templates)

    assert templates.soul.strip()
    assert (
        "`request_capability(capability, objective, slots, interrupt, wait_policy)`"
        in system_prompt
    )
    assert (
        "`request_perception(question, modality, scope, freshness, wait_policy)`"
        in system_prompt
    )
    assert "`get_robot_status(include_observation)`" in system_prompt
    assert "`get_task_context(detail_level)`" in system_prompt
    assert "`search_memory(query, kind, mode, limit)`" in system_prompt
    assert "`write_memory(kind, summary, name, location, ...)`" in system_prompt
    assert (
        "`propose_capability(capability, objective, slots, interrupt, confirmation_prompt)`"
        in system_prompt
    )
    assert "`wait(reason)`" in system_prompt
    assert "调用 `request_capability` 执行 `set_gripper`" in system_prompt
    assert "不要输出内部工具名、skill name、skill_id、trace_id" in system_prompt
    assert "request_quick_action" not in system_prompt
    assert "get_last_execution_feedback" not in system_prompt
    assert "get_recovery_options" not in system_prompt
    assert "notify_user" not in system_prompt


def test_template_store_uses_runtime_override_before_packaged_default(tmp_path) -> None:
    template_root = tmp_path / "templates"
    override = template_root / "robot" / "execution_feedback" / "SYSTEM.md"
    override.parent.mkdir(parents=True)
    override.write_text("custom {{ value }}", encoding="utf-8")

    store = TemplateStore(template_root)

    assert (
        store.render("robot/execution_feedback/SYSTEM.md", value="prompt")
        == "custom prompt"
    )
    assert "我是 小白" in store.read("agent/SOUL.md")


def test_template_store_supports_includes_and_strict_undefined(tmp_path) -> None:
    template_root = tmp_path / "templates"
    snippet = template_root / "agent" / "_snippets" / "name.md"
    parent = template_root / "agent" / "parent.md"
    snippet.parent.mkdir(parents=True)
    snippet.write_text("{{ robot_name }}", encoding="utf-8")
    parent.write_text(
        "Robot: {% include 'agent/_snippets/name.md' %}", encoding="utf-8"
    )
    store = TemplateStore(template_root)

    assert store.render("agent/parent.md", robot_name="Hey Robot") == "Robot: Hey Robot"
    with pytest.raises(UndefinedError):
        store.render("agent/parent.md")


def test_agent_templates_can_be_overridden_from_template_root(tmp_path) -> None:
    template_root = tmp_path / "templates"
    agent_dir = template_root / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "SOUL.md").write_text("I am a lab robot.", encoding="utf-8")
    (agent_dir / "SYSTEM.md").write_text("SOUL:\n{{ agent_soul }}", encoding="utf-8")
    (agent_dir / "TURN.md").write_text(
        "Task={{ task }}\nState={{ robot_state }}\n{{ next_hint }}",
        encoding="utf-8",
    )

    templates = load_agent_prompt_templates(template_root=template_root)
    system_prompt = build_system_prompt(templates)
    turn_prompt = build_turn_prompt(
        templates=templates,
        task="inspect",
        robot_state="idle",
        next_hint="use camera",
    )

    assert system_prompt == "SOUL:\nI am a lab robot."
    assert "Task=inspect" in turn_prompt
    assert "State=idle" in turn_prompt
    assert "Hint: use camera" in turn_prompt


def test_build_turn_prompt_includes_loop_warning_when_present(tmp_path) -> None:
    template_root = tmp_path / "templates"
    agent_dir = template_root / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "SOUL.md").write_text("I am a lab robot.", encoding="utf-8")
    (agent_dir / "SYSTEM.md").write_text("SOUL:\n{{ agent_soul }}", encoding="utf-8")
    (agent_dir / "TURN.md").write_text(
        "Task={{ task }}\nState={{ robot_state }}\n{{ loop_warning }}",
        encoding="utf-8",
    )

    templates = load_agent_prompt_templates(template_root=template_root)
    turn_prompt = build_turn_prompt(
        templates=templates,
        task="inspect",
        robot_state="idle",
        loop_warning="avoid retrying the same failed grasp",
    )

    assert "Task=inspect" in turn_prompt
    assert "Loop warning: avoid retrying the same failed grasp" in turn_prompt


def test_template_helpers_render_packaged_defaults_and_reject_path_escape() -> None:
    assert "我是 小白" in load_template("agent/SOUL.md")
    assert (
        render_template(
            "agent/TURN.md",
            task="inspect",
            robot_state="idle",
            skill_status_context="",
            last_feedback="",
            recovery_context="",
            task_contract_context="",
            next_hint="",
            loop_warning="",
            memory_context="",
            autonomy_context="",
        )
        != ""
    )

    with pytest.raises(ValueError, match="invalid template name"):
        load_template("../secrets.md")
