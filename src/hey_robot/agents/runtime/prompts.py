from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from hey_robot.templates.loader import TemplateStore


@dataclass(frozen=True)
class AgentPromptTemplates:
    soul: str
    system: str
    turn: str
    store: TemplateStore = field(default_factory=TemplateStore, compare=False)


class AgentTemplateLoader:
    """Load agent prompt templates from a workspace folder or packaged defaults."""

    def __init__(
        self,
        template_root: str | Path | None = None,
    ) -> None:
        self.store = TemplateStore(template_root)

    def load(self, *, soul_path: str | Path | None = None) -> AgentPromptTemplates:
        return AgentPromptTemplates(
            soul=self._read_path(Path(soul_path))
            if soul_path is not None
            else self.store.read("agent/SOUL.md"),
            system=self.store.read("agent/SYSTEM.md"),
            turn=self.store.read("agent/TURN.md"),
            store=self.store,
        )

    @staticmethod
    def _read_path(path: Path) -> str:
        return path.expanduser().read_text(encoding="utf-8").strip()


def load_agent_prompt_templates(
    *,
    template_root: str | Path | None = None,
    soul_path: str | Path | None = None,
) -> AgentPromptTemplates:
    return AgentTemplateLoader(template_root).load(soul_path=soul_path)


def build_system_prompt(templates: AgentPromptTemplates) -> str:
    return templates.store.render_text(
        templates.system, agent_soul=templates.soul.strip()
    )


def build_turn_prompt(
    *,
    templates: AgentPromptTemplates,
    task: str,
    robot_state: str,
    memory_context: str | None = None,
    autonomy_context: str | None = None,
    last_feedback: str | None = None,
    next_hint: str | None = None,
    skill_in_progress: bool = False,
    recovery_context: str | None = None,
    loop_warning: str | None = None,
    task_contract_context: str | None = None,
) -> str:
    values = {
        "task": task,
        "robot_state": robot_state,
        "skill_status_context": _block(
            "Skill status",
            "A skill is currently executing. Wait before issuing new actuation skills.",
            enabled=skill_in_progress,
        ),
        "last_feedback": _block("Execution feedback", last_feedback),
        "recovery_context": _block("Recovery", recovery_context),
        "task_contract_context": _block("Task contract", task_contract_context),
        "next_hint": _block("Hint", next_hint),
        "loop_warning": _block("Loop warning", loop_warning),
        "memory_context": _raw_block(_trim_context(memory_context, max_chars=4000)),
        "autonomy_context": _raw_block(autonomy_context),
    }
    return _compact_prompt(templates.store.render_text(templates.turn, **values))


def _block(title: str, content: str | None, *, enabled: bool = True) -> str:
    if not enabled or not content or not content.strip():
        return ""
    return f"\n{title}: {content.strip()}\n"


def _raw_block(content: str | None) -> str:
    if not content or not content.strip():
        return ""
    return f"\n{content.strip()}\n"


def _trim_context(content: str | None, *, max_chars: int) -> str:
    if not content or not content.strip():
        return ""
    text = content.strip()
    if len(text) <= max_chars:
        return text
    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars
    return f"{text[:head_chars].rstrip()}\n...\n{text[-tail_chars:].lstrip()}"


def _compact_prompt(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    compacted: list[str] = []
    blank = False
    for line in lines:
        if line:
            compacted.append(line)
            blank = False
        elif not blank:
            compacted.append("")
            blank = True
    return "\n".join(compacted).strip()
