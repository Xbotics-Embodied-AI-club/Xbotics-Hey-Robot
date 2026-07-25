from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from hey_robot.cognition.tools.registry import ToolDependencies, ToolRegistry
from hey_robot.skills.models import Skill, SkillResult

ROOT = Path(__file__).resolve().parents[2]
COGNITION_ROOT = ROOT / "src" / "hey_robot" / "cognition"


class SkillList:
    def __init__(self, skills: tuple[Skill, ...]) -> None:
        self._skills = {skill.name: skill for skill in skills}

    def get(self, name: str) -> Skill:
        return self._skills[name]

    def list(self) -> tuple[Skill, ...]:
        return tuple(self._skills.values())


async def _noop(*_args: Any, **_kwargs: Any) -> SkillResult:
    return SkillResult(True, "ok", "completed")


def _cognition_source_files() -> list[Path]:
    return sorted(path for path in COGNITION_ROOT.rglob("*.py") if path.is_file())


def test_robot_agent_has_one_canonical_tool_registry() -> None:
    registry = ToolRegistry(ToolDependencies(()))
    names = {definition["function"]["name"] for definition in registry.definitions}
    assert names == set()


def test_cognition_path_does_not_import_legacy_tools() -> None:
    """No cognition source may import any deleted legacy tool."""
    forbidden_tools = (
        re.compile(r"\bget_robot_status\b"),
        re.compile(r"\bget_task_context\b"),
        re.compile(r"\bpropose_skill\b"),
        re.compile(r"\bsearch_memory\b"),
        re.compile(r"\bwrite_memory\b"),
        re.compile(r"\brequest_perception\b"),
    )
    offenders: list[str] = []
    for path in COGNITION_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        offenders.extend(
            f"{path.relative_to(ROOT)}: {pattern.pattern}"
            for pattern in forbidden_tools
            if pattern.search(text)
        )
    assert offenders == [], f"autonomous path imports legacy tools: {offenders}"


def test_agent_runner_does_not_import_task_contract() -> None:
    """AgentRunner must not import TaskContract or TaskEvaluator."""
    runner_path = COGNITION_ROOT / "runtime" / "agent_runner.py"
    text = runner_path.read_text(encoding="utf-8")
    forbidden = ("TaskContract", "TaskEvaluator", "EvidenceFact", "GoalSnapshot")
    offenders = [f"agent_runner.py: {name}" for name in forbidden if name in text]
    assert offenders == [], f"agent_runner imports task state: {offenders}"


def test_agent_service_does_not_publish_skill_intent() -> None:
    """AutonomousAgentService must never publish skill.intent directly."""
    agent_path = COGNITION_ROOT / "autonomous_agent_service.py"
    text = agent_path.read_text(encoding="utf-8")
    assert "skill_intent" not in text, "agent_service references skill_intent"
    assert "SkillIntent(" not in text, "agent_service constructs SkillIntent"


def test_agent_service_consumes_skill_client_events_not_bus_topic() -> None:
    agent_path = COGNITION_ROOT / "autonomous_agent_service.py"
    text = agent_path.read_text(encoding="utf-8")
    assert "self.topics.skill_run_event" not in text
    assert "_consume_skill_events" in text


def test_removed_supervisor_path_does_not_exist() -> None:
    assert not (COGNITION_ROOT / "autonomous").exists()
