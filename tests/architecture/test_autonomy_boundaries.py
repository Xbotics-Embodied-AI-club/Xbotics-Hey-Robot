from __future__ import annotations

import re
from pathlib import Path

from hey_robot.cognition.tools.robot import ToolDependencies, ToolRegistry
from hey_robot.skill_os.base import SkillCatalog

ROOT = Path(__file__).resolve().parents[2]
COGNITION_ROOT = ROOT / "src" / "hey_robot" / "cognition"


def _cognition_source_files() -> list[Path]:
    return sorted(path for path in COGNITION_ROOT.rglob("*.py") if path.is_file())


def test_robot_agent_has_one_canonical_tool_registry() -> None:
    registry = ToolRegistry(ToolDependencies(SkillCatalog(())))
    names = {definition["function"]["name"] for definition in registry.definitions}
    assert names == {
        "request_observation",
        "request_skill",
        "complete_task",
        "control_task",
    }


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


def test_removed_supervisor_path_does_not_exist() -> None:
    assert not (COGNITION_ROOT / "autonomous").exists()
