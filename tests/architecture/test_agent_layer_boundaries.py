from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENTS_ROOT = ROOT / "src" / "hey_robot" / "agents"

ALLOWED_SKILL_INTENT_CONSTRUCTORS = {
    Path("src/hey_robot/agents/skill_gateway.py"),
    Path("src/hey_robot/agents/robot_agent.py"),
}


def _agent_source_files() -> list[Path]:
    return sorted(path for path in AGENTS_ROOT.rglob("*.py") if path.is_file())


def test_agent_layer_does_not_construct_skill_intents_outside_gateway() -> None:
    offenders: list[str] = []
    for path in _agent_source_files():
        rel_path = path.relative_to(ROOT)
        if rel_path in ALLOWED_SKILL_INTENT_CONSTRUCTORS:
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"\bSkillIntent\s*\(", text):
            offenders.append(str(rel_path))

    assert offenders == []


def test_agent_layer_does_not_depend_on_robot_actions_or_driver_primitives() -> None:
    forbidden = (
        re.compile(r"\bRobotAction\b"),
        re.compile(r"\bdriver_primitives\b"),
        re.compile(r"\bset_joint_positions\b"),
        re.compile(r"['\"]base\.forward['\"]"),
        re.compile(r"['\"]arm\.set_joint_positions['\"]"),
        re.compile(r"['\"]arm\."),
    )
    offenders: list[str] = []
    for path in _agent_source_files():
        text = path.read_text(encoding="utf-8")
        offenders.extend(
            f"{path.relative_to(ROOT)}: {pattern.pattern}"
            for pattern in forbidden
            if pattern.search(text)
        )

    assert offenders == []
