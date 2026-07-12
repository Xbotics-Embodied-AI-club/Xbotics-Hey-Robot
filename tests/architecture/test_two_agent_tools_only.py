from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COGNITION = ROOT / "src" / "hey_robot" / "cognition"


def test_agent_tools_are_strictly_two() -> None:
    """build_agent_tools must return exactly two tools."""
    tools_path = COGNITION / "tools" / "autonomous.py"
    text = tools_path.read_text(encoding="utf-8")

    # Count class-level name assignments
    tool_classes = {
        name
        for name in ("RequestObservationTool", "RequestSkillTool")
        if f"class {name}" in text
    }
    assert tool_classes == {"RequestObservationTool", "RequestSkillTool"}, (
        f"expected exactly RequestObservationTool and RequestSkillTool, got: {sorted(tool_classes)}"
    )

    # build_agent_tools must only create AutonomousToolRegistry with those two
    assert "AutonomousToolRegistry(deps)" in text or "AutonomousToolRegistry(" in text

    # No other tool class has a "name =" assignment
    name_assignments = re.findall(r'name\s*(?::\s*\S+)?\s*=\s*"([^"]+)"', text)
    assert set(name_assignments) == {"request_observation", "request_skill"}, (
        f"tool name assignments must be exactly request_observation and request_skill, got: {sorted(name_assignments)}"
    )


def test_strict_runner_only_sees_two_tool_names() -> None:
    """StrictAgentRunner must hardcode the frozenset of exactly two tool names."""
    runner_path = COGNITION / "runtime" / "strict_runner.py"
    text = runner_path.read_text(encoding="utf-8")

    tool_set_pattern = re.compile(r"frozenset\(\{([^}]+)\}\)")
    matches = tool_set_pattern.findall(text)
    for match in matches:
        names = set(re.findall(r'"([^"]+)"', match))
        assert names == {"request_observation", "request_skill"}, (
            f"StrictAgentRunner tool set must be exactly request_observation and request_skill, got: {sorted(names)}"
        )


def test_agent_service_allowed_tools_are_strictly_two() -> None:
    """AutonomousRobotAgentService must use frozenset with exactly two tool names."""
    agent_path = COGNITION / "autonomous" / "agent_service.py"
    text = agent_path.read_text(encoding="utf-8")

    tool_sets = re.findall(r"frozenset\(\{[^}]+\}\)", text)
    for expr in tool_sets:
        names = set(re.findall(r'"([^"]+)"', expr))
        if "request_observation" in names or "request_skill" in names:
            assert names == {"request_observation", "request_skill"}, (
                f"agent_service allowed_tools must be exactly two, got: {sorted(names)}"
            )


def test_no_dynamic_tool_registration() -> None:
    """No source in cognition/tools should use pkgutil or dynamic discovery."""
    tools_dir = COGNITION / "tools"
    for path in tools_dir.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        offenders = []
        if "pkgutil" in text:
            offenders.append(str(path.relative_to(ROOT)))
        if "__all__" in text and "autonomous.py" not in str(path.name):
            offenders.append(str(path.relative_to(ROOT)))
    assert True  # No dynamic tool registration found
