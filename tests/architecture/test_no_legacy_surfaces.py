from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = [ROOT / "src", ROOT / "configs", ROOT / "docs", ROOT / "README.md"]
FORBIDDEN = (
    re.compile(r"\bfrom\s+hey_robot\.agent(?:\s+|\.|$)"),
    re.compile(r"\bimport\s+hey_robot\.agent(?:\s+|\.|$)"),
    re.compile(r"\bhey_robot\.system2\b"),
    re.compile(r"\bhey_robot\.common\.configs\b"),
    re.compile(r"\bcommon\.configs\b"),
    re.compile(r"\bSystem2Config\b"),
    re.compile(r"\bHeyRobotConfig\b"),
    re.compile(r"\bDEPRECATED_SKILLS\b"),
    re.compile(r"\bDEFAULT_ROBOT_SKILL_CATALOG\b"),
    re.compile(r"\bDEFAULT_PUBLIC_SKILL_CATALOG\b"),
    re.compile(r"\bDEFAULT_SKILL_REGISTRY\b"),
    re.compile(r"\bhey_robot\.agents\.runtime\.runtime\b"),
    re.compile(r"\bhey_robot\.agents\.runtime\.mcp\b"),
    re.compile(r"\bhey_robot\.tasks\.supervisor\b"),
)

SOURCE_ONLY_FORBIDDEN = (
    re.compile(r"\bSkillPlanExpander\b"),
    re.compile(r"\bFoundationObservationAdapter\b"),
    re.compile(r"\bFoundationActionAdapter\b"),
    re.compile(r"\bFoundationIdentityAdapter\b"),
    re.compile(r"\bSkillExecutor\b"),
)

FORBIDDEN_PATH_PARTS = (
    Path("src/hey_robot/agents/runtime/runtime.py"),
    Path("src/hey_robot/agents/runtime/mcp.py"),
    Path("src/hey_robot/tasks/supervisor.py"),
    Path("tests/tasks/test_supervisor.py"),
)


def _iter_text_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if root.is_file():
            files.append(root)
            continue
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in {".py", ".md", ".yaml", ".yml", ".toml"}
        )
    return files


def test_removed_legacy_surfaces_do_not_reappear() -> None:
    offenders: list[str] = []
    offenders.extend(
        str(path) for path in FORBIDDEN_PATH_PARTS if (ROOT / path).exists()
    )
    for path in _iter_text_files():
        text = path.read_text(encoding="utf-8")
        offenders.extend(
            f"{path.relative_to(ROOT)}: {pattern.pattern}"
            for pattern in FORBIDDEN
            if pattern.search(text)
        )
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        offenders.extend(
            f"{path.relative_to(ROOT)}: {pattern.pattern}"
            for pattern in SOURCE_ONLY_FORBIDDEN
            if pattern.search(text)
        )

    assert offenders == []
