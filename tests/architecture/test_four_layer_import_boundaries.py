from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "hey_robot"


def _python_files(path: Path) -> list[Path]:
    return sorted(item for item in path.rglob("*.py") if item.is_file())


def _assert_no_imports(
    paths: list[Path], forbidden: tuple[re.Pattern[str], ...]
) -> None:
    offenders: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        offenders.extend(
            f"{path.relative_to(ROOT)}: {pattern.pattern}"
            for pattern in forbidden
            if pattern.search(text)
        )
    assert offenders == []


def test_robot_runtime_does_not_import_upper_layers() -> None:
    _assert_no_imports(
        _python_files(SRC / "robot_runtime"),
        (
            re.compile(r"\bfrom\s+hey_robot\.cognition(?:\s+|\.|$)"),
            re.compile(r"\bimport\s+hey_robot\.cognition(?:\s+|\.|$)"),
            re.compile(r"\bfrom\s+hey_robot\.skill_os(?:\s+|\.|$)"),
            re.compile(r"\bimport\s+hey_robot\.skill_os(?:\s+|\.|$)"),
            re.compile(r"\bfrom\s+hey_robot\.foundation\.backends(?:\s+|\.|$)"),
            re.compile(r"\bimport\s+hey_robot\.foundation\.backends(?:\s+|\.|$)"),
        ),
    )


def test_foundation_runtime_does_not_import_cognition_or_skill_os() -> None:
    _assert_no_imports(
        _python_files(SRC / "foundation"),
        (
            re.compile(r"\bfrom\s+hey_robot\.cognition(?:\s+|\.|$)"),
            re.compile(r"\bimport\s+hey_robot\.cognition(?:\s+|\.|$)"),
            re.compile(r"\bfrom\s+hey_robot\.skill_os(?:\s+|\.|$)"),
            re.compile(r"\bimport\s+hey_robot\.skill_os(?:\s+|\.|$)"),
        ),
    )
