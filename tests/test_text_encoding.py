from __future__ import annotations

from pathlib import Path

TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".ini", ".txt"}
SCAN_ROOTS = [
    Path("src"),
    Path("tests"),
    Path("docs"),
    Path("configs"),
    Path("README.md"),
]

MOJIBAKE_MARKER_CODES = (
    0xFFFD,  # replacement character
    0x9435,
    0x934F,
    0x6D60,
    0x9239,
    0x9366,
    0x95C3,
    0x9A9E,
    0x9291,
    0x94BF,
    0x9365,
    0x6D93,
    0x9230,
    0x6500,
    0x20AC,
)
MOJIBAKE_MARKERS = tuple(chr(code) for code in MOJIBAKE_MARKER_CODES)


def iter_text_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if root.is_file():
            files.append(root)
            continue
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
        )
    return files


def test_first_party_text_files_are_utf8_without_mojibake() -> None:
    failures: list[str] = []
    for path in iter_text_files():
        text = path.read_text(encoding="utf-8")
        markers = [marker for marker in MOJIBAKE_MARKERS if marker in text]
        if markers:
            failures.append(f"{path}: {', '.join(markers)}")

    assert failures == []
