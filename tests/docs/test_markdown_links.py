from __future__ import annotations

import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")
HTML_LINK = re.compile(r"""(?:href|src)=["']([^"']+)["']""")
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "app://")


def _markdown_files() -> list[Path]:
    root_files = [path for path in REPOSITORY_ROOT.glob("*.md") if path.is_file()]
    return sorted([*root_files, *REPOSITORY_ROOT.joinpath("docs").rglob("*.md")])


def test_local_document_links_resolve() -> None:
    broken: list[str] = []
    for document in _markdown_files():
        text = document.read_text(encoding="utf-8")
        targets = [*MARKDOWN_LINK.findall(text), *HTML_LINK.findall(text)]
        for raw_target in targets:
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if (
                not target
                or raw_target.startswith("#")
                or target.startswith(EXTERNAL_PREFIXES)
            ):
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                relative_document = document.relative_to(REPOSITORY_ROOT)
                broken.append(f"{relative_document}: {raw_target}")

    assert not broken, "broken local document links:\n" + "\n".join(broken)
