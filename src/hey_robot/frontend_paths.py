from __future__ import annotations

from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def frontend_root() -> Path:
    return repository_root() / "frontend"
