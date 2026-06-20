from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

from jinja2 import (
    ChoiceLoader,
    Environment,
    FileSystemLoader,
    PackageLoader,
    StrictUndefined,
)
from jinja2.loaders import BaseLoader


class TemplateStore:
    """Resolve and render prompt templates from runtime overrides or packaged defaults."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        package: str = "hey_robot",
        package_path: str = "templates",
    ) -> None:
        self.root = Path(root) if root is not None else None
        self.package = package
        self.package_path = package_path
        self._env = Environment(
            loader=self._loader(),
            autoescape=False,  # noqa: S701 - prompt templates render plain text, not HTML.
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined,
        )

    def read(self, name: str) -> str:
        normalized = _normalize_name(name)
        if self.root is not None:
            path = self.root / normalized
            if path.exists():
                return path.read_text(encoding="utf-8").strip()
        return (
            resources.files(self.package)
            .joinpath(self.package_path)
            .joinpath(normalized)
            .read_text(encoding="utf-8")
            .strip()
        )

    def render(self, name: str, **values: Any) -> str:
        return self._env.get_template(_normalize_name(name)).render(**values).strip()

    def render_text(self, text: str, **values: Any) -> str:
        return self._env.from_string(text).render(**values).strip()

    def _loader(self) -> ChoiceLoader:
        loaders: list[BaseLoader] = []
        if self.root is not None:
            loaders.append(FileSystemLoader(str(self.root)))
        loaders.append(PackageLoader(self.package, self.package_path))
        return ChoiceLoader(loaders)


def load_template(name: str, *, root: str | Path | None = None) -> str:
    return TemplateStore(root).read(name)


def render_template(name: str, *, root: str | Path | None = None, **values: Any) -> str:
    return TemplateStore(root).render(name, **values)


def _normalize_name(name: str) -> str:
    normalized = str(name).replace("\\", "/").strip("/")
    if not normalized or normalized.startswith("../") or "/../" in normalized:
        raise ValueError(f"invalid template name: {name!r}")
    return normalized
