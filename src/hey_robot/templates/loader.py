from __future__ import annotations

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
    """从运行时覆盖项或随包默认值中解析并渲染 Prompt 模板。"""

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

    def render(self, name: str, **values: Any) -> str:
        rendered = self._env.get_template(_normalize_name(name)).render(**values)
        return str(rendered).strip()

    def _loader(self) -> ChoiceLoader:
        loaders: list[BaseLoader] = []
        if self.root is not None:
            loaders.append(FileSystemLoader(str(self.root)))
        loaders.append(PackageLoader(self.package, self.package_path))
        return ChoiceLoader(loaders)


def _normalize_name(name: str) -> str:
    normalized = str(name).replace("\\", "/").strip("/")
    if not normalized or normalized.startswith("../") or "/../" in normalized:
        raise ValueError(f"invalid template name: {name!r}")
    return normalized
