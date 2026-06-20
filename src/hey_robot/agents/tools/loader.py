"""Auto-discovery of :class:`~hey_robot.agents.tools.base.Tool` subclasses.

Scans the ``hey_robot.agents.tools`` package for concrete tool classes, then
instantiates and registers them. New tools are added by dropping a file into
this package; no edits to ``core.py`` are needed.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hey_robot.agents.tools.base import Tool
    from hey_robot.agents.tools.context import ToolContext
    from hey_robot.agents.tools.registry import ToolRegistry

logger = logging.getLogger("hey_robot.agents.tools.loader")

_SKIP_MODULES = frozenset(
    {
        "base",
        "schema",
        "registry",
        "context",
        "loader",
        "__init__",
    }
)

_PRODUCTION_TOOL_MODULES = frozenset(
    {
        "get_robot_status",
        "get_task_context",
        "propose_capability",
        "request_capability",
        "request_perception",
        "search_memory",
        "wait",
        "write_memory",
    }
)


class ToolLoader:
    """Discover and register all ``Tool`` subclasses in the tools package."""

    def __init__(
        self, package: Any = None, *, test_classes: list[type[Tool]] | None = None
    ):
        if package is None:
            import hey_robot.agents.tools as _pkg

            package = _pkg
        self._package = package
        self._test_classes = test_classes
        self._discovered: list[type[Tool]] | None = None

    def discover(self) -> list[type[Tool]]:
        """Find all concrete ``Tool`` subclasses in the package."""
        if self._test_classes is not None:
            return list(self._test_classes)
        if self._discovered is not None:
            return self._discovered

        from hey_robot.agents.tools.base import Tool

        seen: set[int] = set()
        results: list[type[Tool]] = []

        for _importer, module_name, _ispkg in pkgutil.iter_modules(
            self._package.__path__
        ):
            if module_name.startswith("_") or module_name in _SKIP_MODULES:
                continue
            if module_name not in _PRODUCTION_TOOL_MODULES:
                continue
            try:
                module = importlib.import_module(
                    f".{module_name}", self._package.__name__
                )
            except Exception:
                logger.exception("Failed to import tool module: %s", module_name)
                continue

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Tool)
                    and attr is not Tool
                    and not attr_name.startswith("_")
                    and not getattr(attr, "__abstractmethods__", None)
                    and getattr(attr, "_plugin_discoverable", True)
                    and id(attr) not in seen
                ):
                    seen.add(id(attr))
                    results.append(attr)

        results.sort(key=lambda cls: cls.__name__)
        self._discovered = results
        return results

    def load(self, ctx: ToolContext, registry: ToolRegistry) -> list[str]:
        """Discover tools, instantiate via ``create(ctx)``, and register all of them.

        Returns the list of registered tool names.
        """
        registered: list[str] = []
        for tool_cls in self.discover():
            cls_label = tool_cls.__name__
            try:
                if not tool_cls.enabled(ctx):
                    continue
                tool = tool_cls.create(ctx)
                registry.register(tool)
                registered.append(tool.name)
            except Exception:
                logger.exception("Failed to register tool: %s", cls_label)
        return registered
