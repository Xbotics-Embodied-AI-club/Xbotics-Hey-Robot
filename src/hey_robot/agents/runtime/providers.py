from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from hey_robot.agents.tools.registry import ToolRegistry


class ToolProvider(Protocol):
    """A source that contributes tools to an agent runtime."""

    name: str
    source: str

    async def register(self, registry: ToolRegistry) -> list[str]: ...


@dataclass(frozen=True)
class ToolProviderInfo:
    name: str
    source: str
    tool_count: int
    tools: tuple[str, ...]
