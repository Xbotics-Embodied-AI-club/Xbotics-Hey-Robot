from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

ToolCallable = Callable[..., Any]
ToolResultPolicy = Literal[
    "continue_reasoning", "require_final_answer", "return_direct"
]


@dataclass(frozen=True)
class ToolSpec:
    """Immutable description of one agent tool.

    Produced by :class:`~hey_robot.agents.tools.base.Tool.to_spec` and
    consumed by the capability / permission / execution pipeline.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    func: ToolCallable
    read_only: bool = False
    destructive: bool = False
    safety_level: str = "normal"
    timeout_sec: float | None = None
    source: str = "local"
    exclusive: bool = False
    resources: tuple[str, ...] = ()
    result_policy: ToolResultPolicy = "continue_reasoning"

    @property
    def concurrency_safe(self) -> bool:
        return self.read_only and not self.destructive and not self.exclusive
