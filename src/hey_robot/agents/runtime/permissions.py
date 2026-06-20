from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from hey_robot.agents.runtime.registry import ToolSpec

PermissionMode = Literal["guarded", "autonomous", "manual", "dry_run"]


@dataclass(frozen=True)
class PermissionDecision:
    behavior: Literal["allow", "deny", "ask"]
    reason: str = ""
    updated_input: dict[str, Any] | None = None


class PermissionManager:
    """Small fail-closed permission gate for robot agent tools."""

    def __init__(self, mode: PermissionMode = "guarded") -> None:
        self.mode = mode

    def check(self, tool: ToolSpec, arguments: dict[str, Any]) -> PermissionDecision:
        if self.mode == "dry_run" and not tool.read_only:
            return PermissionDecision(
                behavior="deny",
                reason=f"dry_run blocks non-read-only tool: {tool.name}",
            )
        if self.mode == "manual" and not tool.read_only:
            return PermissionDecision(
                behavior="ask",
                reason=f"manual mode requires approval for tool: {tool.name}",
            )
        if self.mode == "guarded" and (
            tool.destructive
            or tool.safety_level in {"unsafe_actuate", "external_destructive"}
        ):
            return PermissionDecision(
                behavior="ask",
                reason=f"guarded mode requires approval for {tool.safety_level}: {tool.name}",
            )
        return PermissionDecision(
            behavior="allow",
            reason="allowed",
            updated_input=dict(arguments),
        )
