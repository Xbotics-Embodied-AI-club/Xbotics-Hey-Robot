from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from hey_robot.foundation.catalog.policy import (
    CapabilityPolicy,
    CapabilityPolicyDecision,
)

CapabilityDecisionBehavior = Literal["allow", "deny", "ask"]


class ToolSpecLike(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def source(self) -> str: ...

    @property
    def safety_level(self) -> str: ...

    @property
    def read_only(self) -> bool: ...

    @property
    def input_schema(self) -> dict[str, Any]: ...

    @property
    def timeout_sec(self) -> float | None: ...


class ToolRegistryLike(Protocol):
    def get_tool(self, name: str) -> ToolSpecLike: ...


@dataclass(frozen=True)
class CapabilityResolution:
    behavior: CapabilityDecisionBehavior
    reason: str
    rule: str
    tool: ToolSpecLike | None = None
    source: str = ""
    safety_level: str = ""
    read_only: bool = False

    @property
    def allowed(self) -> bool:
        return self.behavior == "allow"


class CapabilityResolver:
    """Resolve whether a runtime capability may be used in the current context."""

    def __init__(
        self, registry: ToolRegistryLike, *, policy: CapabilityPolicy | None = None
    ) -> None:
        self.registry = registry
        self.policy = policy or CapabilityPolicy()

    def resolve(
        self, name: str, *, context: dict[str, Any] | None = None
    ) -> CapabilityResolution:
        try:
            tool = self.registry.get_tool(name)
        except ValueError as exc:
            return CapabilityResolution(
                behavior="deny",
                reason=str(exc),
                rule="tool_exists",
            )
        decision = self.policy.decide(
            tool_name=tool.name,
            source=tool.source,
            safety_level=tool.safety_level,
            read_only=tool.read_only,
            robot_state=_robot_state(context),
        )
        return _resolution_from_decision(decision, tool)


def _resolution_from_decision(
    decision: CapabilityPolicyDecision, tool: ToolSpecLike
) -> CapabilityResolution:
    return CapabilityResolution(
        behavior=decision.behavior,
        reason=decision.reason,
        rule=decision.rule,
        tool=tool,
        source=tool.source,
        safety_level=tool.safety_level,
        read_only=tool.read_only,
    )


def _robot_state(context: dict[str, Any] | None) -> str | None:
    if not isinstance(context, dict):
        return None
    status = context.get("robot_status")
    if isinstance(status, dict):
        value = status.get("state") or status.get("status")
        return str(value).strip().lower() if value is not None else None
    return None
