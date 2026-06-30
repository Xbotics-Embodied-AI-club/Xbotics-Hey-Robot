from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

CapabilityBehavior = Literal["allow", "deny", "ask"]


@dataclass(frozen=True)
class CapabilityPolicyDecision:
    behavior: CapabilityBehavior
    reason: str
    rule: str = "default"


@dataclass(frozen=True)
class CapabilityPolicy:
    """Declarative guardrails for runtime capabilities.

    The policy is intentionally small and explicit. It governs capability use
    before lower-level permission checks and safety hooks run.
    """

    mode: str = "agent"
    allow_tools: tuple[str, ...] = ()
    deny_tools: tuple[str, ...] = ()
    deny_sources: tuple[str, ...] = ()
    deny_safety_levels: tuple[str, ...] = ()
    require_approval_for: tuple[str, ...] = ()
    deny_when_robot_states: tuple[str, ...] = ("estop", "emergency", "fault", "failed")
    safe_on_blocked_robot: tuple[str, ...] = (
        "get_task_context",
        "get_robot_status",
        "request_capability",
        "wait",
    )

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> CapabilityPolicy:
        if not isinstance(payload, dict):
            return cls()
        return cls(
            mode=str(payload.get("mode") or payload.get("runtime_mode") or "agent"),
            allow_tools=_tuple(payload.get("allow_tools")),
            deny_tools=_tuple(payload.get("deny_tools")),
            deny_sources=_tuple(payload.get("deny_sources")),
            deny_safety_levels=_tuple(payload.get("deny_safety_levels")),
            require_approval_for=_tuple(payload.get("require_approval_for")),
            deny_when_robot_states=_tuple(
                payload.get("deny_when_robot_states"),
                default=("estop", "emergency", "fault", "failed"),
            ),
            safe_on_blocked_robot=_tuple(
                payload.get("safe_on_blocked_robot"),
                default=(
                    "get_task_context",
                    "get_robot_status",
                    "request_capability",
                    "wait",
                ),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "allow_tools": list(self.allow_tools),
            "deny_tools": list(self.deny_tools),
            "deny_sources": list(self.deny_sources),
            "deny_safety_levels": list(self.deny_safety_levels),
            "require_approval_for": list(self.require_approval_for),
            "deny_when_robot_states": list(self.deny_when_robot_states),
            "safe_on_blocked_robot": list(self.safe_on_blocked_robot),
        }

    def decide(
        self,
        *,
        tool_name: str,
        source: str,
        safety_level: str,
        read_only: bool,
        robot_state: str | None = None,
    ) -> CapabilityPolicyDecision:
        if self.allow_tools and tool_name not in self.allow_tools:
            return CapabilityPolicyDecision(
                "deny", f"tool is not in allow_tools: {tool_name}", "allow_tools"
            )
        if tool_name in self.deny_tools:
            return CapabilityPolicyDecision(
                "deny", f"tool is denied: {tool_name}", "deny_tools"
            )
        if source in self.deny_sources:
            return CapabilityPolicyDecision(
                "deny", f"source is denied: {source}", "deny_sources"
            )
        if safety_level in self.deny_safety_levels:
            return CapabilityPolicyDecision(
                "deny", f"safety level is denied: {safety_level}", "deny_safety_levels"
            )
        robot_state_blocked = (
            robot_state in self.deny_when_robot_states
            and not read_only
            and tool_name not in self.safe_on_blocked_robot
        )
        if robot_state_blocked:
            return CapabilityPolicyDecision(
                "deny",
                f"robot state blocks non-read-only capability: {robot_state}",
                "robot_state",
            )
        if safety_level in self.require_approval_for:
            return CapabilityPolicyDecision(
                "ask", f"safety level requires approval: {safety_level}", "approval"
            )
        return CapabilityPolicyDecision("allow", "capability policy allowed", "default")


@dataclass(frozen=True)
class CapabilityPolicySet:
    default: CapabilityPolicy = field(default_factory=CapabilityPolicy)
    modes: dict[str, CapabilityPolicy] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> CapabilityPolicySet:
        if not isinstance(payload, dict):
            return cls()
        mode_payloads = _mapping(payload.get("modes"))
        default_payload = {
            key: value for key, value in payload.items() if key != "modes"
        }
        return cls(
            default=CapabilityPolicy.from_dict(default_payload),
            modes={
                str(name): CapabilityPolicy.from_dict(value)
                for name, value in mode_payloads.items()
            },
        )

    def for_mode(self, mode: str | None) -> CapabilityPolicy:
        if mode and mode in self.modes:
            return self.modes[mode]
        return self.default


def _tuple(value: Any, *, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple | set):
        return tuple(str(item) for item in value)
    return default


def _mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}
