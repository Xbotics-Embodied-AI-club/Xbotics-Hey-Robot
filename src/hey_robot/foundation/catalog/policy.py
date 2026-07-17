from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ToolPolicyBehavior = Literal["allow", "deny", "ask"]


@dataclass(frozen=True)
class ToolPolicyDecision:
    behavior: ToolPolicyBehavior
    reason: str
    rule: str = "default"


@dataclass(frozen=True)
class ToolPolicy:
    """运行时工具的声明式护栏。

    这层策略刻意保持小而明确，在更底层的权限检查和安全 hook 运行前决定工具是否可用。
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
        "request_skill",
        "wait",
    )

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> ToolPolicy:
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
                    "request_skill",
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
    ) -> ToolPolicyDecision:
        if self.allow_tools and tool_name not in self.allow_tools:
            return ToolPolicyDecision(
                "deny", f"tool is not in allow_tools: {tool_name}", "allow_tools"
            )
        if tool_name in self.deny_tools:
            return ToolPolicyDecision(
                "deny", f"tool is denied: {tool_name}", "deny_tools"
            )
        if source in self.deny_sources:
            return ToolPolicyDecision(
                "deny", f"source is denied: {source}", "deny_sources"
            )
        if safety_level in self.deny_safety_levels:
            return ToolPolicyDecision(
                "deny", f"safety level is denied: {safety_level}", "deny_safety_levels"
            )
        robot_state_blocked = (
            robot_state in self.deny_when_robot_states
            and not read_only
            and tool_name not in self.safe_on_blocked_robot
        )
        if robot_state_blocked:
            return ToolPolicyDecision(
                "deny",
                f"robot state blocks non-read-only tool: {robot_state}",
                "robot_state",
            )
        if safety_level in self.require_approval_for:
            return ToolPolicyDecision(
                "ask", f"safety level requires approval: {safety_level}", "approval"
            )
        return ToolPolicyDecision("allow", "tool policy allowed", "default")


@dataclass(frozen=True)
class ToolPolicySet:
    default: ToolPolicy = field(default_factory=ToolPolicy)
    modes: dict[str, ToolPolicy] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> ToolPolicySet:
        if not isinstance(payload, dict):
            return cls()
        mode_payloads = _mapping(payload.get("modes"))
        default_payload = {
            key: value for key, value in payload.items() if key != "modes"
        }
        return cls(
            default=ToolPolicy.from_dict(default_payload),
            modes={
                str(name): ToolPolicy.from_dict(value)
                for name, value in mode_payloads.items()
            },
        )

    def for_mode(self, mode: str | None) -> ToolPolicy:
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
