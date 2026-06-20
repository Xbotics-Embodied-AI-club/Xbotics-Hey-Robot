from __future__ import annotations

from hey_robot.agents.tools.registry import ToolRegistry
from hey_robot.capability.catalog import CapabilityPolicy, CapabilityResolver


def test_capability_resolver_blocks_denied_safety_level() -> None:
    registry = ToolRegistry()
    registry.register_simple("move", lambda: "ok", safety_level="actuate")
    resolver = CapabilityResolver(
        registry, policy=CapabilityPolicy(deny_safety_levels=("actuate",))
    )

    decision = resolver.resolve("move")

    assert decision.behavior == "deny"
    assert decision.rule == "deny_safety_levels"
    assert "actuate" in decision.reason


def test_capability_resolver_blocks_non_read_only_when_robot_state_failed() -> None:
    registry = ToolRegistry()
    registry.register_simple("move", lambda: "ok", safety_level="actuate")

    decision = CapabilityResolver(registry).resolve(
        "move", context={"robot_status": {"state": "failed"}}
    )

    assert decision.behavior == "deny"
    assert decision.rule == "robot_state"


def test_capability_resolver_allows_request_capability_when_robot_state_failed() -> (
    None
):
    registry = ToolRegistry()

    def submit_capability(_capability: str, _objective: str) -> str:
        return "ok"

    registry.register_simple(
        "request_capability",
        submit_capability,
        safety_level="actuate",
    )

    decision = CapabilityResolver(registry).resolve(
        "request_capability", context={"robot_status": {"state": "failed"}}
    )

    assert decision.behavior == "allow"


def test_capability_resolver_denies_unknown_tool() -> None:
    registry = ToolRegistry()

    decision = CapabilityResolver(registry).resolve("missing")

    assert decision.behavior == "deny"
    assert decision.allowed is False
    assert decision.rule == "tool_exists"
    assert "Unknown tool: missing" in decision.reason


def test_capability_resolver_extracts_robot_state_from_status_payload() -> None:
    registry = ToolRegistry()
    registry.register_simple("move", lambda: "ok", safety_level="actuate")

    decision = CapabilityResolver(registry).resolve(
        "move", context={"robot_status": {"status": "failed"}}
    )

    assert decision.behavior == "deny"
    assert decision.rule == "robot_state"


def test_capability_resolver_allows_when_context_has_no_robot_state() -> None:
    registry = ToolRegistry()
    registry.register_simple("move", lambda: "ok", safety_level="actuate")

    decision = CapabilityResolver(registry).resolve("move", context={})

    assert decision.behavior == "allow"
    assert decision.allowed is True


def test_capability_resolver_ignores_textual_robot_summary_with_default_fields() -> (
    None
):
    registry = ToolRegistry()
    registry.register_simple("move", lambda: "ok", safety_level="actuate")

    decision = CapabilityResolver(registry).resolve(
        "move",
        context={
            "robot_state": (
                "robot_id=xlerobot state=idle "
                "metrics=hardware={default_arm=arm,default_camera=front}"
            )
        },
    )

    assert decision.behavior == "allow"
    assert decision.allowed is True


def test_capability_resolver_uses_structured_status_over_text_summary() -> None:
    registry = ToolRegistry()
    registry.register_simple("move", lambda: "ok", safety_level="actuate")

    decision = CapabilityResolver(registry).resolve(
        "move",
        context={
            "robot_state": "robot_id=xlerobot state=idle default_arm=arm",
            "robot_status": {
                "state": "failed",
                "metrics": {"hardware": {"default_arm": "arm"}},
            },
        },
    )

    assert decision.behavior == "deny"
    assert decision.rule == "robot_state"
