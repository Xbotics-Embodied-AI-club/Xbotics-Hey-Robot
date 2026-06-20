from __future__ import annotations

from hey_robot.capability.catalog import CapabilityPolicy, CapabilityPolicySet


def test_capability_policy_from_dict_roundtrip_and_mode_override() -> None:
    policy_set = CapabilityPolicySet.from_dict(
        {
            "allow_tools": ["get_robot_status"],
            "require_approval_for": ["actuate"],
            "modes": {
                "manual": {
                    "mode": "manual",
                    "allow_tools": ["move_base"],
                }
            },
        }
    )

    default_policy = policy_set.for_mode(None)
    manual_policy = policy_set.for_mode("manual")

    assert default_policy.to_dict()["allow_tools"] == ["get_robot_status"]
    assert manual_policy.mode == "manual"
    assert manual_policy.allow_tools == ("move_base",)


def test_capability_policy_decision_paths_cover_ask_deny_and_allow() -> None:
    policy = CapabilityPolicy.from_dict(
        {
            "allow_tools": [
                "move_base",
                "move_arm",
                "get_robot_status",
                "get_task_context",
            ],
            "deny_tools": ["move_arm"],
            "deny_sources": ["remote"],
            "deny_safety_levels": ["unsafe_actuate"],
            "require_approval_for": ["actuate"],
            "deny_when_robot_states": ["failed"],
            "safe_on_blocked_robot": ["get_task_context", "get_robot_status"],
        }
    )

    assert (
        policy.decide(
            tool_name="unknown_tool",
            source="local",
            safety_level="observe",
            read_only=True,
        ).rule
        == "allow_tools"
    )
    assert (
        policy.decide(
            tool_name="move_arm",
            source="local",
            safety_level="actuate",
            read_only=False,
        ).rule
        == "deny_tools"
    )
    assert (
        policy.decide(
            tool_name="move_base",
            source="remote",
            safety_level="actuate",
            read_only=False,
        ).rule
        == "deny_sources"
    )
    assert (
        policy.decide(
            tool_name="move_base",
            source="local",
            safety_level="unsafe_actuate",
            read_only=False,
        ).rule
        == "deny_safety_levels"
    )
    assert (
        policy.decide(
            tool_name="move_base",
            source="local",
            safety_level="actuate",
            read_only=False,
            robot_state="failed",
        ).rule
        == "robot_state"
    )
    assert (
        policy.decide(
            tool_name="move_base",
            source="local",
            safety_level="actuate",
            read_only=False,
        ).behavior
        == "ask"
    )
    assert (
        policy.decide(
            tool_name="get_task_context",
            source="local",
            safety_level="observe",
            read_only=False,
            robot_state="failed",
        ).behavior
        == "allow"
    )
