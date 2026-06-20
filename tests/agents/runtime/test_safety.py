from __future__ import annotations

import asyncio

from hey_robot.agents.runtime.safety import RobotSafetyHook
from hey_robot.agents.runtime.tool_executor import ToolExecutor
from hey_robot.agents.tools.registry import ToolRegistry


def test_robot_safety_hook_blocks_request_capability_on_estop():
    registry = ToolRegistry()
    registry.register_simple(
        "request_capability",
        lambda capability, objective: f"issued {capability}: {objective}",
        safety_level="actuate",
    )
    executor = ToolExecutor(
        registry,
        hooks=[RobotSafetyHook(lambda: {"frame_id": 0, "emergency_stop": True})],
    )

    result = asyncio.run(
        executor.execute(
            "request_capability",
            {"capability": "open_drawer", "objective": "open drawer"},
        )
    )

    assert result.success is False
    assert "robot safety gate blocked" in result.result


def test_robot_safety_hook_allows_unmanaged_actuation_tool_on_estop():
    registry = ToolRegistry()
    registry.register_simple(
        "move", lambda objective: f"moved {objective}", safety_level="actuate"
    )
    executor = ToolExecutor(
        registry,
        hooks=[RobotSafetyHook(lambda: {"emergency_stop": True})],
    )

    result = asyncio.run(executor.execute("move", {"objective": "open drawer"}))

    assert result.success is True
    assert result.result == "moved open drawer"


def test_robot_safety_hook_blocks_compound_skill_objective():
    registry = ToolRegistry()
    registry.register_simple(
        "request_capability",
        lambda capability, objective: f"issued {capability}: {objective}",
        safety_level="actuate",
    )
    executor = ToolExecutor(
        registry,
        hooks=[RobotSafetyHook(lambda: {"frame_id": 0})],
    )

    result = asyncio.run(
        executor.execute(
            "request_capability",
            {
                "capability": "vla_manipulation",
                "objective": "pick up the cup then place it",
            },
        )
    )

    assert result.success is False
    assert "compound skill objective" in result.result


def test_robot_safety_hook_requires_status_snapshot_for_skill_submission():
    registry = ToolRegistry()
    registry.register_simple(
        "request_capability",
        lambda capability, objective: f"issued {capability}: {objective}",
        safety_level="actuate",
    )
    executor = ToolExecutor(
        registry,
        hooks=[RobotSafetyHook(lambda: None)],
    )

    result = asyncio.run(
        executor.execute(
            "request_capability",
            {"capability": "open_drawer", "objective": "open drawer"},
        )
    )

    assert result.success is False
    assert "missing robot status snapshot" in result.result


def test_robot_safety_hook_allows_skill_submission_with_valid_frame_snapshot():
    registry = ToolRegistry()
    registry.register_simple(
        "request_capability",
        lambda capability, objective: f"issued {capability}: {objective}",
        safety_level="actuate",
    )
    executor = ToolExecutor(
        registry,
        hooks=[RobotSafetyHook(lambda: {"frame_id": 42})],
    )

    result = asyncio.run(
        executor.execute(
            "request_capability",
            {"capability": "open_drawer", "objective": "open drawer"},
        )
    )

    assert result.success is True
    assert result.result == "issued open_drawer: open drawer"


def test_robot_safety_hook_blocks_consecutive_base_motion_without_perception():
    registry = ToolRegistry()
    registry.register_simple(
        "request_capability",
        lambda capability, objective: f"issued {capability}: {objective}",
        safety_level="actuate",
    )
    executor = ToolExecutor(
        registry,
        hooks=[
            RobotSafetyHook(
                lambda: {
                    "frame_id": 42,
                    "recent_tool_calls": [
                        {
                            "name": "request_capability",
                            "arguments": {
                                "capability": "move_base",
                                "objective": "move forward",
                            },
                            "success": True,
                        }
                    ],
                }
            )
        ],
    )

    result = asyncio.run(
        executor.execute(
            "request_capability",
            {"capability": "move_base", "objective": "move forward again"},
        )
    )

    assert result.success is False
    assert "consecutive move_base requires fresh perception evidence" in result.result


def test_robot_safety_hook_allows_base_motion_after_perception_evidence():
    registry = ToolRegistry()
    registry.register_simple(
        "request_capability",
        lambda capability, objective: f"issued {capability}: {objective}",
        safety_level="actuate",
    )
    executor = ToolExecutor(
        registry,
        hooks=[
            RobotSafetyHook(
                lambda: {
                    "frame_id": 42,
                    "recent_tool_calls": [
                        {
                            "name": "request_capability",
                            "arguments": {
                                "capability": "move_base",
                                "objective": "move forward",
                            },
                            "success": True,
                        },
                        {
                            "name": "request_perception",
                            "arguments": {
                                "question": "what is ahead",
                                "freshness": "fresh",
                            },
                            "success": True,
                        },
                    ],
                }
            )
        ],
    )

    result = asyncio.run(
        executor.execute(
            "request_capability",
            {"capability": "move_base", "objective": "move forward again"},
        )
    )

    assert result.success is True
    assert result.result == "issued move_base: move forward again"


def test_robot_safety_hook_does_not_block_after_failed_prior_base_motion():
    registry = ToolRegistry()
    registry.register_simple(
        "request_capability",
        lambda capability, objective: f"issued {capability}: {objective}",
        safety_level="actuate",
    )
    executor = ToolExecutor(
        registry,
        hooks=[
            RobotSafetyHook(
                lambda: {
                    "frame_id": 42,
                    "recent_tool_calls": [
                        {
                            "name": "request_capability",
                            "arguments": {
                                "capability": "move_base",
                                "objective": "move forward",
                            },
                            "success": False,
                        }
                    ],
                }
            )
        ],
    )

    result = asyncio.run(
        executor.execute(
            "request_capability",
            {"capability": "move_base", "objective": "retry move forward"},
        )
    )

    assert result.success is True
    assert result.result == "issued move_base: retry move forward"


def test_robot_safety_hook_does_not_block_non_motion_capability_with_prior_base_motion():
    registry = ToolRegistry()
    registry.register_simple(
        "request_capability",
        lambda capability, objective: f"issued {capability}: {objective}",
        safety_level="actuate",
    )
    executor = ToolExecutor(
        registry,
        hooks=[
            RobotSafetyHook(
                lambda: {
                    "frame_id": 42,
                    "recent_tool_calls": [
                        {
                            "name": "request_capability",
                            "arguments": {
                                "capability": "move_base",
                                "objective": "move forward",
                            },
                            "success": True,
                        }
                    ],
                }
            )
        ],
    )

    result = asyncio.run(
        executor.execute(
            "request_capability",
            {"capability": "inspect_scene", "objective": "inspect the front area"},
        )
    )

    assert result.success is True
    assert result.result == "issued inspect_scene: inspect the front area"
