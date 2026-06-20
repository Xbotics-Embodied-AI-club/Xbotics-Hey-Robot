from __future__ import annotations

import asyncio
import json
from pathlib import Path

from hey_robot.agents.runtime.audit import ToolAuditLogger
from hey_robot.agents.runtime.permissions import PermissionManager
from hey_robot.agents.runtime.tool_executor import ToolExecutor
from hey_robot.agents.tools.registry import ToolRegistry
from hey_robot.capability.catalog import CapabilityPolicy, CapabilityResolver


def test_tool_executor_validates_input_and_audits(tmp_path: Path):
    registry = ToolRegistry()
    registry.register_simple(
        "echo",
        lambda text: text,
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        read_only=True,
    )
    audit = ToolAuditLogger(tmp_path, agent_run_id="s1")
    executor = ToolExecutor(registry, audit_logger=audit)

    ok = asyncio.run(executor.execute("echo", {"text": "hello"}, task="demo"))
    bad = asyncio.run(executor.execute("echo", {}, task="demo"))

    assert ok.success is True
    assert ok.result == "hello"
    assert bad.success is False
    assert "missing required argument" in bad.result

    records = [
        json.loads(line)
        for line in (tmp_path / "agent_runs" / "s1" / "tool_calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["status"] for record in records] == ["success", "failed"]


def test_tool_executor_allows_explicit_no_timeout():
    registry = ToolRegistry()

    async def slow_tool() -> str:
        await asyncio.sleep(0.02)
        return "done"

    registry.register_simple("slow_tool", slow_tool, timeout_sec=0.0)
    executor = ToolExecutor(registry, default_timeout_sec=0.001)

    result = asyncio.run(executor.execute("slow_tool", {}))

    assert result.success is True
    assert result.result == "done"


def test_tool_executor_permission_dry_run_blocks_actuation():
    registry = ToolRegistry()
    registry.register_simple(
        "move", lambda objective: f"moved {objective}", safety_level="actuate"
    )
    executor = ToolExecutor(
        registry,
        permission_manager=PermissionManager("dry_run"),
    )

    result = asyncio.run(executor.execute("move", {"objective": "open drawer"}))

    assert result.success is False
    assert result.permission_behavior == "deny"
    assert "dry_run blocks" in result.result


def test_tool_executor_capability_policy_blocks_before_permission(tmp_path: Path):
    registry = ToolRegistry()
    registry.register_simple(
        "move", lambda objective: f"moved {objective}", safety_level="actuate"
    )
    audit = ToolAuditLogger(tmp_path, agent_run_id="s1")
    executor = ToolExecutor(
        registry,
        audit_logger=audit,
        capability_resolver=CapabilityResolver(
            registry,
            policy=CapabilityPolicy(deny_safety_levels=("actuate",)),
        ),
    )

    result = asyncio.run(executor.execute("move", {"objective": "open drawer"}))

    assert result.success is False
    assert result.capability_behavior == "deny"
    assert result.capability_rule == "deny_safety_levels"
    records = [
        json.loads(line)
        for line in (tmp_path / "agent_runs" / "s1" / "tool_calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert records[0]["capability_behavior"] == "deny"
    assert records[0]["capability_rule"] == "deny_safety_levels"
