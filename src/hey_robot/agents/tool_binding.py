from __future__ import annotations

from typing import TYPE_CHECKING

from hey_robot.agents.tools import (
    ToolContext,
    ToolLoader,
    ToolRegistry as NewToolRegistry,
)
from hey_robot.logging import HeyRobotLogger

if TYPE_CHECKING:
    from hey_robot.agents.core import RobotAgentCore

logger = HeyRobotLogger(name="tool_binding")


def bind_agent_tools(core: RobotAgentCore) -> ToolContext:
    """Discover agent tools and bind them to the runtime registry."""
    ctx = ToolContext(
        io=core.io,
        spec=core.spec,
        memory=core.memory,
        autonomy=core.autonomy,
        skill_catalog=None,
        skill_catalog_runtime=core.skill_catalog,
        runtime=core.runtime,
        skill_gateway=core.skill_gateway,
        runtime_state=core.runtime.state,
        pending_skills=core._pending_skills,
        task_runtime=getattr(core.io, "task_runtime", None),
        robot_type=core._configured_robot_type(),
        turn_context=None,
        _current_envelope=core._current_envelope,
        _configured_robot_type=core._configured_robot_type,
        _get_task=lambda: core.runtime.state.task,
        _get_robot_status=core.get_robot_status,
    )

    registry = NewToolRegistry()
    loader = ToolLoader()
    names = loader.load(ctx, registry)
    logger.info(f"从类加载了 {len(names)} 个工具：{names}")
    core.runtime.tools = registry
    core.runtime.tool_executor.registry = registry
    if core.runtime.tool_executor.capability_resolver is not None:
        core.runtime.tool_executor.capability_resolver.registry = registry
    return ctx
