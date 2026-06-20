from __future__ import annotations

import json
from typing import Any

from hey_robot.agents.tools.base import Tool, tool_parameters
from hey_robot.agents.tools.context import ToolContext
from hey_robot.agents.tools.schema import StringSchema, tool_parameters_schema
from hey_robot.agents.tools.task_introspection import (
    current_episode_id,
    current_task_snapshot,
    last_execution_feedback_snapshot,
    recovery_options,
    recovery_snapshot,
)


@tool_parameters(
    tool_parameters_schema(
        detail_level=StringSchema(
            "brief or full. brief returns the decision summary; full includes raw task fields.",
            enum=["brief", "full"],
        )
    )
)
class GetTaskContextTool(Tool):
    name = "get_task_context"
    description = (
        "Read the current task, latest execution feedback, recovery state, "
        "and recommended next steps through one stable context boundary."
    )
    read_only = True
    safety_level = "observe"

    def __init__(self, ctx: ToolContext) -> None:
        self._ctx = ctx

    @classmethod
    def create(cls, ctx: ToolContext):
        return cls(ctx)

    async def execute(self, detail_level: str = "brief") -> str:
        task = current_task_snapshot(self._ctx)
        feedback = last_execution_feedback_snapshot(self._ctx)
        recovery = recovery_snapshot(self._ctx)
        options = recovery_options(self._ctx)
        payload: dict[str, Any] = {
            "episode_id": current_episode_id(self._ctx),
            "task": _compact_task(task),
            "latest_execution_feedback": feedback,
            "recovery": recovery,
            "recommended_next_steps": options,
            "loop_warning": _loop_warning(self._ctx),
            "recent_tools": _recent_tools(self._ctx),
        }
        if detail_level == "full":
            payload["raw_task"] = task
        return json.dumps(payload, ensure_ascii=False)


def _compact_task(task: dict[str, Any] | None) -> dict[str, Any] | None:
    if task is None:
        return None
    attempts = [item for item in task.get("attempts", []) if isinstance(item, dict)]
    latest_attempt = attempts[-1] if attempts else None
    return {
        "root_task": task.get("root_task"),
        "status": task.get("status"),
        "robot_id": task.get("robot_id"),
        "failure_reason": task.get("failure_reason"),
        "latest_attempt": latest_attempt,
    }


def _recent_tools(ctx: ToolContext) -> str | None:
    runtime_state = getattr(ctx, "runtime_state", None)
    if runtime_state is None:
        return None
    recent = runtime_state.recent_tool_context(limit=4)
    return recent.strip() if isinstance(recent, str) and recent.strip() else None


def _loop_warning(ctx: ToolContext) -> str | None:
    runtime_state = getattr(ctx, "runtime_state", None)
    if runtime_state is None:
        return None
    warning = runtime_state.loop_warning_context(limit=6)
    return warning.strip() if isinstance(warning, str) and warning.strip() else None
