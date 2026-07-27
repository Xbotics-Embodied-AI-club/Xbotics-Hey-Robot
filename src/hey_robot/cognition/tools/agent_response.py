"""The one model-visible response boundary for interactive Agent turns."""

from __future__ import annotations

from typing import Any, Literal, cast

from hey_robot.cognition.tools.models import AgentResponseCall, ToolSpec
from hey_robot.tool_schema import validate_arguments


class AgentResponseTool:
    """Prepare a user response and its task state without executing IO."""

    name = "respond"
    spec = ToolSpec(
        name,
        (
            "Return every user-facing response through this function. Set "
            "task_state='none' for ordinary conversation that does not change a "
            "sustained robot task and only when no task is active; 'wait' only when "
            "the active objective itself is unfinished and requires confirmation or "
            "more work; 'complete' when trusted physical results support the whole "
            "objective; and 'cancel' when the user withdraws it. An optional offer "
            "of additional help after fulfilling the objective does not make the "
            "task unfinished: use 'complete'. The message is shown to the user."
        ),
        {
            "type": "object",
            "properties": {
                "task_state": {
                    "type": "string",
                    "enum": ["none", "wait", "complete", "cancel"],
                },
                "message": {"type": "string", "minLength": 1},
            },
            "required": ["task_state", "message"],
            "additionalProperties": False,
        },
    )

    @property
    def schema(self) -> dict[str, Any]:
        return self.spec.definition

    def prepare(self, arguments: dict[str, Any]) -> AgentResponseCall:
        normalized = validate_arguments(self.spec.parameters, arguments)
        return AgentResponseCall(
            task_state=cast(
                Literal["none", "wait", "complete", "cancel"],
                normalized["task_state"],
            ),
            message=str(normalized["message"]).strip(),
        )
