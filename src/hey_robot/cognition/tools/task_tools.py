"""Agent-facing task lifecycle tools."""

from __future__ import annotations

from typing import Any, ClassVar

from hey_robot.cognition.tools.models import (
    CompleteTaskProposal,
    ControlTaskProposal,
    ToolSpec,
)


class CompleteTaskTool:
    name: ClassVar[str] = "complete_task"
    schema: ClassVar[dict[str, Any]] = {
        "type": "function",
        "function": {
            "name": name,
            "description": (
                "Finish the active task only when the complete user objective is "
                "supported by successful tool results, and provide a concise recap."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "recap": {"type": "string"},
                },
                "required": ["recap"],
                "additionalProperties": False,
            },
        },
    }

    spec: ClassVar[ToolSpec] = ToolSpec(
        name,
        schema["function"]["description"],
        schema["function"]["parameters"],
    )

    def prepare(self, arguments: dict[str, Any]) -> CompleteTaskProposal:
        recap = arguments.get("recap")
        if not isinstance(recap, str) or not recap.strip():
            raise ValueError("recap must be a non-empty string")
        return CompleteTaskProposal(recap.strip())


class ControlTaskTool:
    name: ClassVar[str] = "control_task"
    schema: ClassVar[dict[str, Any]] = {
        "type": "function",
        "function": {
            "name": name,
            "description": (
                "Stop the active task without claiming success: cancel it, block for "
                "human input, or request an emergency stop."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["cancel", "block", "emergency_stop"],
                    },
                    "reason": {
                        "type": "string",
                        "description": "面向用户的简短原因。",
                    },
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        },
    }

    spec: ClassVar[ToolSpec] = ToolSpec(
        name,
        schema["function"]["description"],
        schema["function"]["parameters"],
    )

    def prepare(self, arguments: dict[str, Any]) -> ControlTaskProposal:
        action = arguments.get("action")
        if action not in {"cancel", "block", "emergency_stop"}:
            raise ValueError("action is invalid")
        reason = arguments.get("reason", "")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("reason must be a string")
        return ControlTaskProposal(
            action, reason.strip() if isinstance(reason, str) else ""
        )
