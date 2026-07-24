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
            "description": "引用当前任务证据并提议结束 active task。",
            "parameters": {
                "type": "object",
                "properties": {
                    "recap": {"type": "string"},
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["recap", "evidence_ids"],
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
        evidence_ids = arguments.get("evidence_ids")
        if not isinstance(recap, str) or not recap.strip():
            raise ValueError("recap must be a non-empty string")
        if not isinstance(evidence_ids, list) or not evidence_ids:
            raise ValueError("evidence_ids must be a non-empty array")
        normalized = tuple(
            item.strip()
            for item in evidence_ids
            if isinstance(item, str) and item.strip()
        )
        if len(normalized) != len(evidence_ids):
            raise ValueError("evidence_ids must contain only non-empty strings")
        return CompleteTaskProposal(recap.strip(), normalized)

    def proposal(self, arguments: dict[str, Any]) -> CompleteTaskProposal:
        return self.prepare(arguments)


class ControlTaskTool:
    name: ClassVar[str] = "control_task"
    schema: ClassVar[dict[str, Any]] = {
        "type": "function",
        "function": {
            "name": name,
            "description": "取消、阻塞确认或紧急停止当前持续任务。",
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

    def proposal(self, arguments: dict[str, Any]) -> ControlTaskProposal:
        return self.prepare(arguments)
