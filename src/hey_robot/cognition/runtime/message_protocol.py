from __future__ import annotations

from collections.abc import Sequence

from hey_robot.providers import ReasoningMessage, ReasoningResponse


def validate_provider_messages(messages: Sequence[ReasoningMessage]) -> None:
    """Validate the canonical message sequence used inside AgentRuntime."""
    pending_tool_call_ids: set[str] = set()
    for message in messages:
        role = message.role
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"invalid message role: {role!r}")
        if role == "assistant":
            for call in message.tool_calls or ():
                if not getattr(call, "id", ""):
                    raise ValueError("assistant tool call is missing id")
                pending_tool_call_ids.add(call.id)
        if role == "tool":
            tool_call_id = message.tool_call_id
            if not tool_call_id:
                raise ValueError("tool result is missing tool_call_id")
            if tool_call_id not in pending_tool_call_ids:
                raise ValueError(f"orphan tool result: {tool_call_id}")
            pending_tool_call_ids.discard(tool_call_id)
    if pending_tool_call_ids:
        missing = ", ".join(sorted(pending_tool_call_ids))
        raise ValueError(f"missing tool result for tool_call_id: {missing}")


def validate_provider_response(response: ReasoningResponse) -> None:
    """Validate provider response shape before it reaches runtime policy."""
    if not response.should_execute_tools:
        return
    for tool_call in response.tool_calls:
        if not tool_call.name:
            raise ValueError("tool call is missing name")
        if not tool_call.id:
            raise ValueError(f"tool call {tool_call.name!r} is missing id")
        if not isinstance(tool_call.arguments, dict):
            raise ValueError(
                f"tool call {tool_call.name!r} arguments must be an object"
            )
