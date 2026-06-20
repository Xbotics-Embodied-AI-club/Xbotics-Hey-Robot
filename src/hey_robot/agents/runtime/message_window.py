from __future__ import annotations

from dataclasses import dataclass

from hey_robot.providers import ReasoningMessage


@dataclass(frozen=True)
class MessageWindowPolicy:
    max_messages: int = 40
    max_tool_result_chars: int = 12_000


def apply_message_window(
    messages: list[ReasoningMessage], policy: MessageWindowPolicy
) -> list[ReasoningMessage]:
    """Apply deterministic message and tool-result limits."""
    capped: list[ReasoningMessage] = []
    for message in messages:
        if message.role == "tool" and isinstance(message.content, str):
            capped.append(
                ReasoningMessage(
                    role=message.role,
                    content=truncate_tool_result(
                        message.content, policy.max_tool_result_chars
                    ),
                    images=message.images,
                    tool_calls=message.tool_calls,
                    tool_call_id=message.tool_call_id,
                    tool_name=message.tool_name,
                )
            )
        else:
            capped.append(message)
    if policy.max_messages <= 0 or len(capped) <= policy.max_messages:
        return capped
    prefix = capped[:1] if capped and capped[0].role == "system" else []
    tail = capped[-max(1, policy.max_messages - len(prefix)) :]
    return [*prefix, *tail]


def truncate_tool_result(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n\n[tool result truncated: {omitted} chars omitted]"
