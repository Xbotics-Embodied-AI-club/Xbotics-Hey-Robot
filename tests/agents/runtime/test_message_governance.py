from __future__ import annotations

import pytest

from hey_robot.agents.runtime.message_protocol import (
    validate_provider_messages,
    validate_provider_response,
)
from hey_robot.agents.runtime.message_window import (
    MessageWindowPolicy,
    apply_message_window,
    truncate_tool_result,
)
from hey_robot.agents.runtime.response_policy import (
    decide_response,
    looks_like_unexecuted_tool_protocol,
)
from hey_robot.providers import ReasoningMessage, ReasoningResponse, ReasoningToolCall


def test_message_protocol_requires_tool_call_result_pairs() -> None:
    messages = [
        ReasoningMessage(role="user", content="move"),
        ReasoningMessage(
            role="assistant",
            content="",
            tool_calls=[
                ReasoningToolCall(id="call_1", name="request_capability", arguments={})
            ],
        ),
    ]

    with pytest.raises(ValueError, match="missing tool result"):
        validate_provider_messages(messages)


def test_message_protocol_rejects_orphan_tool_results() -> None:
    messages = [
        ReasoningMessage(role="user", content="move"),
        ReasoningMessage(role="tool", content="done", tool_call_id="missing"),
    ]

    with pytest.raises(ValueError, match="orphan tool result"):
        validate_provider_messages(messages)


def test_message_protocol_accepts_valid_tool_pair_and_response_shape() -> None:
    validate_provider_messages(
        [
            ReasoningMessage(role="user", content="move"),
            ReasoningMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ReasoningToolCall(
                        id="call_1",
                        name="request_capability",
                        arguments={"capability": "stop_motion"},
                    )
                ],
            ),
            ReasoningMessage(role="tool", content="done", tool_call_id="call_1"),
        ]
    )
    validate_provider_response(
        ReasoningResponse(
            tool_calls=[
                ReasoningToolCall(
                    id="call_2",
                    name="request_capability",
                    arguments={"capability": "inspect_scene"},
                )
            ],
            finish_reason="tool_calls",
        )
    )


def test_message_window_keeps_system_prompt_and_truncates_tool_results() -> None:
    messages = [
        ReasoningMessage(role="system", content="system"),
        ReasoningMessage(role="user", content="old"),
        ReasoningMessage(role="assistant", content="middle"),
        ReasoningMessage(role="tool", content="abcdef", tool_call_id="call_1"),
        ReasoningMessage(role="user", content="new"),
    ]

    windowed = apply_message_window(
        messages, MessageWindowPolicy(max_messages=3, max_tool_result_chars=3)
    )

    assert [message.content for message in windowed] == [
        "system",
        "abc\n\n[tool result truncated: 3 chars omitted]",
        "new",
    ]
    assert windowed[1].tool_call_id == "call_1"


def test_truncate_tool_result_respects_non_positive_limit() -> None:
    assert truncate_tool_result("abcdef", 0) == "abcdef"


def test_response_policy_classifies_error_tool_text_and_empty_responses() -> None:
    assert (
        decide_response(
            ReasoningResponse(content="rate limited", finish_reason="error")
        ).action
        == "provider_error"
    )
    assert (
        decide_response(
            ReasoningResponse(
                tool_calls=[
                    ReasoningToolCall(
                        id="call_1", name="request_capability", arguments={}
                    )
                ],
                finish_reason="tool_calls",
            )
        ).action
        == "execute_tools"
    )
    assert decide_response(ReasoningResponse(content="  ")).action == "empty"
    assert looks_like_unexecuted_tool_protocol("```tool_call request_capability({})```")
