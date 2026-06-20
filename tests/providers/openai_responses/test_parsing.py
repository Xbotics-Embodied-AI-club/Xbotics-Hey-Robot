from __future__ import annotations

import json
import logging

from hey_robot.providers.openai_responses.converters import (
    convert_messages,
    convert_tools,
    convert_user_message,
    split_tool_call_id,
)
from hey_robot.providers.openai_responses.parsing import (
    FINISH_REASON_MAP,
    map_finish_reason,
    parse_response_output,
)


class TestMapFinishReason:
    def test_known_statuses(self) -> None:
        assert map_finish_reason("completed") == "stop"
        assert map_finish_reason("incomplete") == "length"
        assert map_finish_reason("failed") == "error"
        assert map_finish_reason("cancelled") == "error"

    def test_unknown_status_defaults_to_stop(self) -> None:
        assert map_finish_reason("unknown_status") == "stop"

    def test_none_defaults_to_stop(self) -> None:
        assert map_finish_reason(None) == "stop"

    def test_empty_string_defaults_to_stop(self) -> None:
        assert map_finish_reason("") == "stop"

    def test_map_coverage(self) -> None:
        assert "completed" in FINISH_REASON_MAP
        assert "incomplete" in FINISH_REASON_MAP
        assert "failed" in FINISH_REASON_MAP
        assert "cancelled" in FINISH_REASON_MAP


class TestParseResponseOutput:
    def test_parses_simple_text_response(self) -> None:
        response = {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hello world"}],
                }
            ],
        }
        result = parse_response_output(response)
        assert result.content == "Hello world"
        assert result.tool_calls == []
        assert result.finish_reason == "stop"

    def test_parses_response_with_reasoning(self) -> None:
        response = {
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Let me think..."}],
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Answer"}],
                },
            ],
        }
        result = parse_response_output(response)
        assert result.content == "Answer"
        assert result.reasoning_content == "Let me think..."

    def test_parses_function_call(self) -> None:
        response = {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "id": "fc_1",
                    "name": "move_arm",
                    "arguments": json.dumps({"x": 1, "y": 2}),
                }
            ],
        }
        result = parse_response_output(response)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "move_arm"
        assert result.tool_calls[0].arguments == {"x": 1, "y": 2}

    def test_parses_function_call_with_dict_args(self) -> None:
        response = {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "id": "fc_1",
                    "name": "grasp",
                    "arguments": {"object": "cup"},
                }
            ],
        }
        result = parse_response_output(response)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {"object": "cup"}

    def test_parses_function_call_with_invalid_json(self, caplog) -> None:
        response = {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "id": "fc_1",
                    "name": "bad",
                    "arguments": "not valid json {{{",
                }
            ],
        }
        hey_robot_logger = logging.getLogger("hey_robot")
        hey_robot_logger.addHandler(caplog.handler)
        try:
            with caplog.at_level("WARNING", logger="hey_robot"):
                result = parse_response_output(response)
        finally:
            hey_robot_logger.removeHandler(caplog.handler)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        # Should fall back to raw dict
        assert isinstance(result.tool_calls[0].arguments, dict)
        assert "tool call 参数解析失败 'bad'" in caplog.text

    def test_parses_empty_output(self) -> None:
        response = {"status": "completed", "output": []}
        result = parse_response_output(response)
        assert result.content is None
        assert result.tool_calls == []

    def test_parses_missing_output(self) -> None:
        response = {"status": "completed"}
        result = parse_response_output(response)
        assert result.content is None
        assert result.tool_calls == []

    def test_parses_usage(self) -> None:
        response = {
            "status": "completed",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "Hi"}]}
            ],
            "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        }
        result = parse_response_output(response)
        assert result.usage == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

    def test_parses_multiple_messages_concatenated(self) -> None:
        response = {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "First"}],
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Second"}],
                },
            ],
        }
        result = parse_response_output(response)
        assert result.content == "FirstSecond"


class TestSplitToolCallId:
    def test_splits_compound_id(self) -> None:
        call_id, item_id = split_tool_call_id("call_1|fc_1")
        assert call_id == "call_1"
        assert item_id == "fc_1"

    def test_simple_id_no_split(self) -> None:
        call_id, item_id = split_tool_call_id("simple_id")
        assert call_id == "simple_id"
        assert item_id is None

    def test_empty_string_returns_default(self) -> None:
        call_id, item_id = split_tool_call_id("")
        assert call_id == "call_0"
        assert item_id is None

    def test_none_returns_default(self) -> None:
        call_id, item_id = split_tool_call_id(None)
        assert call_id == "call_0"
        assert item_id is None

    def test_non_string_returns_default(self) -> None:
        call_id, item_id = split_tool_call_id(42)
        assert call_id == "call_0"
        assert item_id is None

    def test_pipe_with_empty_second_part(self) -> None:
        call_id, item_id = split_tool_call_id("call_1|")
        assert call_id == "call_1"
        assert item_id is None


class TestConvertUserMessage:
    def test_string_content(self) -> None:
        result = convert_user_message("hello")
        assert result["role"] == "user"
        assert result["content"][0]["type"] == "input_text"
        assert result["content"][0]["text"] == "hello"

    def test_list_with_text_block(self) -> None:
        result = convert_user_message([{"type": "text", "text": "hello"}])
        assert result["role"] == "user"
        assert result["content"][0]["type"] == "input_text"
        assert result["content"][0]["text"] == "hello"

    def test_list_with_image_block(self) -> None:
        result = convert_user_message(
            [{"type": "image_url", "image_url": {"url": "http://example.com/img.jpg"}}]
        )
        assert result["role"] == "user"
        assert result["content"][0]["type"] == "input_image"
        assert result["content"][0]["image_url"] == "http://example.com/img.jpg"

    def test_list_with_mixed_blocks(self) -> None:
        result = convert_user_message(
            [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "http://img.jpg"}},
            ]
        )
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "input_text"
        assert result["content"][1]["type"] == "input_image"

    def test_image_without_url_is_skipped(self) -> None:
        result = convert_user_message(
            [
                {"type": "image_url", "image_url": {}},
            ]
        )
        # No valid items, falls through to empty text
        assert result["content"][0]["type"] == "input_text"
        assert result["content"][0]["text"] == ""

    def test_list_with_invalid_item_is_skipped(self) -> None:
        result = convert_user_message(["not a dict"])
        # Falls through to empty message
        assert result["content"][0]["type"] == "input_text"
        assert result["content"][0]["text"] == ""

    def test_non_string_non_list_returns_empty(self) -> None:
        result = convert_user_message(42)
        assert result["role"] == "user"
        assert result["content"][0]["type"] == "input_text"
        assert result["content"][0]["text"] == ""


class TestConvertMessages:
    def test_system_prompt_extraction(self) -> None:
        system_prompt, items = convert_messages(
            [
                {"role": "system", "content": "You are a robot."},
                {"role": "user", "content": "hello"},
            ]
        )
        assert system_prompt == "You are a robot."
        assert len(items) == 1
        assert items[0]["role"] == "user"

    def test_user_message_conversion(self) -> None:
        _, items = convert_messages([{"role": "user", "content": "hello"}])
        assert len(items) == 1
        assert items[0]["content"][0]["type"] == "input_text"
        assert items[0]["content"][0]["text"] == "hello"

    def test_tool_message_conversion(self) -> None:
        _, items = convert_messages(
            [
                {
                    "role": "tool",
                    "tool_call_id": "call_1|fc_1",
                    "content": '{"result": "ok"}',
                }
            ]
        )
        assert len(items) == 1
        assert items[0]["type"] == "function_call_output"
        assert items[0]["call_id"] == "call_1"

    def test_assistant_with_tool_calls(self) -> None:
        _, items = convert_messages(
            [
                {
                    "role": "assistant",
                    "content": "Let me check",
                    "tool_calls": [
                        {
                            "id": "call_1|fc_1",
                            "function": {
                                "name": "grasp",
                                "arguments": '{"object": "cup"}',
                            },
                        }
                    ],
                }
            ]
        )
        assert len(items) >= 2
        assert items[0]["role"] == "assistant"
        assert items[1]["type"] == "function_call"
        assert items[1]["name"] == "grasp"

    def test_empty_messages(self) -> None:
        system_prompt, items = convert_messages([])
        assert system_prompt == ""
        assert items == []

    def test_assistant_without_content_but_with_tool_calls(self) -> None:
        """Assistant messages with only tool calls (no text) should still produce function_call items."""
        _, items = convert_messages(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1|fc_1",
                            "function": {"name": "move", "arguments": '{"x": 1}'},
                        }
                    ],
                }
            ]
        )
        function_calls = [item for item in items if item.get("type") == "function_call"]
        assert len(function_calls) == 1
        assert function_calls[0]["name"] == "move"


class TestConvertTools:
    def test_converts_function_tool(self) -> None:
        result = convert_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "move_arm",
                        "description": "Move the robot arm",
                        "parameters": {
                            "type": "object",
                            "properties": {"x": {"type": "number"}},
                        },
                    },
                }
            ]
        )
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["name"] == "move_arm"

    def test_skips_tool_without_name(self) -> None:
        result = convert_tools(
            [{"type": "function", "function": {"name": "", "description": "bad"}}]
        )
        assert result == []

    def test_handles_non_function_type_as_fallback(self) -> None:
        result = convert_tools(
            [{"name": "fallback_tool", "description": "test", "parameters": {}}]
        )
        assert len(result) == 1
        assert result[0]["name"] == "fallback_tool"

    def test_empty_tools(self) -> None:
        assert convert_tools([]) == []

    def test_parameters_defaults_to_empty_dict(self) -> None:
        result = convert_tools([{"type": "function", "function": {"name": "test"}}])
        assert result[0]["parameters"] == {}
