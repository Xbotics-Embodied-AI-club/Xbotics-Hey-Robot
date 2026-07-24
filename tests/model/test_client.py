from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from hey_robot.model import ModelClient, ModelImage, ModelMessage, ModelToolCall
from hey_robot.model.client import _message_payload, _parse_response, _tool_payload


class _CaptureCreate:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.body: dict[str, Any] | None = None

    async def create(self, **body: Any) -> Any:
        self.body = body
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class _AsyncChunks:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = chunks

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk


def _client_with_response(result: Any) -> tuple[ModelClient, _CaptureCreate]:
    create = _CaptureCreate(result)
    client = ModelClient(model="test-model", api_key="test-key")
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create.create))
    )
    return client, create


async def test_chat_uses_one_async_chat_completions_endpoint() -> None:
    client, create = _client_with_response(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "move",
                                    "arguments": '{"distance": 1}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
    )

    response = await client.chat(
        messages=[ModelMessage("user", "move")],
        tools=[{"type": "function", "function": {"name": "move"}}],
    )

    assert create.body is not None
    assert create.body["model"] == "test-model"
    assert create.body["tool_choice"] == "auto"
    assert response.tool_calls == [ModelToolCall("call-1", "move", {"distance": 1})]
    assert response.usage["total_tokens"] == 15


async def test_sdk_error_becomes_model_error_response() -> None:
    client, _ = _client_with_response(ConnectionError("offline"))

    response = await client.chat(messages=[ModelMessage("user", "hello")])

    assert response.finish_reason == "error"
    assert response.error_kind == "request"
    assert "offline" in (response.content or "")


async def test_stream_emits_text_deltas_and_returns_complete_text() -> None:
    client, create = _client_with_response(
        _AsyncChunks(
            [
                {"choices": [{"delta": {"content": "你"}}]},
                {"choices": [{"delta": {"content": "好"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ]
        )
    )
    deltas: list[str] = []

    async def collect(delta: str) -> None:
        deltas.append(delta)

    response = await client.chat(
        messages=[ModelMessage("user", "hello")],
        on_text_delta=collect,
    )

    assert create.body is not None
    assert create.body["stream"] is True
    assert deltas == ["你", "好"]
    assert response.content == "你好"
    assert response.finish_reason == "stop"


async def test_stream_buffers_tool_arguments_until_complete() -> None:
    client, _ = _client_with_response(
        _AsyncChunks(
            [
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-1",
                                        "function": {
                                            "name": "move",
                                            "arguments": '{"distance":',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": " 1}"},
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            ]
        )
    )
    deltas: list[str] = []

    async def collect(delta: str) -> None:
        deltas.append(delta)

    response = await client.chat(
        messages=[ModelMessage("user", "move")],
        tools=[{"type": "function", "function": {"name": "move"}}],
        on_text_delta=collect,
    )

    assert deltas == []
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls == [ModelToolCall("call-1", "move", {"distance": 1})]


def test_message_payload_preserves_tool_history() -> None:
    payload = _message_payload(
        ModelMessage(
            "assistant",
            "",
            tool_calls=[ModelToolCall("call-1", "inspect", {"area": "front"})],
        )
    )

    assert payload["tool_calls"][0]["function"]["name"] == "inspect"
    assert payload["tool_calls"][0]["function"]["arguments"] == ('{"area": "front"}')


def test_image_message_is_encoded_as_data_url() -> None:
    payload = _message_payload(
        ModelMessage(
            "user",
            "what is here?",
            images=[ModelImage(np.zeros((8, 8, 3), dtype=np.uint8))],
        )
    )

    image_url = payload["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/jpeg;base64,")


def test_tool_payload_accepts_project_schema_shape() -> None:
    payload = _tool_payload(
        {
            "name": "move",
            "description": "Move the robot",
            "inputSchema": {
                "type": "object",
                "properties": {"distance": {"type": "number"}},
            },
        }
    )

    assert payload["type"] == "function"
    assert payload["function"]["name"] == "move"
    assert "distance" in payload["function"]["parameters"]["properties"]


def test_invalid_tool_arguments_are_preserved_for_local_validation() -> None:
    response = _parse_response(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "bad",
                                "function": {
                                    "name": "move",
                                    "arguments": "not-json",
                                },
                            }
                        ]
                    }
                }
            ]
        }
    )

    assert response.tool_calls[0].arguments == {"raw": "not-json"}
