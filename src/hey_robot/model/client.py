from __future__ import annotations

import base64
import io
import json
import logging
from typing import Any

import numpy as np
from PIL import Image

from hey_robot.model.types import (
    ModelImage,
    ModelMessage,
    ModelResponse,
    ModelToolCall,
    TextDeltaCallback,
)

logger = logging.getLogger("hey_robot.model")


class ModelClient:
    """One async OpenAI-compatible Chat Completions client."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        reasoning_effort: str | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("model client requires api_key")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.extra_headers = extra_headers
        self.extra_body = extra_body
        self._client: Any | None = None

    def _sdk_client(self) -> Any:
        if self._client is not None:
            return self._client

        from openai import AsyncOpenAI

        options: dict[str, Any] = {"api_key": self.api_key, "timeout": 60.0}
        if self.base_url:
            options["base_url"] = self.base_url
        if self.extra_headers:
            options["default_headers"] = self.extra_headers
        self._client = AsyncOpenAI(**options)
        return self._client

    async def chat(
        self,
        *,
        messages: list[ModelMessage],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_text_delta: TextDeltaCallback | None = None,
    ) -> ModelResponse:
        effort = self.reasoning_effort if reasoning_effort is None else reasoning_effort
        body: dict[str, Any] = {
            "model": model or self.model,
            "messages": [_message_payload(message) for message in messages],
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if not _reasoning_enabled(effort):
            body["temperature"] = (
                self.temperature if temperature is None else temperature
            )
        if _reasoning_enabled(effort):
            body["reasoning_effort"] = effort
        if tools:
            body["tools"] = [_tool_payload(tool) for tool in tools]
            body["tool_choice"] = tool_choice or "auto"
        if self.extra_body:
            body["extra_body"] = self.extra_body

        logger.debug(
            "model request: model=%s base_url=%s messages=%d tools=%d",
            body["model"],
            self.base_url,
            len(messages),
            len(tools or []),
        )
        try:
            if on_text_delta is not None:
                return await self._stream_chat(body, on_text_delta)
            response = await self._sdk_client().chat.completions.create(**body)
        except Exception as exc:
            return ModelResponse(
                content=f"Model request failed: {type(exc).__name__}: {exc}",
                finish_reason="error",
                error_kind="request",
            )
        return _parse_response(response)

    async def _stream_chat(
        self, body: dict[str, Any], on_text_delta: TextDeltaCallback
    ) -> ModelResponse:
        stream = await self._sdk_client().chat.completions.create(
            **body,
            stream=True,
        )
        content: list[str] = []
        tool_calls: dict[int, dict[str, str]] = {}
        finish_reason = "stop"
        usage: dict[str, int] = {}
        async for chunk in stream:
            payload = _payload(chunk)
            raw_usage = payload.get("usage") or {}
            if raw_usage:
                usage = {
                    "prompt_tokens": int(raw_usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(raw_usage.get("completion_tokens") or 0),
                    "total_tokens": int(raw_usage.get("total_tokens") or 0),
                }
            for choice in payload.get("choices") or []:
                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])
                delta = choice.get("delta") or {}
                text = _text_content(delta.get("content"))
                if text:
                    content.append(text)
                    await on_text_delta(text)
                for raw_call in delta.get("tool_calls") or []:
                    index = int(raw_call.get("index") or 0)
                    call = tool_calls.setdefault(
                        index, {"id": "", "name": "", "arguments": ""}
                    )
                    call["id"] += str(raw_call.get("id") or "")
                    function = raw_call.get("function") or {}
                    call["name"] += str(function.get("name") or "")
                    call["arguments"] += str(function.get("arguments") or "")
        parsed_calls = [
            ModelToolCall(
                id=call["id"],
                name=call["name"],
                arguments=_parse_arguments(call["arguments"]),
            )
            for _, call in sorted(tool_calls.items())
        ]
        return ModelResponse(
            content="".join(content) or None,
            tool_calls=parsed_calls,
            finish_reason=finish_reason,
            usage=usage,
        )


def _message_payload(message: ModelMessage) -> dict[str, Any]:
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id or "",
            "content": message.content,
        }
    if message.role == "assistant" and message.tool_calls:
        return {
            "role": "assistant",
            "content": message.content or None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ],
        }
    if not message.images:
        return {"role": message.role, "content": message.content}
    content: list[dict[str, Any]] = [{"type": "text", "text": message.content}]
    content.extend(_image_payload(image) for image in message.images)
    return {"role": message.role, "content": content}


def _tool_payload(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") == "function":
        return tool
    schema = (
        tool.get("inputSchema")
        or tool.get("input_schema")
        or {"type": "object", "properties": {}}
    )
    return {
        "type": "function",
        "function": {
            "name": str(tool.get("name") or ""),
            "description": str(tool.get("description") or ""),
            "parameters": schema,
        },
    }


def _image_payload(image: ModelImage) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{image.media_type};base64,{_encode_image(image)}",
            "detail": image.detail,
        },
    }


def _encode_image(image: ModelImage) -> str:
    data = image.data
    if data.ndim == 4:
        data = data[0]
    if data.dtype != np.uint8:
        data = np.clip(data, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(data)
    if max(pil_image.size) > 2048:
        pil_image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _parse_response(response: Any) -> ModelResponse:
    payload = _payload(response)
    if not payload:
        return ModelResponse(
            content=f"Unexpected model response: {type(response).__name__}",
            finish_reason="error",
            error_kind="protocol",
        )
    choices = payload.get("choices") or []
    if not choices:
        return ModelResponse(
            content="Model returned no completion choices",
            finish_reason="error",
            error_kind="protocol",
        )
    choice = choices[0]
    message = choice.get("message") or {}
    tool_calls = []
    for raw_call in message.get("tool_calls") or []:
        function = raw_call.get("function") or {}
        tool_calls.append(
            ModelToolCall(
                id=str(raw_call.get("id") or ""),
                name=str(function.get("name") or ""),
                arguments=_parse_arguments(function.get("arguments")),
            )
        )
    usage = payload.get("usage") or {}
    return ModelResponse(
        content=_text_content(message.get("content")),
        tool_calls=tool_calls,
        finish_reason=str(choice.get("finish_reason") or "stop"),
        usage={
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        }
        if usage
        else {},
    )


def _payload(value: Any) -> dict[str, Any]:
    dump = getattr(value, "model_dump", None)
    payload = dump() if callable(dump) else value
    return payload if isinstance(payload, dict) else {}


def _parse_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {"raw": value}
    return parsed if isinstance(parsed, dict) else {"raw": value}


def _text_content(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [
            str(item.get("text") or "")
            for item in value
            if isinstance(item, dict) and item.get("text")
        ]
        return "".join(parts) or None
    return str(value)


def _reasoning_enabled(effort: str | None) -> bool:
    return bool(effort and effort.lower() not in {"none", "minimal", "minimum"})
