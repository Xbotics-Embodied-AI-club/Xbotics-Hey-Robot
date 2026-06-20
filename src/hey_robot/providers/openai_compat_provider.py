from __future__ import annotations

import base64
import io
import json
from typing import Any

import numpy as np
from PIL import Image

from hey_robot.providers.base import BaseReasoningProvider, GenerationSettings
from hey_robot.providers.openai_responses import (
    convert_messages,
    convert_tools,
    parse_response_output,
)
from hey_robot.providers.types import (
    ReasoningImage,
    ReasoningMessage,
    ReasoningResponse,
    ReasoningToolCall,
)

json_repair: Any
try:
    import json_repair
except ModuleNotFoundError:  # pragma: no cover - exercised only in lean envs
    json_repair = None


class OpenAICompatReasoningProvider(BaseReasoningProvider):
    """OpenAI-compatible model provider with nanobot-style protocol handling.

    Chat Completions is used for generic OpenAI-compatible gateways. Direct
    OpenAI GPT-5/o-series and explicit reasoning-effort requests use the
    Responses API so tool calls are emitted through the provider protocol.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        generation: GenerationSettings | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        use_responses_api: bool | None = None,
        provider_name: str | None = None,
        supports_required_tool_choice: bool = True,
    ) -> None:
        super().__init__(generation=generation)
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.extra_headers = extra_headers
        self.extra_body = extra_body
        self.use_responses_api = use_responses_api
        self.provider_name = provider_name
        self.supports_required_tool_choice = supports_required_tool_choice
        self._client: Any | None = None

    def _client_or_create(self) -> Any:
        if self._client is not None:
            return self._client
        from openai import AsyncOpenAI

        api_key = self.api_key or ""
        api_base = self.api_base
        if not api_key:
            raise ValueError("model provider requires api_key")
        if api_base:
            self._client = AsyncOpenAI(
                api_key=api_key, base_url=api_base, default_headers=self.extra_headers
            )
        else:
            self._client = AsyncOpenAI(
                api_key=api_key, default_headers=self.extra_headers
            )
        return self._client

    async def chat(
        self,
        *,
        messages: list[ReasoningMessage],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ReasoningResponse:
        client = self._client_or_create()
        model_name = model or self.model
        max_output = max(
            1, int(self.generation.max_tokens if max_tokens is None else max_tokens)
        )
        temp = self.generation.temperature if temperature is None else temperature
        effort = (
            self.generation.reasoning_effort
            if reasoning_effort is None
            else reasoning_effort
        )
        openai_messages = [
            _to_openai_message(
                message, force_string_content=self.provider_name == "deepseek"
            )
            for message in messages
        ]
        openai_tools = [_to_openai_tool(tool) for tool in tools or []]
        effective_tool_choice = self._effective_tool_choice(tool_choice)
        try:
            if self._should_use_responses_api(model_name, effort):
                response = await client.responses.create(
                    **self._build_responses_body(
                        openai_messages,
                        openai_tools,
                        model_name,
                        max_output,
                        temp,
                        effort,
                        effective_tool_choice,
                    )
                )
                parsed = parse_response_output(response)
                return _validate_required_tool_call(
                    parsed, effective_tool_choice, openai_tools
                )
            response = await client.chat.completions.create(
                **self._build_chat_kwargs(
                    openai_messages,
                    openai_tools,
                    model_name,
                    max_output,
                    temp,
                    effort,
                    effective_tool_choice,
                )
            )
        except Exception as exc:
            return _error_response(exc)
        parsed = _parse_chat_response(response)
        return _validate_required_tool_call(parsed, effective_tool_choice, openai_tools)

    def get_default_model(self) -> str:
        return self.model

    def _should_use_responses_api(
        self, model_name: str, reasoning_effort: str | None
    ) -> bool:
        if self.use_responses_api is not None:
            return self.use_responses_api
        if not _is_direct_openai_base(self.api_base):
            return False
        lowered = model_name.lower()
        if reasoning_effort and reasoning_effort.lower() != "none":
            return True
        return any(token in lowered for token in ("gpt-5", "o1", "o3", "o4"))

    def _effective_tool_choice(
        self, tool_choice: str | dict[str, Any] | None
    ) -> str | dict[str, Any] | None:
        if self.supports_required_tool_choice:
            return tool_choice
        if tool_choice == "required":
            return "auto"
        return tool_choice

    def _build_chat_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model_name: str,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
        }
        if _supports_temperature(model_name, reasoning_effort):
            kwargs["temperature"] = temperature
        if _uses_max_completion_tokens(model_name):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
        if reasoning_effort and reasoning_effort.lower() != "none":
            kwargs["reasoning_effort"] = reasoning_effort
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        if self.provider_name == "deepseek" and _deepseek_thinking_enabled(
            model_name, reasoning_effort
        ):
            kwargs.setdefault("extra_body", {}).update(
                {"thinking": {"type": "enabled"}}
            )
            for message in kwargs["messages"]:
                if (
                    message.get("role") == "assistant"
                    and "reasoning_content" not in message
                ):
                    message["reasoning_content"] = ""
        if self.extra_body:
            kwargs["extra_body"] = _deep_merge(
                kwargs.get("extra_body", {}), self.extra_body
            )
        return kwargs

    def _build_responses_body(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model_name: str,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, Any]:
        instructions, input_items = convert_messages(messages)
        body: dict[str, Any] = {
            "model": model_name,
            "instructions": instructions or None,
            "input": input_items,
            "max_output_tokens": max_tokens,
            "store": False,
            "stream": False,
        }
        if _supports_temperature(model_name, reasoning_effort):
            body["temperature"] = temperature
        if reasoning_effort and reasoning_effort.lower() != "none":
            body["reasoning"] = {"effort": reasoning_effort}
            body["include"] = ["reasoning.encrypted_content"]
        if tools:
            body["tools"] = convert_tools(tools)
            body["tool_choice"] = tool_choice or "auto"
        if self.extra_body:
            body.update(self.extra_body)
        return body


def _to_openai_message(
    message: ReasoningMessage, *, force_string_content: bool = False
) -> dict[str, Any]:
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id or "",
            "name": message.tool_name,
            "content": message.content,
        }
    if message.role == "assistant" and message.tool_calls:
        return {
            "role": "assistant",
            "content": message.content or None,
            "tool_calls": [
                _to_openai_tool_call(tool_call) for tool_call in message.tool_calls
            ],
        }
    if not message.images:
        return {"role": message.role, "content": message.content}
    if force_string_content:
        return {"role": message.role, "content": message.content}
    content: list[dict[str, Any]] = [{"type": "text", "text": message.content}]
    content.extend(_image_block(image) for image in message.images)
    return {"role": message.role, "content": content}


def _to_openai_tool_call(tool_call: ReasoningToolCall) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
        },
    }


def _image_block(image: ReasoningImage) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{image.media_type};base64,{_encode_image(image)}",
            "detail": image.detail,
        },
    }


def _encode_image(image: ReasoningImage) -> str:
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


def _to_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") == "function":
        return tool
    input_schema = (
        tool.get("inputSchema")
        or tool.get("input_schema")
        or {"type": "object", "properties": {}}
    )
    return {
        "type": "function",
        "function": {
            "name": str(tool.get("name") or ""),
            "description": str(tool.get("description") or ""),
            "parameters": input_schema,
        },
    }


def _parse_chat_response(response: Any) -> ReasoningResponse:
    if isinstance(response, dict):
        return _parse_chat_response_dict(response)
    response_dump = getattr(response, "model_dump", None)
    if callable(response_dump):
        return _parse_chat_response_dict(response_dump())

    choice = response.choices[0]
    message = choice.message
    tool_calls: list[ReasoningToolCall] = []
    for call in getattr(message, "tool_calls", None) or []:
        function = call.function
        tool_calls.append(
            ReasoningToolCall(
                id=str(call.id),
                name=str(function.name),
                arguments=_parse_arguments(getattr(function, "arguments", "{}")),
            )
        )
    usage = getattr(response, "usage", None)
    usage_dict = {}
    if usage is not None:
        usage_dict = {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
    return ReasoningResponse(
        content=getattr(message, "content", None),
        tool_calls=tool_calls,
        finish_reason=str(getattr(choice, "finish_reason", None) or "stop"),
        usage=usage_dict,
        reasoning_content=getattr(message, "reasoning_content", None) or None,
    )


def _parse_chat_response_dict(response: dict[str, Any]) -> ReasoningResponse:
    choices = response.get("choices") or []
    if not choices:
        return ReasoningResponse(
            content="Error: API returned empty choices.", finish_reason="error"
        )
    content: str | None = None
    reasoning_content: str | None = None
    finish_reason = "stop"
    raw_tool_calls: list[Any] = []
    for choice in choices:
        message = choice.get("message") or {}
        finish_reason = str(choice.get("finish_reason") or finish_reason)
        if not content:
            content = _extract_text_content(message.get("content"))
        if not reasoning_content:
            reasoning_content = _extract_text_content(
                message.get("reasoning_content") or message.get("reasoning")
            )
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            raw_tool_calls.extend(tool_calls)
            if choice.get("finish_reason") in ("tool_calls", "stop"):
                finish_reason = str(choice["finish_reason"])
    parsed_tool_calls = []
    for call in raw_tool_calls:
        function = (call.get("function") or {}) if isinstance(call, dict) else {}
        parsed_tool_calls.append(
            ReasoningToolCall(
                id=str(call.get("id") or ""),
                name=str(function.get("name") or ""),
                arguments=_parse_arguments(function.get("arguments", "{}")),
            )
        )
    usage = response.get("usage") or {}
    usage_dict = (
        {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        }
        if isinstance(usage, dict) and usage
        else {}
    )
    return ReasoningResponse(
        content=content,
        tool_calls=parsed_tool_calls,
        finish_reason=finish_reason,
        usage=usage_dict,
        reasoning_content=reasoning_content,
    )


def _parse_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str) or not arguments.strip():
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        if json_repair is None:
            return {}
        parsed = json_repair.loads(arguments)
    return parsed if isinstance(parsed, dict) else {}


def _validate_required_tool_call(
    response: ReasoningResponse,
    tool_choice: str | dict[str, Any] | None,
    tools: list[dict[str, Any]],
) -> ReasoningResponse:
    if not tools or tool_choice != "required" or response.tool_calls:
        return response
    return ReasoningResponse(
        content=(
            "Model provider violated the native tool-call protocol: "
            "tool_choice=required was requested, but the provider returned no tool_calls."
        ),
        finish_reason="error",
        error_kind="tool_protocol",
    )


def _extract_text_content(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts) or None
    return str(value)


def _supports_temperature(model_name: str, reasoning_effort: str | None = None) -> bool:
    if reasoning_effort and reasoning_effort.lower() != "none":
        return False
    name = model_name.lower()
    return not any(token in name for token in ("gpt-5", "o1", "o3", "o4"))


def _uses_max_completion_tokens(model_name: str) -> bool:
    name = model_name.lower()
    return any(token in name for token in ("gpt-5", "o1", "o3", "o4"))


def _is_direct_openai_base(api_base: str | None) -> bool:
    if not api_base:
        return True
    normalized = api_base.lower()
    return "api.openai.com" in normalized and "openrouter" not in normalized


def _deepseek_thinking_enabled(model_name: str, reasoning_effort: str | None) -> bool:
    effort = (reasoning_effort or "").lower()
    if effort in {"none", "minimal", "minimum"}:
        return False
    return any(
        token in model_name.lower() for token in ("deepseek-v4", "deepseek-reasoner")
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _error_response(exc: Exception) -> ReasoningResponse:
    status_code = getattr(exc, "status_code", None)
    error = getattr(exc, "body", None)
    error_type = None
    error_code = None
    if isinstance(error, dict):
        payload = error.get("error", error)
        if isinstance(payload, dict):
            error_type = payload.get("type")
            error_code = payload.get("code")
    return ReasoningResponse(
        content=f"Error calling model provider: {type(exc).__name__}: {exc}",
        finish_reason="error",
        error_status_code=int(status_code) if status_code is not None else None,
        error_type=str(error_type) if error_type else None,
        error_code=str(error_code) if error_code else None,
    )
