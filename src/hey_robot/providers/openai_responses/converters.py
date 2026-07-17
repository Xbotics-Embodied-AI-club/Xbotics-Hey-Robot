"""把 Chat Completions 消息和工具转换为 Responses API 格式。

代码移植自 nanobot 的 provider 层，并按 hey-robot 的 provider 消息结构调整。
"""

from __future__ import annotations

import json
from typing import Any


def convert_messages(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """把 Chat Completions 消息转换为 Responses API input items。

    返回 ``(system_prompt, input_items)``：其中 system_prompt 来自 system
    角色消息，input_items 是 Responses API 的 ``input`` 数组。
    """
    system_prompt = ""
    input_items: list[dict[str, Any]] = []

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            system_prompt = content if isinstance(content, str) else ""
            continue

        if role == "user":
            input_items.append(convert_user_message(content))
            continue

        if role == "assistant":
            if isinstance(content, str) and content:
                assistant_item = {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                    "status": "completed",
                    "id": f"msg_{idx}",
                }
                tool_calls = msg.get("tool_calls", []) or []
                if tool_calls:
                    assistant_item["tool_calls"] = [
                        {
                            "id": tool_call.get("id"),
                            "type": "function_call",
                            "function": tool_call.get("function") or {},
                        }
                        for tool_call in tool_calls
                    ]
                input_items.append(assistant_item)
            for tool_call in msg.get("tool_calls", []) or []:
                fn = tool_call.get("function") or {}
                call_id, item_id = split_tool_call_id(tool_call.get("id"))
                input_items.append(
                    {
                        "type": "function_call",
                        "id": item_id or f"fc_{idx}",
                        "call_id": call_id or f"call_{idx}",
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments") or "{}",
                    }
                )
            continue

        if role == "tool":
            call_id, _ = split_tool_call_id(msg.get("tool_call_id"))
            output_text = (
                content
                if isinstance(content, str)
                else json.dumps(content, ensure_ascii=False)
            )
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output_text,
                }
            )

    return system_prompt, input_items


def convert_user_message(content: Any) -> dict[str, Any]:
    """把用户消息内容转换为 Responses API 格式。

    支持纯字符串、``text`` 块到 ``input_text`` 的转换，以及 ``image_url`` 块到
    ``input_image`` 的转换。
    """
    if isinstance(content, str):
        return {"role": "user", "content": [{"type": "input_text", "text": content}]}
    if isinstance(content, list):
        converted: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                converted.append({"type": "input_text", "text": item.get("text", "")})
            elif item.get("type") == "image_url":
                url = (item.get("image_url") or {}).get("url")
                if url:
                    converted.append(
                        {"type": "input_image", "image_url": url, "detail": "auto"}
                    )
        if converted:
            return {"role": "user", "content": converted}
    return {"role": "user", "content": [{"type": "input_text", "text": ""}]}


def convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 OpenAI function-calling 工具 schema 转为 Responses API 扁平格式。"""
    converted: list[dict[str, Any]] = []
    for tool in tools:
        fn = (tool.get("function") or {}) if tool.get("type") == "function" else tool
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters") or {}
        converted.append(
            {
                "type": "function",
                "name": name,
                "description": fn.get("description") or "",
                "parameters": params if isinstance(params, dict) else {},
            }
        )
    return converted


def split_tool_call_id(tool_call_id: Any) -> tuple[str, str | None]:
    """拆分复合格式的 ``call_id|item_id`` 字符串。"""
    if isinstance(tool_call_id, str) and tool_call_id:
        if "|" in tool_call_id:
            call_id, item_id = tool_call_id.split("|", 1)
            return call_id, item_id or None
        return tool_call_id, None
    return "call_0", None
