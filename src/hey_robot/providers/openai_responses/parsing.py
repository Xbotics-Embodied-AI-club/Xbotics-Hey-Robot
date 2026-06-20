"""Parse OpenAI Responses API response objects.

Ported from nanobot's provider layer and adjusted to hey-robot's native
ReasoningResponse / ReasoningToolCall types.
"""

from __future__ import annotations

import json
from typing import Any

from hey_robot.logging import HeyRobotLogger
from hey_robot.providers.types import ReasoningResponse, ReasoningToolCall

logger = HeyRobotLogger(name="parser")

json_repair: Any
try:
    import json_repair
except ModuleNotFoundError:  # pragma: no cover - exercised only in lean envs
    json_repair = None

FINISH_REASON_MAP = {
    "completed": "stop",
    "incomplete": "length",
    "failed": "error",
    "cancelled": "error",
}


def map_finish_reason(status: str | None) -> str:
    """Map a Responses API status string to a Chat-Completions-style finish_reason."""
    return FINISH_REASON_MAP.get(status or "completed", "stop")


def parse_response_output(response: Any) -> ReasoningResponse:
    """Parse an SDK ``Response`` object into a ReasoningResponse."""
    if not isinstance(response, dict):
        dump = getattr(response, "model_dump", None)
        response = dump() if callable(dump) else vars(response)

    output = response.get("output") or []
    content_parts: list[str] = []
    tool_calls: list[ReasoningToolCall] = []
    reasoning_content: str | None = None

    for item in output:
        if not isinstance(item, dict):
            dump = getattr(item, "model_dump", None)
            item = dump() if callable(dump) else vars(item)

        item_type = item.get("type")
        if item_type == "message":
            for block in item.get("content") or []:
                if not isinstance(block, dict):
                    dump = getattr(block, "model_dump", None)
                    block = dump() if callable(dump) else vars(block)
                if block.get("type") == "output_text":
                    content_parts.append(block.get("text") or "")
        elif item_type == "reasoning":
            for summary in item.get("summary") or []:
                if not isinstance(summary, dict):
                    dump = getattr(summary, "model_dump", None)
                    summary = dump() if callable(dump) else vars(summary)
                if summary.get("type") == "summary_text" and summary.get("text"):
                    reasoning_content = (reasoning_content or "") + summary["text"]
        elif item_type == "function_call":
            call_id = item.get("call_id") or ""
            item_id = item.get("id") or "fc_0"
            args_raw = item.get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except Exception:
                logger.warning(
                    f"tool call 参数解析失败 {item.get('name')!r}: {str(args_raw)[:200]}"
                )
                args = (
                    json_repair.loads(args_raw)
                    if json_repair is not None and isinstance(args_raw, str)
                    else args_raw
                )
                if not isinstance(args, dict):
                    args = {"raw": args_raw}
            tool_calls.append(
                ReasoningToolCall(
                    id=f"{call_id}|{item_id}",
                    name=item.get("name") or "",
                    arguments=args if isinstance(args, dict) else {},
                )
            )

    usage_raw = response.get("usage") or {}
    if not isinstance(usage_raw, dict):
        dump = getattr(usage_raw, "model_dump", None)
        usage_raw = dump() if callable(dump) else vars(usage_raw)
    usage = {}
    if usage_raw:
        usage = {
            "prompt_tokens": int(usage_raw.get("input_tokens") or 0),
            "completion_tokens": int(usage_raw.get("output_tokens") or 0),
            "total_tokens": int(usage_raw.get("total_tokens") or 0),
        }

    status = response.get("status")
    finish_reason = map_finish_reason(status)

    return ReasoningResponse(
        content="".join(content_parts) or None,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage,
        reasoning_content=reasoning_content
        if isinstance(reasoning_content, str)
        else None,
    )
