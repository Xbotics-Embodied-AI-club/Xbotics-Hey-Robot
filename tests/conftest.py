import json
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hey_robot.providers import ReasoningMessage, ReasoningResponse, ReasoningToolCall


class FakeProvider:
    """Test provider that returns native tool-call responses."""

    def __init__(
        self,
        responses: str
        | dict[str, Any]
        | ReasoningResponse
        | list[str | dict[str, Any] | ReasoningResponse],
    ):
        self.responses = (
            [responses] if not isinstance(responses, list) else list(responses)
        )
        self.last_messages: list[ReasoningMessage] | None = None

    async def chat(self, **kwargs: Any) -> ReasoningResponse:
        self.last_messages = list(kwargs.get("messages") or [])
        if self.responses:
            return _to_response(self.responses.pop(0))
        return ReasoningResponse(content="done.", finish_reason="stop")

    def get_default_model(self) -> str:
        return "fake-provider"


def _to_response(item: str | dict[str, Any] | ReasoningResponse) -> ReasoningResponse:
    if isinstance(item, ReasoningResponse):
        return item
    if isinstance(item, str):
        try:
            parsed = json.loads(item)
        except json.JSONDecodeError:
            return ReasoningResponse(content=item, finish_reason="stop")
        if not isinstance(parsed, dict):
            return ReasoningResponse(content=item, finish_reason="stop")
        item = parsed
    if "tool" in item:
        return ReasoningResponse(
            content=item.get("reason"),
            tool_calls=[
                ReasoningToolCall(
                    id=str(uuid4()),
                    name=str(item["tool"]),
                    arguments=dict(item.get("args", {}) or {}),
                    provider_metadata={"plan": list(item.get("plan", []) or [])},
                )
            ],
            finish_reason="tool_calls",
        )
    return ReasoningResponse(
        content=json.dumps(item), finish_reason=str(item.get("finish_reason", "stop"))
    )
