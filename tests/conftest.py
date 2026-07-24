import json
import os
import platform
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

if platform.system() == "Windows" and os.environ.get("MUJOCO_GL") == "egl":
    os.environ["MUJOCO_GL"] = "wgl"

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hey_robot.model import ModelMessage, ModelResponse, ModelToolCall


class FakeModel:
    """Test model client that returns native tool-call responses."""

    def __init__(
        self,
        responses: str
        | dict[str, Any]
        | ModelResponse
        | list[str | dict[str, Any] | ModelResponse],
    ):
        self.responses = (
            [responses] if not isinstance(responses, list) else list(responses)
        )
        self.last_messages: list[ModelMessage] | None = None

    async def chat(self, **kwargs: Any) -> ModelResponse:
        self.last_messages = list(kwargs.get("messages") or [])
        if self.responses:
            return _to_response(self.responses.pop(0))
        return ModelResponse(content="done.", finish_reason="stop")


def _to_response(item: str | dict[str, Any] | ModelResponse) -> ModelResponse:
    if isinstance(item, ModelResponse):
        return item
    if isinstance(item, str):
        try:
            parsed = json.loads(item)
        except json.JSONDecodeError:
            return ModelResponse(content=item, finish_reason="stop")
        if not isinstance(parsed, dict):
            return ModelResponse(content=item, finish_reason="stop")
        item = parsed
    if "tool" in item:
        return ModelResponse(
            content=item.get("reason"),
            tool_calls=[
                ModelToolCall(
                    id=str(uuid4()),
                    name=str(item["tool"]),
                    arguments=dict(item.get("args", {}) or {}),
                    metadata={"plan": list(item.get("plan", []) or [])},
                )
            ],
            finish_reason="tool_calls",
        )
    return ModelResponse(
        content=json.dumps(item), finish_reason=str(item.get("finish_reason", "stop"))
    )
