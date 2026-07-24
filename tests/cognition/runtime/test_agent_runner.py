from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from hey_robot.cognition.runtime.agent_runner import (
    AgentRunner,
    AgentTurnRequest,
)
from hey_robot.cognition.tools.robot import ToolDependencies, ToolRegistry
from hey_robot.providers import ReasoningMessage, ReasoningResponse, ReasoningToolCall
from hey_robot.skills.models import Skill, SkillResult


class SkillList:
    def __init__(self, skills: tuple[Skill, ...]) -> None:
        self._skills = {skill.name: skill for skill in skills}

    def get(self, name: str) -> Skill:
        return self._skills[name]

    def list(self) -> tuple[Skill, ...]:
        return tuple(self._skills.values())


async def _noop(*_args: Any, **_kwargs: Any) -> SkillResult:
    return SkillResult(True, "ok", "completed")


class Provider:
    def __init__(
        self,
        response: ReasoningResponse | None = None,
        *,
        error: Exception | None = None,
        delay: float = 0,
    ) -> None:
        self.response = response
        self.error = error
        self.delay = delay
        self.calls: list[dict] = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.response


def _tools():
    return ToolRegistry(
        ToolDependencies(
            SkillList(
                (
                    Skill(
                        name="move",
                        description="move",
                        parameters={"type": "object", "properties": {}},
                        handler=_noop,
                    ),
                    Skill(
                        name="inspect_scene",
                        description="inspect",
                        parameters={
                            "type": "object",
                            "properties": {
                                "question": {"type": "string", "minLength": 1}
                            },
                            "required": ["question"],
                            "additionalProperties": False,
                        },
                        handler=_noop,
                    ),
                )
            ),
        )
    )


@pytest.mark.asyncio
async def test_conversation_can_return_text_with_the_shared_runner() -> None:
    provider = Provider(ReasoningResponse(content="你好"))
    runner = AgentRunner(provider, _tools())
    result = await runner.run(
        AgentTurnRequest(
            (ReasoningMessage("user", "你好"),),
            frozenset({"move", "complete_task"}),
            time.monotonic() + 1,
            "turn-1",
        )
    )
    assert result.status == "returned"
    assert result.final_text == "你好"


@pytest.mark.asyncio
async def test_removed_start_task_tool_is_rejected() -> None:
    provider = Provider(
        ReasoningResponse(
            tool_calls=[
                ReasoningToolCall(
                    "g1",
                    "start_task",
                    {"objective": "find the cup"},
                )
            ]
        )
    )
    runner = AgentRunner(provider, _tools())
    result = await runner.run(
        AgentTurnRequest(
            (ReasoningMessage("system", "continue goal"),),
            frozenset({"inspect_scene", "move"}),
            time.monotonic() + 1,
            "goal-1",
        )
    )
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.code == "UNKNOWN_TOOL"
    sent_names = {item["function"]["name"] for item in provider.calls[0]["tools"]}
    assert sent_names == {"inspect_scene", "move"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "deadline_offset", "expected_code"),
    [
        (Provider(error=RuntimeError("boom")), 1, "PROVIDER_ERROR"),
        (
            Provider(ReasoningResponse(content="", finish_reason="stop")),
            1,
            "EMPTY_MODEL_RESPONSE",
        ),
        (
            Provider(ReasoningResponse(content="bad", finish_reason="error")),
            1,
            "PROVIDER_ERROR",
        ),
        (
            Provider(ReasoningResponse(content="late"), delay=0.1),
            0.01,
            "PROVIDER_TIMEOUT",
        ),
    ],
)
async def test_provider_and_protocol_failures_are_typed(
    provider: Provider, deadline_offset: float, expected_code: str
) -> None:
    result = await AgentRunner(provider, _tools()).run(
        AgentTurnRequest(
            (ReasoningMessage("system", "test"),),
            frozenset({"inspect_scene", "move"}),
            time.monotonic() + deadline_offset,
            "failure-turn",
        )
    )
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.code == expected_code


@pytest.mark.asyncio
async def test_elapsed_deadline_does_not_call_provider() -> None:
    provider = Provider(ReasoningResponse(content="unused"))
    result = await AgentRunner(provider, _tools()).run(
        AgentTurnRequest(
            (ReasoningMessage("system", "test"),),
            frozenset({"move"}),
            time.monotonic() - 1,
            "expired",
        )
    )
    assert result.failure is not None
    assert result.failure.code == "PROVIDER_TIMEOUT"
    assert provider.calls == []


@pytest.mark.asyncio
async def test_invalid_context_and_tool_set_are_rejected_before_provider() -> None:
    provider = Provider(ReasoningResponse(content="unused"))
    runner = AgentRunner(provider, _tools())
    invalid_messages = await runner.run(
        AgentTurnRequest((), frozenset({"move"}), time.monotonic() + 1, "m")
    )
    invalid_tools = await runner.run(
        AgentTurnRequest(
            (ReasoningMessage("system", "test"),),
            frozenset({"missing_tool"}),
            time.monotonic() + 1,
            "t",
        )
    )
    assert invalid_messages.failure is not None
    assert invalid_messages.failure.code == "INVALID_MODEL_MESSAGES"
    assert invalid_tools.failure is not None
    assert invalid_tools.failure.code == "INVALID_TOOL_SET"
    assert provider.calls == []


@pytest.mark.asyncio
async def test_multiple_and_invalid_tool_calls_do_not_produce_proposals() -> None:
    multiple = Provider(
        ReasoningResponse(
            tool_calls=[
                ReasoningToolCall("a", "inspect_scene", {"question": "front"}),
                ReasoningToolCall("b", "inspect_scene", {"question": "back"}),
            ]
        )
    )
    invalid = Provider(
        ReasoningResponse(
            tool_calls=[ReasoningToolCall("c", "inspect_scene", {"question": ""})]
        )
    )
    request = AgentTurnRequest(
        (ReasoningMessage("system", "test"),),
        frozenset({"inspect_scene", "move"}),
        time.monotonic() + 1,
        "tool-errors",
    )
    multiple_result = await AgentRunner(multiple, _tools()).run(request)
    invalid_result = await AgentRunner(invalid, _tools()).run(request)
    assert multiple_result.failure is not None
    assert multiple_result.failure.code == "MULTIPLE_ACTION_PROPOSALS"
    assert invalid_result.failure is not None
    assert invalid_result.failure.code == "INVALID_TOOL_ARGUMENTS"


@pytest.mark.asyncio
async def test_one_valid_skill_call_returns_one_proposal() -> None:
    provider = Provider(
        ReasoningResponse(
            tool_calls=[
                ReasoningToolCall(
                    "move-1",
                    "move",
                    {},
                )
            ]
        )
    )
    result = await AgentRunner(provider, _tools()).run(
        AgentTurnRequest(
            (ReasoningMessage("system", "test"),),
            frozenset({"move"}),
            time.monotonic() + 1,
            "one-action",
        )
    )
    assert result.status == "action_proposed"
    assert result.proposal is not None
    assert result.proposal.skill_name == "move"
    assert len(provider.calls) == 1
