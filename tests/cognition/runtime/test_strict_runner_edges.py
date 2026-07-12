"""Cover all edge cases in StrictAgentRunner."""

from __future__ import annotations

import time

import pytest

from hey_robot.cognition.runtime.result import AgentRunRequest
from hey_robot.cognition.runtime.strict_runner import StrictAgentRunner
from hey_robot.cognition.tools.autonomous import (
    AgentToolDependencies,
    build_agent_tools,
)
from hey_robot.providers import ReasoningMessage, ReasoningResponse, ReasoningToolCall
from hey_robot.skill_os.base import SkillCatalog, SkillSpec


class _FakeProvider:
    def __init__(self, response=None, error=None, delay=0):
        self.response = response
        self.error = error
        self.delay = delay
        self.calls = 0

    async def chat(self, **_kwargs):
        self.calls += 1
        if self.delay:
            await __import__("asyncio").sleep(self.delay)
        if self.error:
            raise self.error
        return self.response

    def get_default_model(self):
        return "fake"


def _runner(provider=None):
    catalog = SkillCatalog((SkillSpec(name="move", description="move"),))
    return StrictAgentRunner(
        provider or _FakeProvider(), build_agent_tools(AgentToolDependencies(catalog))
    )


def _req(
    allowed_tools=frozenset({"request_observation", "request_skill"}), deadline=None
):
    return AgentRunRequest(
        (ReasoningMessage("system", "test"),),
        allowed_tools,
        deadline or (time.monotonic() + 60),
        "run1",
        "d1",
    )


# ── Tool set mismatch ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rejects_wrong_tool_set() -> None:
    runner = _runner()
    result = await runner.run(_req(allowed_tools=frozenset({"wrong_tool"})))
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.code == "INVALID_TOOL_SET"


# ── Deadline ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rejects_elapsed_deadline() -> None:
    runner = _runner()
    result = await runner.run(_req(deadline=time.monotonic() - 10))
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.code == "PROVIDER_TIMEOUT"


# ── Invalid messages ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rejects_non_reasoning_message() -> None:
    runner = _runner()
    result = await runner.run(
        AgentRunRequest(
            ("not a message",),
            frozenset({"request_observation", "request_skill"}),
            time.monotonic() + 60,
            "run1",
            "d2",
        )
    )
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.code == "INVALID_MODEL_RESPONSE"


# ── Provider timeout ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_provider_timeout() -> None:
    provider = _FakeProvider(delay=999)
    runner = _runner(provider)
    result = await runner.run(
        AgentRunRequest(
            (ReasoningMessage("system", "test"),),
            frozenset({"request_observation", "request_skill"}),
            time.monotonic() + 0.01,
            "run1",
            "d3",
        )
    )
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.code == "PROVIDER_TIMEOUT"


# ── Provider error ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_provider_error() -> None:
    provider = _FakeProvider(error=RuntimeError("boom"))
    runner = _runner(provider)
    result = await runner.run(_req())
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.code == "PROVIDER_ERROR"


# ── Provider finish_reason error ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_provider_error_finish_reason() -> None:
    provider = _FakeProvider(
        response=ReasoningResponse(content="err", finish_reason="error")
    )
    runner = _runner(provider)
    result = await runner.run(_req())
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.code == "PROVIDER_ERROR"


# ── Empty model response ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_model_response() -> None:
    provider = _FakeProvider(
        response=ReasoningResponse(content="", finish_reason="stop")
    )
    runner = _runner(provider)
    result = await runner.run(_req())
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.code == "EMPTY_MODEL_RESPONSE"


# ── MODEL_STOPPED_BEFORE_GOAL (returned text) ────────────────────────────


@pytest.mark.asyncio
async def test_model_returned_text_without_tool() -> None:
    provider = _FakeProvider(
        response=ReasoningResponse(content="I give up.", finish_reason="stop")
    )
    runner = _runner(provider)
    result = await runner.run(_req())
    assert result.status == "returned"
    assert result.final_text == "I give up."


# ── Tool call with unknown name ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_tool_in_response() -> None:
    provider = _FakeProvider(
        response=ReasoningResponse(
            tool_calls=[ReasoningToolCall("c1", "bad_tool", {})],
            finish_reason="tool_calls",
        )
    )
    runner = _runner(provider)
    result = await runner.run(_req())
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.code == "UNKNOWN_TOOL"


# ── Invalid tool arguments ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_invalid_tool_arguments() -> None:
    provider = _FakeProvider(
        response=ReasoningResponse(
            tool_calls=[
                ReasoningToolCall("c1", "request_observation", {"question": ""})
            ],
            finish_reason="tool_calls",
        )
    )
    runner = _runner(provider)
    result = await runner.run(_req())
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.code == "INVALID_TOOL_ARGUMENTS"


@pytest.mark.asyncio
async def test_request_skill_missing_required_fields() -> None:
    provider = _FakeProvider(
        response=ReasoningResponse(
            tool_calls=[
                ReasoningToolCall(
                    "c1", "request_skill", {"skill": "", "objective": "x"}
                )
            ],
            finish_reason="tool_calls",
        )
    )
    runner = _runner(provider)
    result = await runner.run(_req())
    assert result.status == "failed"
    assert result.failure is not None


@pytest.mark.asyncio
async def test_request_skill_with_observe_category() -> None:
    provider = _FakeProvider(
        response=ReasoningResponse(
            tool_calls=[
                ReasoningToolCall(
                    "c1",
                    "request_skill",
                    {"skill": "inspect_scene", "objective": "look", "slots": {}},
                )
            ],
            finish_reason="tool_calls",
        )
    )
    catalog = SkillCatalog(
        (
            SkillSpec(name="move", description="move"),
            SkillSpec(name="inspect_scene", category="observe", description="look"),
        )
    )
    runner = StrictAgentRunner(
        provider, build_agent_tools(AgentToolDependencies(catalog))
    )
    result = await runner.run(_req())
    assert result.status == "failed"
    assert result.failure is not None
