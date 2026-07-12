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


class Provider:
    def __init__(self, response: ReasoningResponse) -> None:
        self.response = response
        self.calls = 0

    async def chat(self, **_: object) -> ReasoningResponse:
        self.calls += 1
        return self.response

    def get_default_model(self) -> str:
        return "test"


def _runner(response: ReasoningResponse) -> tuple[StrictAgentRunner, Provider]:
    catalog = SkillCatalog((SkillSpec(name="move", description="move"),))
    provider = Provider(response)
    return StrictAgentRunner(
        provider, build_agent_tools(AgentToolDependencies(catalog))
    ), provider


@pytest.mark.asyncio
async def test_strict_runner_returns_one_proposal_from_one_request() -> None:
    runner, provider = _runner(
        ReasoningResponse(
            tool_calls=[
                ReasoningToolCall(
                    "c",
                    "request_skill",
                    {"skill": "move", "objective": "go", "slots": {}},
                )
            ],
            finish_reason="tool_calls",
        )
    )
    result = await runner.run(
        AgentRunRequest(
            (ReasoningMessage("system", "x"),),
            frozenset({"request_observation", "request_skill"}),
            time.monotonic() + 1,
            "run",
            "d",
        )
    )
    assert result.status == "action_proposed"
    assert result.proposal is not None and result.proposal.skill_name == "move"
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_strict_runner_rejects_multiple_proposals() -> None:
    runner, _ = _runner(
        ReasoningResponse(
            tool_calls=[
                ReasoningToolCall("a", "request_observation", {"question": "x"}),
                ReasoningToolCall("b", "request_observation", {"question": "y"}),
            ]
        )
    )
    result = await runner.run(
        AgentRunRequest(
            (ReasoningMessage("system", "x"),),
            frozenset({"request_observation", "request_skill"}),
            time.monotonic() + 1,
            "run",
            "d",
        )
    )
    assert result.failure is not None
    assert result.failure.code == "MULTIPLE_ACTION_PROPOSALS"
