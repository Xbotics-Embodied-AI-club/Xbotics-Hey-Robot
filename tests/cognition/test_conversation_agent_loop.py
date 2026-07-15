import asyncio

from hey_robot.cognition.runtime.conversation_runner import ConversationToolRunner
from hey_robot.cognition.tools.autonomous import (
    AgentToolDependencies,
    build_agent_tools,
)
from hey_robot.protocol import ToolOutcome
from hey_robot.providers import ReasoningMessage, ReasoningResponse, ReasoningToolCall
from hey_robot.skill_os.base import SkillCatalog, SkillSpec


class _LoopProvider:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return ReasoningResponse(
                tool_calls=[
                    ReasoningToolCall(
                        "bad", "request_skill", {"skill": "move_base", "slots": {}}
                    )
                ]
            )
        if len(self.calls) == 2:
            return ReasoningResponse(
                tool_calls=[
                    ReasoningToolCall(
                        "good",
                        "request_skill",
                        {
                            "skill": "move_base",
                            "objective": "move forward",
                            "slots": {"direction": "forward", "distance_cm": 20},
                        },
                    )
                ]
            )
        return ReasoningResponse(content="Moved forward 20 cm.")


def test_agent_loop_retries_after_contract_error_then_returns_final_answer() -> None:
    provider = _LoopProvider()
    catalog = SkillCatalog(
        (
            SkillSpec(
                name="move_base",
                description="move",
                input_schema={
                    "type": "object",
                    "properties": {
                        "direction": {"type": "string"},
                        "distance_cm": {"type": "number"},
                    },
                    "required": ["direction", "distance_cm"],
                },
            ),
        )
    )
    tools = build_agent_tools(AgentToolDependencies(catalog))
    executed = []

    async def execute(proposal):
        executed.append(proposal)
        return ToolOutcome("completed", "motion completed")

    result = asyncio.run(
        ConversationToolRunner(provider, tools).run(
            [ReasoningMessage(role="user", content="move forward")], execute
        )
    )

    assert result == "Moved forward 20 cm."
    assert len(executed) == 1
    assert executed[0].arguments["distance_cm"] == 20
    assert provider.calls[1]["tools"] == tools.definitions
