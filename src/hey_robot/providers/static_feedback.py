from __future__ import annotations

from hey_robot.providers.base import BaseReasoningProvider, GenerationSettings
from hey_robot.providers.types import ReasoningMessage, ReasoningResponse


class DeterministicExecutionFeedbackReasoner(BaseReasoningProvider):
    """Fixed feedback provider for tests or explicit feedback-only deployments."""

    def __init__(self) -> None:
        super().__init__(generation=GenerationSettings(temperature=0.0, max_tokens=128))

    async def chat(
        self,
        *,
        messages: list[ReasoningMessage],
        tools: list[dict] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        tool_choice: str | dict | None = None,
    ) -> ReasoningResponse:
        del (
            messages,
            tools,
            model,
            max_tokens,
            temperature,
            reasoning_effort,
            tool_choice,
        )
        return ReasoningResponse(
            content=(
                '{"subgoal_success": true, "task_success": false, '
                '"summary": "execution feedback disabled or deterministic", '
                '"next_hint": "continue with the next useful skill"}'
            ),
            finish_reason="stop",
        )

    def get_default_model(self) -> str:
        return "deterministic-execution-feedback-reasoner"
