from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from hey_robot.providers import ReasoningResponse

ResponseAction = Literal["provider_error", "execute_tools", "text", "empty"]


@dataclass(frozen=True)
class ResponseDecision:
    action: ResponseAction
    content: str = ""
    reason: str = ""


def decide_response(response: ReasoningResponse) -> ResponseDecision:
    """Classify a provider response without executing tools."""
    if response.finish_reason == "error":
        return ResponseDecision(
            action="provider_error",
            content=response.content or "model provider error",
            reason=response.error_kind or "provider_error",
        )
    if response.should_execute_tools:
        return ResponseDecision(action="execute_tools")
    content = (response.content or "").strip()
    if content:
        return ResponseDecision(action="text", content=content, reason="text_response")
    return ResponseDecision(
        action="empty",
        content="no tool call or text from provider",
        reason="empty_response",
    )


def looks_like_unexecuted_tool_protocol(content: str) -> bool:
    text = content.strip().lower()
    if not text:
        return False
    normalized = text.replace("｜", "|").replace("锝滐綔", "")
    markers = (
        "<tool_calls",
        "</tool_calls>",
        "<invoke name=",
        "<||dsml||tool_calls",
        "<||dsml||invoke",
        "<||dsml||parameter",
        "```tool_call",
        "dsml",
    )
    return any(marker in normalized for marker in markers)


def looks_like_internal_agent_protocol(content: str) -> bool:
    """Return true for agent-only continuation/debug payloads.

    These strings are useful as model context, but they must not be surfaced as
    final user-facing replies or used as task-completion summaries.
    """
    text = (content or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered.startswith("execution feedback for skill "):
        return True
    if any(
        token in lowered
        for token in (
            "consecutivemotionblocked:",
            "camerastale:",
            "cameraunsafe:",
            "task continuation:",
            "task recovery guidance:",
        )
    ):
        return True
    markers = (
        "\ntask continuation:",
        "\nrecovery continuation:",
        "\nskill trace:",
        "\n- task_success:",
        "\n- recommended_action:",
        "\n- next_instruction:",
        "\n- remaining_goal:",
    )
    return any(marker in lowered for marker in markers)
