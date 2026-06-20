from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class AgentExecutionFeedbackResult:
    subgoal_success: bool
    task_success: bool
    summary: str
    next_hint: str | None = None
    failure_reason: str | None = None
    confidence: float | None = None


def parse_execution_feedback_response(response: str) -> AgentExecutionFeedbackResult:
    text = (response or "").strip()
    payload = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                payload = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                payload = None

    if isinstance(payload, dict):
        subgoal_success = bool(
            payload.get("subgoal_success", payload.get("success", False))
        )
        task_success = bool(
            payload.get(
                "task_success",
                payload.get("full_task_complete", payload.get("task_complete", False)),
            )
        )
        summary = str(
            payload.get("summary") or payload.get("missing_progress") or text
        ).strip()
        if not summary:
            summary = "execution feedback unavailable"
        next_hint = payload.get("next_hint", payload.get("next_step"))
        next_hint = str(next_hint).strip() if next_hint else None
        failure_reason = payload.get("failure_reason")
        failure_reason = str(failure_reason).strip() if failure_reason else None
        confidence = payload.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None
        return AgentExecutionFeedbackResult(
            subgoal_success=subgoal_success,
            task_success=task_success,
            summary=summary,
            next_hint=next_hint,
            failure_reason=failure_reason,
            confidence=confidence,
        )

    lowered = text.lower()
    success = '"success": true' in lowered or lowered.startswith("success")
    return AgentExecutionFeedbackResult(
        subgoal_success=success,
        task_success=False,
        summary=text or "execution feedback unavailable",
        next_hint=None,
    )
