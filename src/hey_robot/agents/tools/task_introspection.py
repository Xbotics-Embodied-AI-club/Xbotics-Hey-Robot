from __future__ import annotations

import json
from typing import Any


def current_episode_id(ctx: Any) -> str | None:
    envelope = ctx._current_envelope() if callable(ctx._current_envelope) else None
    episode_id = getattr(envelope, "episode_id", None)
    if isinstance(episode_id, str) and episode_id.strip():
        return episode_id
    turn_context = getattr(ctx, "turn_context", None)
    turn_envelope = getattr(turn_context, "envelope", None)
    episode_id = getattr(turn_envelope, "episode_id", None)
    return (
        str(episode_id).strip()
        if isinstance(episode_id, str) and episode_id.strip()
        else None
    )


def current_task_snapshot(ctx: Any) -> dict[str, Any] | None:
    task_runtime = getattr(ctx, "task_runtime", None)
    episode_id = current_episode_id(ctx)
    if task_runtime is None or not episode_id:
        return None
    task = task_runtime.task_runs.load_active(episode_id)
    return task.to_dict() if task is not None else None


def last_execution_feedback_snapshot(ctx: Any) -> dict[str, Any] | None:
    task_runtime = getattr(ctx, "task_runtime", None)
    episode_id = current_episode_id(ctx)
    if task_runtime is None or not episode_id:
        return None
    state = task_runtime.robot_states.load(episode_id)
    if state is None or not isinstance(state.last_execution_feedback, dict):
        return None
    return dict(state.last_execution_feedback)


def recovery_snapshot(ctx: Any) -> dict[str, Any] | None:
    task = current_task_snapshot(ctx)
    if task is None:
        return None
    recovery = task.get("recovery")
    if isinstance(recovery, dict):
        return _enrich_recovery(dict(recovery))
    if task.get("status") == "recovering":
        feedback = last_execution_feedback_snapshot(ctx)
        return _enrich_recovery(
            {
                "strategy": "reobserve",
                "summary": task.get("failure_reason") or "task is currently recovering",
                "metadata": {"last_execution_feedback": feedback},
            }
        )
    return None


def recovery_options(ctx: Any) -> list[dict[str, Any]]:
    recovery = recovery_snapshot(ctx)
    if recovery is None:
        return [
            {
                "option": "continue",
                "label": "Continue",
                "description": "No explicit recovery is required right now.",
            }
        ]
    strategy = str(recovery.get("strategy") or "reobserve")
    summary = str(recovery.get("summary") or "recovery required")
    mappings: dict[str, list[dict[str, Any]]] = {
        "reobserve": [
            {
                "option": "reobserve",
                "label": "Reobserve",
                "description": "Capture fresh perception before continuing.",
                "recommended": True,
                "recommended_tools": [
                    "request_perception",
                    "get_task_context",
                ],
            },
            {
                "option": "wait",
                "label": "Wait",
                "description": "Hold position until new information arrives.",
            },
        ],
        "reposition": [
            {
                "option": "reposition",
                "label": "Reposition",
                "description": "Adjust viewpoint or robot pose before observing again.",
                "recommended": True,
                "recommended_tools": [
                    "get_task_context",
                ],
            },
            {
                "option": "reobserve",
                "label": "Reobserve",
                "description": summary,
                "recommended_tools": ["request_perception"],
            },
        ],
        "safe_abort": [
            {
                "option": "safe_abort",
                "label": "Safe Abort",
                "description": "Stop autonomous execution immediately.",
                "recommended": True,
                "recommended_tools": ["get_task_context", "wait"],
            },
            {
                "option": "ask_operator",
                "label": "Ask Operator",
                "description": "Request human intervention before continuing.",
                "recommended_tools": ["get_task_context", "wait"],
            },
        ],
        "clarify": [
            {
                "option": "clarify",
                "label": "Clarify",
                "description": "Ask the user or operator to resolve the blockage.",
                "recommended": True,
                "recommended_tools": ["get_task_context", "wait"],
            },
            {
                "option": "safe_abort",
                "label": "Safe Abort",
                "description": "Abort if the task cannot continue safely.",
            },
        ],
        "ask_operator": [
            {
                "option": "clarify",
                "label": "Clarify",
                "description": "Ask the user or operator to resolve the blockage.",
                "recommended": True,
                "recommended_tools": ["get_task_context", "wait"],
            },
            {
                "option": "safe_abort",
                "label": "Safe Abort",
                "description": "Abort if the task cannot continue safely.",
            },
        ],
    }
    return mappings.get(
        strategy,
        [
            {
                "option": "clarify",
                "label": "Clarify",
                "description": summary or "Clarify the current recovery need.",
                "recommended": True,
                "recommended_tools": ["get_task_context", "wait"],
            }
        ],
    )


def summarize_task(ctx: Any) -> str:
    task = current_task_snapshot(ctx)
    if task is None:
        return "No active task is tracked for this episode."
    lines = [
        f"Current task: {task.get('root_task') or 'unknown'}",
        f"Status: {task.get('status') or 'unknown'}",
        f"Robot: {task.get('robot_id') or 'unknown'}",
    ]
    attempts = [item for item in task.get("attempts", []) if isinstance(item, dict)]
    if attempts:
        latest = attempts[-1]
        lines.append(
            f"Latest attempt: {latest.get('text') or latest.get('objective') or 'unknown'}"
        )
        lines.append(f"Latest attempt status: {latest.get('status') or 'unknown'}")
        if latest.get("skill_id"):
            lines.append(f"Latest skill: {latest['skill_id']}")
    if task.get("failure_reason"):
        lines.append(f"Failure reason: {task['failure_reason']}")
    feedback = last_execution_feedback_snapshot(ctx)
    if feedback:
        lines.append(f"Last feedback: {feedback.get('summary') or 'available'}")
        if feedback.get("next_hint"):
            lines.append(f"Next hint: {feedback['next_hint']}")
    recovery = recovery_snapshot(ctx)
    if recovery:
        lines.append(
            f"Recovery: {recovery.get('summary') or recovery.get('strategy') or 'required'}"
        )
    recent_tools = (
        ctx.runtime_state.recent_tool_context(limit=4)
        if getattr(ctx, "runtime_state", None)
        else ""
    )
    if isinstance(recent_tools, str) and recent_tools.strip():
        lines.append(recent_tools)
    loop_warning = (
        ctx.runtime_state.loop_warning_context(limit=6)
        if getattr(ctx, "runtime_state", None)
        else ""
    )
    if isinstance(loop_warning, str) and loop_warning.strip():
        lines.append(loop_warning)
    return "\n".join(lines)


def json_dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _enrich_recovery(recovery: dict[str, Any]) -> dict[str, Any]:
    strategy = (
        str(recovery.get("strategy") or "reobserve").strip().lower() or "reobserve"
    )
    recovery.setdefault("recommended_option", _recommended_option(strategy))
    recovery.setdefault("recommended_tools", _recommended_tools(strategy))
    recovery.setdefault("continuation_guidance", _continuation_guidance(strategy))
    return recovery


def _recommended_option(strategy: str) -> str:
    mapping = {
        "clarify": "clarify",
        "reobserve": "reobserve",
        "reposition": "reposition",
        "safe_abort": "safe_abort",
        "ask_operator": "clarify",
    }
    return mapping.get(strategy, "clarify")


def _recommended_tools(strategy: str) -> list[str]:
    mapping = {
        "clarify": ["get_task_context", "wait"],
        "reobserve": ["get_task_context", "request_perception"],
        "reposition": ["get_task_context", "request_perception"],
        "safe_abort": ["get_task_context", "wait"],
        "ask_operator": ["get_task_context", "wait"],
    }
    return list(mapping.get(strategy, ["get_task_context", "wait"]))


def _continuation_guidance(strategy: str) -> str:
    mapping = {
        "clarify": (
            "Explain the blockage, ask one focused clarification, and wait before issuing another robot action."
        ),
        "reobserve": (
            "Collect fresh perception, verify the latest result, then decide whether the original task can continue."
        ),
        "reposition": (
            "Do not retry the original action immediately; first improve viewpoint or pose, then inspect again."
        ),
        "safe_abort": "Do not continue autonomous actuation; stop and wait for operator intervention.",
        "ask_operator": "Route the task back to the user or operator before continuing.",
    }
    return mapping.get(
        strategy, "Resolve the recovery condition before the next robot action."
    )
