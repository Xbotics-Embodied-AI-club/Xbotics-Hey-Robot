"""Context construction from DeliberationRequest.

No repair, no auto-cropping, no summarization.  Failure to build a valid
context is a structured MODEL_REQUEST / CONTEXT_BUILD failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from hey_robot.protocol import (
    DeliberationRequest,
    FailurePayload,
    RobotObservation,
    RobotStatus,
)
from hey_robot.providers import ReasoningMessage

_SYSTEM_POLICY = (
    "You control a single physical robot. Your ONLY goal is to satisfy the "
    "immutable contract below. You have exactly two tools:\n"
    "- request_observation(question) — request one fresh scene observation\n"
    "- request_skill(skill, objective, slots) — request one physical skill\n\n"
    "RULES:\n"
    "1. Call EXACTLY ONE tool per deliberation. Text alone cannot satisfy the contract.\n"
    "2. Evidence marked SATISFIED means the contract is complete — do not continue.\n"
    "3. If you are unsure, request an observation first.\n"
    "4. Never guess object locations or robot state."
)


@dataclass(frozen=True)
class ContextBuildResult:
    messages: tuple[ReasoningMessage, ...]
    usage: dict[str, int] = field(default_factory=dict)
    failure: FailurePayload | None = None


def build_context(
    request: DeliberationRequest,
    *,
    evaluation_text: str,
    max_system_chars: int = 8000,
    max_evidence_items: int = 64,
) -> ContextBuildResult:
    """Build the structured model context from a DeliberationRequest.

    Returns ContextBuildResult with either messages or a failure.
    """
    parts: dict[str, Any] = {
        "policy": _SYSTEM_POLICY,
        "goal": {
            "goal_id": request.goal.goal_id,
            "objective": request.goal.objective,
            "status": request.goal.status,
        },
        "contract": {
            "contract_id": request.goal.contract_id,
            "contract_hash": request.goal.contract_hash,
            "success_criteria": [
                {
                    "criterion_id": c.criterion_id,
                    "criterion_type": c.criterion_type,
                    "subject_id": c.subject_id,
                    "predicate": c.predicate,
                    "object_id": c.object_id,
                    "max_age_sec": c.max_age_sec,
                }
                for c in request.goal.success_criteria
            ],
        },
        "evaluation": evaluation_text,
        "budget": {
            "elapsed_wall_time_sec": request.budget_state.elapsed_wall_time_sec,
            "deliberations_used": request.budget_state.deliberations_used,
            "skills_used": request.budget_state.skills_used,
            "battery_percentage": request.budget_state.battery_percentage,
        },
    }

    if request.robot_status:
        parts["robot_status"] = _status_summary(request.robot_status)

    if request.robot_observation:
        parts["robot_observation"] = _observation_summary(request.robot_observation)

    if request.actions:
        parts["action_history"] = [
            {
                "skill_id": a.skill_id,
                "deliberation_id": a.deliberation_id,
                "intent_kind": a.intent_kind,
                "name": a.name,
                "objective": a.objective,
                "arguments": a.arguments,
                "status": a.status,
            }
            for a in request.actions
        ]

    if len(request.evidence) > max_evidence_items:
        return ContextBuildResult(
            messages=(),
            failure=FailurePayload(
                "CONTEXT_BUILD",
                "CONTEXT_BUDGET_EXCEEDED",
                "ContextBuilder",
                "evidence count exceeds context budget",
                {"max_evidence_items": max_evidence_items},
            ),
        )
    evidence_items = [
        {
            "evidence_id": e.evidence_id,
            "source_kind": e.source_kind,
            "source_id": e.source_id,
            "observed_at": e.observed_at,
            "subject_id": e.subject_id,
            "predicate": e.predicate,
            "object_id": e.object_id,
        }
        for e in request.evidence
    ]
    if evidence_items:
        parts["evidence"] = evidence_items

    if request.latest_skill_result:
        parts["latest_skill_result"] = {
            "skill_id": request.latest_skill_result.skill_id,
            "name": request.latest_skill_result.name,
            "status": request.latest_skill_result.status,
            "success": request.latest_skill_result.success,
            "summary": request.latest_skill_result.summary,
            "error": request.latest_skill_result.error,
        }

    content = json.dumps(parts, sort_keys=True, default=str)
    if len(content) > max_system_chars:
        return ContextBuildResult(
            messages=(),
            failure=FailurePayload(
                "CONTEXT_BUILD",
                "CONTEXT_BUDGET_EXCEEDED",
                "ContextBuilder",
                f"context size {len(content)} exceeds budget",
                {"max_chars": max_system_chars},
            ),
        )

    message = ReasoningMessage(role="system", content=content)
    return ContextBuildResult(messages=(message,))


def _status_summary(status: RobotStatus) -> dict[str, Any]:
    return {
        "state": status.state,
        "location_id": status.location_id,
        "motion_state": status.motion_state,
        "battery_percentage": status.battery_percentage,
        "skill_id": status.skill_id,
        "success": status.success,
        "error": status.error,
    }


def _observation_summary(obs: RobotObservation) -> dict[str, Any]:
    return {
        "frame_id": obs.frame_id,
        "image_count": len(obs.images or ()),
    }
