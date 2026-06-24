from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from hey_robot.agents.runtime.grounding import is_perception_skill_name
from hey_robot.agents.runtime.state import ToolCallRecord

FeedbackParser = Callable[[str], dict[str, Any] | None]


def reusable_scene_evidence_result(
    records: Sequence[ToolCallRecord],
    requested_capability: str,
    *,
    parse_feedback: FeedbackParser,
) -> str | None:
    """Return reusable scene evidence from this turn, if one exists.

    A later inspect/look request can reuse earlier successful perception evidence
    in the same turn. Failed records are intentionally ignored so recovery can
    retry perception normally.
    """

    if not is_perception_skill_name((requested_capability or "").strip()):
        return None
    for record in reversed(records):
        result = _record_reusable_scene_result(record, parse_feedback=parse_feedback)
        if result:
            return result
    return None


def is_scene_observation_evidence_record(record: ToolCallRecord) -> bool:
    if not record.success:
        return False
    if record.name == "request_perception":
        return True
    if record.name != "request_capability":
        return False
    capability = str(record.arguments.get("capability") or "").strip()
    return is_perception_skill_name(capability)


def _record_reusable_scene_result(
    record: ToolCallRecord, *, parse_feedback: FeedbackParser
) -> str | None:
    if not is_scene_observation_evidence_record(record):
        return None
    text = record.result.strip()
    if not text:
        return None
    if record.name == "request_perception":
        return (
            text
            + "\n\nPerception result reused: fresh scene evidence already succeeded in this turn; do not call inspect_scene again unless the user asks for a refresh."
        )
    parsed = parse_feedback(text)
    if parsed is not None and parsed.get("subgoal_success") is not True:
        return None
    if parsed is not None:
        return (
            text
            + "\n\nPerception result reused: the same perception skill already succeeded in this turn; do not call it again unless new evidence is required."
        )
    return (
        text
        + "\n\nPerception result reused: the same perception skill already returned a result in this turn."
    )
