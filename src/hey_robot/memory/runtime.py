from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from hey_robot.memory.long_term import LongTermMemoryStore

MemoryKind = Literal[
    "event",
    "entity",
    "place",
    "task_result",
    "skill_experience",
    "user_preference",
    "scene_anchor",
    "task_lesson",
]


class _AutonomyMemory(Protocol):
    def remember(
        self, event_type: str, summary: str, *, frame_id: int | None = None
    ) -> None: ...


class MemoryRuntime:
    """Small runtime boundary between agent code and memory stores.

    The long-term store remains a simple JSONL persistence layer. Agent tools,
    prompt construction, and tool-result hooks talk to this runtime so memory
    policy does not leak across the core, runner, and tools.
    """

    def __init__(
        self,
        store: LongTermMemoryStore,
        *,
        autonomy: _AutonomyMemory | None = None,
    ) -> None:
        self.store = store
        self.autonomy = autonomy

    @classmethod
    def from_path(
        cls, path: str | Path, *, autonomy: _AutonomyMemory | None = None
    ) -> MemoryRuntime:
        return cls(LongTermMemoryStore(path), autonomy=autonomy)

    def build_agent_context(
        self, task: str, current_context: str | None = None, *, limit: int = 6
    ) -> str | None:
        long_term = self.store.prompt_context(task, limit=limit)
        parts = [part for part in (current_context, long_term) if part]
        return "\n\n".join(parts) if parts else None

    def search(
        self,
        *,
        query: str | None = None,
        kind: str | None = None,
        mode: str | None = None,
        limit: int = 8,
        fallback_query: str = "",
    ) -> str:
        effective_query = query or ""
        effective_mode = mode or "semantic"
        if effective_mode == "semantic":
            return self.store.prompt_context(
                effective_query or fallback_query, limit=limit
            ) or ("no relevant memory")

        memory_kind: MemoryKind | None = None
        if kind in {
            "event",
            "entity",
            "place",
            "task_result",
            "skill_experience",
            "user_preference",
            "scene_anchor",
            "task_lesson",
        }:
            memory_kind = cast(MemoryKind, kind)
        records = self.store.query(effective_query, kind=memory_kind, limit=limit)
        return json.dumps([record.to_dict() for record in records], ensure_ascii=False)

    def write(
        self,
        *,
        kind: str,
        summary: str | None = None,
        name: str | None = None,
        value: str | None = None,
        location: str | None = None,
        entity_type: str | None = None,
        confidence: float | None = None,
        skill_name: str | None = None,
        success: bool | None = None,
        object_name: str | None = None,
        failure_mode: str | None = None,
        recovery_hint: str | None = None,
        verification_summary: str | None = None,
        duration_sec: float | None = None,
        attributes: str | None = None,
        turn_context: Any = None,
        task: str = "",
    ) -> str:
        if not kind:
            raise ValueError("kind is required for memory write")

        snapshot = turn_context.snapshot if turn_context else None
        frame_id = snapshot.status.frame_id if snapshot and snapshot.status else None
        robot_id = snapshot.robot_id if snapshot else None

        if kind == "event":
            effective_summary = summary or f"event: {name or 'note'}"
            if self.autonomy is not None:
                self.autonomy.remember(
                    name or "note", effective_summary, frame_id=frame_id
                )
            rec = self.store.remember(
                kind="event",
                key=name or "note",
                summary=effective_summary,
                metadata={"frame_id": frame_id, "source": "write_memory"},
            )
            return json.dumps(rec.to_dict(), ensure_ascii=False)

        if kind == "entity":
            rec = self.store.remember_entity(
                name=name or "entity",
                summary=summary or f"{name} at {location or 'unknown'}",
                entity_type=entity_type or "object",
                location=location,
                confidence=confidence or 1.0,
                attributes=_parse_attributes(attributes),
                frame_id=frame_id,
                robot_id=robot_id,
            )
            return json.dumps(rec.to_dict(), ensure_ascii=False)

        if kind == "place":
            rec = self.store.remember_place(
                name=name or "place",
                description=summary or f"place: {name or 'unknown'}",
                confidence=confidence or 1.0,
                robot_id=robot_id,
            )
            return json.dumps(rec.to_dict(), ensure_ascii=False)

        if kind == "user_preference":
            preference_name = name or "user_preference"
            preference_value = value or ""
            rec = self.store.remember_user_preference(
                name=preference_name,
                value=preference_value,
                summary=summary
                or f"User preference: {preference_name}={preference_value}",
                confidence=confidence or 1.0,
            )
            return json.dumps(rec.to_dict(), ensure_ascii=False)

        if kind == "scene_anchor":
            rec = self.store.record_scene_anchor(
                name=name or "scene_anchor",
                location=location or "unknown",
                summary=summary
                or f"{name or 'scene anchor'} is usually at {location or 'unknown'}",
                entity_type=entity_type or "object",
                confidence=confidence or 1.0,
                frame_id=frame_id,
                robot_id=robot_id,
            )
            return json.dumps(rec.to_dict(), ensure_ascii=False)

        if kind == "task_lesson":
            rec = self.store.remember_task_lesson(
                key=name or skill_name or "task_lesson",
                summary=summary or "task lesson",
                task=task,
                success=success,
                failure_mode=failure_mode,
                recovery_hint=recovery_hint,
                skill_name=skill_name,
            )
            return json.dumps(rec.to_dict(), ensure_ascii=False)

        if kind == "location":
            entity_summary = (
                summary or f"{name or 'entity'} is at {location or 'unknown'}"
            )
            place_description = (
                summary or f"{name or 'entity'} at {location or 'unknown'}"
            )
            entity_rec = self.store.remember_entity(
                name=name or "entity",
                summary=entity_summary,
                entity_type=entity_type or "object",
                location=location,
                confidence=confidence or 1.0,
                frame_id=frame_id,
                robot_id=robot_id,
            )
            place_rec = self.store.remember_place(
                name=location or name or "place",
                description=place_description,
                confidence=confidence or 1.0,
                robot_id=robot_id,
            )
            return json.dumps(
                {"entity": entity_rec.to_dict(), "place": place_rec.to_dict()},
                ensure_ascii=False,
            )

        if kind == "task_result":
            rec = self.store.remember_task_result(
                task_id=name or "task_result",
                root_goal=task,
                status="completed" if success is not False else "failed",
                summary=summary or "task result",
                failure_reason=failure_mode,
            )
            return json.dumps(rec.to_dict(), ensure_ascii=False)

        if kind != "skill_experience":
            raise ValueError(f"unknown memory kind for write: {kind}")

        rec = self.store.remember_skill_experience(
            skill_name=skill_name or "skill_experience",
            arguments={"object": object_name, "location": location},
            context_summary=turn_context.observation_summary if turn_context else "",
            success=bool(success) if success is not None else True,
            summary=summary or ("skill succeeded" if success else "skill failed"),
            failure_mode=failure_mode,
            recovery_hint=recovery_hint,
            verification_summary=verification_summary,
            duration_sec=duration_sec,
        )
        return json.dumps(rec.to_dict(), ensure_ascii=False)

    def record_tool_result(
        self,
        tool: str,
        args: dict[str, Any],
        result: str,
        success: bool,
        *,
        context_summary: str = "",
    ) -> None:
        if tool != "request_capability":
            return
        if _is_transient_safety_gate_result(result):
            return
        skill_name = str(args.get("capability") or "request_capability").strip()
        (
            memory_summary,
            failure_mode,
            recovery_hint,
            verification_summary,
            memory_success,
        ) = _normalize_skill_result_for_memory(
            result,
            success=success,
        )
        self.store.remember_skill_experience(
            skill_name=skill_name,
            arguments=_skill_experience_arguments(args),
            context_summary=context_summary,
            success=memory_success,
            summary=memory_summary,
            failure_mode=failure_mode,
            recovery_hint=recovery_hint,
            verification_summary=verification_summary,
        )
        if failure_mode or recovery_hint or not memory_success:
            objective = str(args.get("objective") or "").strip()
            self.store.remember_task_lesson(
                key=skill_name,
                summary=memory_summary,
                task=objective,
                success=memory_success,
                failure_mode=failure_mode,
                recovery_hint=recovery_hint,
                skill_name=skill_name,
            )


def _parse_attributes(raw: str | None) -> dict[str, object] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _skill_experience_arguments(args: dict[str, Any]) -> dict[str, Any]:
    slots = args.get("slots")
    if isinstance(slots, dict):
        return dict(slots)
    nested = args.get("arguments")
    if isinstance(nested, dict):
        return dict(nested)
    return {
        key: value
        for key, value in args.items()
        if key not in {"capability", "objective", "interrupt"} and value is not None
    }


def _normalize_skill_result_for_memory(
    result: str,
    *,
    success: bool,
) -> tuple[str, str | None, str | None, str | None, bool]:
    text = (result or "").strip()
    if not text:
        fallback = "skill succeeded" if success else "skill failed"
        failure_mode = None if success else "request_capability_failed"
        return fallback, failure_mode, None, None, bool(success)

    feedback = _parse_agent_execution_feedback(text)
    if feedback is None:
        failure_mode = None if success else "request_capability_failed"
        return text, failure_mode, None, text, bool(success)

    summary = feedback.get("summary") or (
        "skill succeeded" if success else "skill failed"
    )
    failure_reason = feedback.get("failure_reason")
    outcome = str(feedback.get("outcome") or "").strip().lower()
    subgoal_success = feedback.get("subgoal_success")
    if isinstance(subgoal_success, bool):
        memory_success = subgoal_success
    else:
        memory_success = (
            outcome in {"confirmed", "skipped"} if outcome else bool(success)
        )
    failure_mode = (
        None
        if memory_success
        else (failure_reason or outcome or "request_capability_failed")
    )
    recovery_hint = feedback.get("next_hint")
    verification_summary = _verification_summary_from_feedback(feedback)
    return summary, failure_mode, recovery_hint, verification_summary, memory_success


def _is_transient_safety_gate_result(result: str) -> bool:
    text = (result or "").lower()
    transient_markers = (
        "consecutivemotionblocked",
        "cameraunsafe:",
        "camerastale:",
        "requires fresh perception evidence",
        "run inspect_scene or request_perception",
    )
    return any(marker in text for marker in transient_markers)


def _parse_agent_execution_feedback(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    prefix = "Execution feedback for skill "
    if not stripped.startswith(prefix):
        return None
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return None
    first = lines[0]
    skill_name = (
        first[len(prefix) :].rstrip(":").strip() if first.startswith(prefix) else ""
    )
    parsed: dict[str, Any] = {"skill_id": skill_name}
    for line in lines[1:]:
        if not line.startswith("- "):
            continue
        key, sep, value = line[2:].partition(":")
        if not sep:
            continue
        parsed[key.strip()] = _parse_feedback_value(value.strip())
    return parsed


def _parse_feedback_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        return float(value)
    except ValueError:
        return value


def _verification_summary_from_feedback(feedback: dict[str, Any]) -> str | None:
    parts: list[str] = []
    outcome = feedback.get("outcome")
    if outcome:
        parts.append(f"outcome={outcome}")
    if "subgoal_success" in feedback:
        parts.append(f"subgoal_success={feedback.get('subgoal_success')}")
    if "task_success" in feedback:
        parts.append(f"task_success={feedback.get('task_success')}")
    summary = feedback.get("summary")
    if summary:
        parts.append(f"summary={summary}")
    failure_reason = feedback.get("failure_reason")
    if failure_reason:
        parts.append(f"failure_reason={failure_reason}")
    next_hint = feedback.get("next_hint")
    if next_hint:
        parts.append(f"next_hint={next_hint}")
    return "; ".join(parts) or None
