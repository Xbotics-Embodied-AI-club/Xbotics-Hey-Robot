from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

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


@dataclass
class LongTermMemoryRecord:
    kind: MemoryKind
    key: str
    summary: str
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LongTermMemoryRecord:
        return cls(
            kind=str(data.get("kind") or "event"),  # type: ignore[arg-type]
            key=str(data.get("key") or ""),
            summary=str(data.get("summary") or ""),
            confidence=float(data.get("confidence", 1.0)),
            metadata=dict(data.get("metadata") or {}),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
        )


EntityMemoryRecord = LongTermMemoryRecord
PlaceMemoryRecord = LongTermMemoryRecord
TaskMemoryRecord = LongTermMemoryRecord
SkillExperienceRecord = LongTermMemoryRecord
PreferenceMemoryRecord = LongTermMemoryRecord
SceneAnchorMemoryRecord = LongTermMemoryRecord
TaskLessonMemoryRecord = LongTermMemoryRecord


class LongTermMemoryStore:
    """Append-only JSONL memory with simple relevance search.

    This is intentionally small: it gives the long-horizon loop persistent
    entities, places, task outcomes, and skill experience without adding a
    database dependency to the first validation pass.
    """

    def __init__(self, path: str | Path = "runtime/memory/long_term.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def remember(
        self,
        *,
        kind: MemoryKind,
        key: str,
        summary: str,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> LongTermMemoryRecord:
        record = LongTermMemoryRecord(
            kind=kind,
            key=(key or kind).strip(),
            summary=(summary or "").strip(),
            confidence=float(confidence),
            metadata=dict(metadata or {}),
        )
        if not record.summary:
            raise ValueError("memory summary must not be empty")
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        return record

    def remember_entity(
        self,
        *,
        name: str,
        summary: str,
        entity_type: str = "object",
        location: str | None = None,
        attributes: dict[str, Any] | None = None,
        confidence: float = 1.0,
        frame_id: int | None = None,
        robot_id: str | None = None,
    ) -> LongTermMemoryRecord:
        return self.remember(
            kind="entity",
            key=name,
            summary=summary,
            confidence=confidence,
            metadata={
                "entity_type": entity_type,
                "location": location,
                "attributes": dict(attributes or {}),
                "last_seen_frame_id": frame_id,
                "robot_id": robot_id,
            },
        )

    def remember_place(
        self,
        *,
        name: str,
        description: str,
        pose: dict[str, Any] | None = None,
        confidence: float = 1.0,
        robot_id: str | None = None,
    ) -> LongTermMemoryRecord:
        return self.remember(
            kind="place",
            key=name,
            summary=description,
            confidence=confidence,
            metadata={"pose": dict(pose or {}), "robot_id": robot_id},
        )

    def remember_task_result(
        self,
        *,
        task_id: str,
        root_goal: str,
        status: str,
        summary: str,
        failure_reason: str | None = None,
    ) -> LongTermMemoryRecord:
        return self.remember(
            kind="task_result",
            key=task_id,
            summary=summary,
            confidence=1.0 if status == "completed" else 0.5,
            metadata={
                "root_goal": root_goal,
                "status": status,
                "failure_reason": failure_reason,
            },
        )

    def remember_skill_experience(
        self,
        *,
        skill_name: str,
        arguments: dict[str, Any] | None = None,
        context_summary: str = "",
        success: bool,
        summary: str,
        failure_mode: str | None = None,
        recovery_hint: str | None = None,
        verification_summary: str | None = None,
        duration_sec: float | None = None,
    ) -> LongTermMemoryRecord:
        return self.remember(
            kind="skill_experience",
            key=skill_name or "skill_experience",
            summary=summary or ("skill succeeded" if success else "skill failed"),
            confidence=1.0 if success else 0.4,
            metadata={
                "skill_name": skill_name,
                "arguments": dict(arguments or {}),
                "context_summary": context_summary,
                "success": bool(success),
                "failure_mode": failure_mode,
                "recovery_hint": recovery_hint,
                "verification_summary": verification_summary,
                "duration_sec": duration_sec,
            },
        )

    def remember_user_preference(
        self,
        *,
        name: str,
        value: str,
        summary: str,
        confidence: float = 1.0,
        source: str = "agent_tool",
    ) -> LongTermMemoryRecord:
        return self.remember(
            kind="user_preference",
            key=name or "user_preference",
            summary=summary,
            confidence=confidence,
            metadata={
                "name": name,
                "value": value,
                "source": source,
            },
        )

    def record_scene_anchor(
        self,
        *,
        name: str,
        location: str,
        summary: str,
        entity_type: str = "object",
        confidence: float = 1.0,
        frame_id: int | None = None,
        robot_id: str | None = None,
    ) -> LongTermMemoryRecord:
        return self.remember(
            kind="scene_anchor",
            key=name or "scene_anchor",
            summary=summary,
            confidence=confidence,
            metadata={
                "name": name,
                "location": location,
                "entity_type": entity_type,
                "frame_id": frame_id,
                "robot_id": robot_id,
            },
        )

    def remember_task_lesson(
        self,
        *,
        key: str,
        summary: str,
        task: str = "",
        success: bool | None = None,
        failure_mode: str | None = None,
        recovery_hint: str | None = None,
        skill_name: str | None = None,
    ) -> LongTermMemoryRecord:
        confidence = 1.0 if success is True else 0.7 if success is None else 0.5
        return self.remember(
            kind="task_lesson",
            key=key or skill_name or "task_lesson",
            summary=summary,
            confidence=confidence,
            metadata={
                "task": task,
                "success": success,
                "failure_mode": failure_mode,
                "recovery_hint": recovery_hint,
                "skill_name": skill_name,
            },
        )

    def query(
        self, text: str = "", *, kind: MemoryKind | None = None, limit: int = 8
    ) -> list[LongTermMemoryRecord]:
        terms = _terms(text)
        scored: list[tuple[float, LongTermMemoryRecord]] = []
        for record in self._read_all():
            if kind is not None and record.kind != kind:
                continue
            score = _memory_score(record, terms=terms, query=text)
            if score > 0:
                scored.append((score, record))
        ranked = _dedupe_scored_records(scored)
        ranked.sort(
            key=lambda item: (item[0], item[1].confidence, item[1].updated_at),
            reverse=True,
        )
        return [record for _, record in ranked[: max(1, int(limit))]]

    def prompt_context(self, text: str = "", *, limit: int = 8) -> str:
        preferences = self.query("", kind="user_preference", limit=min(limit, 3))
        anchors = self.query(text, kind="scene_anchor", limit=min(limit, 3))
        lessons = self.query(text, kind="task_lesson", limit=min(limit, 3))
        task_results = self.query(text, kind="task_result", limit=min(limit, 2))
        skill_experiences = self.query(
            text, kind="skill_experience", limit=min(limit, 2)
        )
        records = [
            record
            for record in self.query(text, limit=limit)
            if record.kind
            not in {
                "user_preference",
                "scene_anchor",
                "task_lesson",
                "task_result",
                "skill_experience",
            }
        ]
        if not any(
            (preferences, anchors, lessons, task_results, skill_experiences, records)
        ):
            return ""
        lines = ["Long-term memory:"]
        continuity = _prompt_continuity_cues(
            preferences, task_results, skill_experiences
        )
        if continuity:
            lines.append("- Continuity cues:")
            lines.extend(f"  - {item}" for item in continuity)
        if preferences:
            lines.append("- User preferences:")
            lines.extend(_prompt_lines_for_preferences(preferences))
        if anchors:
            lines.append("- Scene anchors:")
            lines.extend(_prompt_lines_for_anchors(anchors))
        if lessons:
            lines.append("- Task lessons:")
            lines.extend(_prompt_lines_for_lessons(lessons))
        if task_results:
            lines.append("- Related task results:")
            lines.extend(_prompt_lines_for_task_results(task_results))
        if skill_experiences:
            lines.append("- Related skill experience:")
            lines.extend(_prompt_lines_for_skill_experiences(skill_experiences))
        if records:
            lines.append("- Other relevant memory:")
            lines.extend(
                f"  - {record.kind}:{record.key}: {record.summary}"
                for record in records
            )
        guidance = _prompt_usage_guidance(preferences, anchors, lessons)
        if guidance:
            lines.append("- How to use this memory:")
            lines.extend(f"  - {item}" for item in guidance)
        return "\n".join(lines)

    def _read_all(self) -> list[LongTermMemoryRecord]:
        if not self.path.exists():
            return []
        records: list[LongTermMemoryRecord] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    records.append(LongTermMemoryRecord.from_dict(data))
        return records


def _memory_score(
    record: LongTermMemoryRecord, *, terms: set[str], query: str
) -> float:
    haystack_text = _searchable_text(record)
    haystack = _terms(haystack_text)
    if terms:
        overlap = terms & haystack
        if not overlap:
            return 0.0
        score = float(len(overlap) * 10)
    else:
        score = 1.0
    lowered_query = str(query).lower()
    lowered_key = record.key.lower()
    if lowered_key and lowered_key in lowered_query:
        score += 8.0
    if lowered_key and lowered_key in haystack_text.lower():
        score += 1.0
    score += max(0.0, min(1.0, record.confidence)) * 2.0
    if record.kind == "skill_experience":
        score += 1.5 if _skill_success(record) is True else 0.0
    if record.kind == "task_result":
        score += 1.5 if record.metadata.get("status") == "completed" else 0.0
    return score


def _dedupe_scored_records(
    scored: list[tuple[float, LongTermMemoryRecord]],
) -> list[tuple[float, LongTermMemoryRecord]]:
    best_by_key: dict[tuple[str, str], tuple[float, LongTermMemoryRecord]] = {}
    passthrough: list[tuple[float, LongTermMemoryRecord]] = []
    for score, record in scored:
        group_key = _dedupe_key(record)
        if group_key is None:
            passthrough.append((score, record))
            continue
        current = best_by_key.get(group_key)
        if current is None or _dedupe_rank(score, record) >= _dedupe_rank(
            current[0], current[1]
        ):
            best_by_key[group_key] = (score, record)
    return [*passthrough, *best_by_key.values()]


def _dedupe_key(record: LongTermMemoryRecord) -> tuple[str, str] | None:
    key = record.key.lower().strip()
    if record.kind in {"entity", "place", "scene_anchor", "user_preference"}:
        return (record.kind, key)
    if record.kind == "skill_experience":
        skill_name = (
            str(record.metadata.get("skill_name") or key or "skill_experience")
            .lower()
            .strip()
        )
        return (
            "skill_experience",
            f"{skill_name}:{_skill_arguments_signature(record.metadata)}",
        )
    if record.kind == "task_result":
        root_goal = str(record.metadata.get("root_goal") or "").strip().lower()
        return ("task_result", root_goal or key)
    if record.kind == "task_lesson":
        task = str(record.metadata.get("task") or "").strip().lower()
        return ("task_lesson", task or key)
    return None


def _dedupe_rank(
    score: float, record: LongTermMemoryRecord
) -> tuple[float, float, float, float]:
    status_bonus = 0.0
    if record.kind == "skill_experience" and _skill_success(record) is True:
        status_bonus = 1.0
    if record.kind == "task_result" and record.metadata.get("status") == "completed":
        status_bonus = 1.0
    return (record.updated_at, status_bonus, record.confidence, score)


def _searchable_text(record: LongTermMemoryRecord) -> str:
    metadata = record.metadata
    if record.kind in {
        "entity",
        "place",
        "scene_anchor",
        "user_preference",
        "task_lesson",
    }:
        metadata_text = json.dumps(metadata, ensure_ascii=False)
    elif record.kind == "skill_experience":
        fields = {
            "skill_name": metadata.get("skill_name"),
            "arguments": _skill_arguments(metadata),
            "success": _skill_success(record),
            "failure_mode": metadata.get("failure_mode"),
            "recovery_hint": metadata.get("recovery_hint"),
            "verification_summary": metadata.get("verification_summary"),
        }
        metadata_text = json.dumps(fields, ensure_ascii=False)
    elif record.kind == "task_result":
        fields = {
            "root_goal": metadata.get("root_goal"),
            "status": metadata.get("status"),
            "failure_reason": metadata.get("failure_reason"),
        }
        metadata_text = json.dumps(fields, ensure_ascii=False)
    else:
        metadata_text = json.dumps(metadata, ensure_ascii=False)
    return f"{record.key} {record.summary} {metadata_text}"


def _skill_success(record: LongTermMemoryRecord) -> bool | None:
    metadata = record.metadata
    if metadata.get("tool") != "request_capability" and isinstance(
        metadata.get("success"), bool
    ):
        return bool(metadata["success"])
    summary = record.summary.lower()
    if any(
        marker in summary
        for marker in (
            "outcome: failed",
            "outcome: interrupted",
            "failure_reason",
            "does not support",
            "keyerror",
            "toolunavailable",
        )
    ):
        return False
    if any(
        marker in summary
        for marker in ("outcome: confirmed", "outcome: skipped", " completed")
    ):
        return True
    if (
        isinstance(metadata.get("success"), bool)
        and metadata.get("tool") != "request_capability"
    ):
        return bool(metadata["success"])
    return None


def _skill_arguments(metadata: dict[str, Any]) -> dict[str, Any]:
    arguments = metadata.get("arguments")
    if isinstance(arguments, dict) and isinstance(arguments.get("arguments"), dict):
        return dict(arguments["arguments"])
    if isinstance(arguments, dict):
        return dict(arguments)
    return {}


def _skill_arguments_signature(metadata: dict[str, Any]) -> str:
    arguments = _skill_arguments(metadata)
    if not arguments:
        return ""
    return json.dumps(
        arguments, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


def _terms(text: str) -> set[str]:
    terms: set[str] = set()
    ascii_buffer: list[str] = []
    cjk_buffer: list[str] = []

    def flush_ascii() -> None:
        if ascii_buffer:
            terms.add("".join(ascii_buffer))
            ascii_buffer.clear()

    def flush_cjk() -> None:
        if not cjk_buffer:
            return
        joined = "".join(cjk_buffer)
        terms.add(joined)
        terms.update(cjk_buffer)
        terms.update(
            joined[index : index + 2] for index in range(max(0, len(joined) - 1))
        )
        cjk_buffer.clear()

    for ch in str(text):
        if _is_cjk(ch):
            flush_ascii()
            cjk_buffer.append(ch)
            continue
        flush_cjk()
        if ch.isalnum():
            ascii_buffer.append(ch.lower())
        else:
            flush_ascii()
    flush_ascii()
    flush_cjk()
    return {term for term in terms if term}


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _prompt_lines_for_preferences(records: list[LongTermMemoryRecord]) -> list[str]:
    lines: list[str] = []
    for record in records:
        value = record.metadata.get("value")
        if value:
            lines.append(f"  - {record.key}={value}: {record.summary}")
        else:
            lines.append(f"  - {record.key}: {record.summary}")
    return lines


def _prompt_lines_for_anchors(records: list[LongTermMemoryRecord]) -> list[str]:
    lines: list[str] = []
    for record in records:
        location = record.metadata.get("location")
        if location:
            lines.append(f"  - {record.key} @ {location}: {record.summary}")
        else:
            lines.append(f"  - {record.key}: {record.summary}")
    return lines


def _prompt_lines_for_lessons(records: list[LongTermMemoryRecord]) -> list[str]:
    lines: list[str] = []
    for record in records:
        failure_mode = record.metadata.get("failure_mode")
        recovery_hint = record.metadata.get("recovery_hint")
        suffix_parts = []
        if failure_mode:
            suffix_parts.append(f"failure_mode={failure_mode}")
        if recovery_hint:
            suffix_parts.append(f"recovery_hint={recovery_hint}")
        suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""
        lines.append(f"  - {record.key}: {record.summary}{suffix}")
    return lines


def _prompt_lines_for_task_results(records: list[LongTermMemoryRecord]) -> list[str]:
    lines: list[str] = []
    for record in records:
        status = str(record.metadata.get("status") or "").strip()
        root_goal = str(record.metadata.get("root_goal") or "").strip()
        prefix = root_goal or record.key
        if status:
            lines.append(f"  - {prefix} [{status}]: {record.summary}")
        else:
            lines.append(f"  - {prefix}: {record.summary}")
    return lines


def _prompt_lines_for_skill_experiences(
    records: list[LongTermMemoryRecord],
) -> list[str]:
    lines: list[str] = []
    for record in records:
        skill_name = str(record.metadata.get("skill_name") or record.key).strip()
        success = _skill_success(record)
        outcome = (
            "success"
            if success is True
            else "failure"
            if success is False
            else "unknown"
        )
        recovery_hint = str(record.metadata.get("recovery_hint") or "").strip()
        suffix = f" recovery_hint={recovery_hint}" if recovery_hint else ""
        lines.append(f"  - {skill_name} [{outcome}]: {record.summary}{suffix}")
    return lines


def _prompt_usage_guidance(
    preferences: list[LongTermMemoryRecord],
    anchors: list[LongTermMemoryRecord],
    lessons: list[LongTermMemoryRecord],
) -> list[str]:
    guidance: list[str] = []
    for record in preferences:
        name = str(record.metadata.get("name") or record.key).strip().lower()
        value = str(record.metadata.get("value") or "").strip()
        if name in {"response_language", "language"} and value:
            guidance.append(
                f"Reply in {value} unless the user explicitly asks for another language."
            )
            continue
        if value:
            guidance.append(
                f"Respect the user's stated preference {record.key}={value} when it does not conflict with safety."
            )
        else:
            guidance.append(
                f"Respect the user's stated preference about {record.key} when it does not conflict with safety."
            )
    if anchors:
        guidance.append(
            "Treat scene anchors as useful priors, but verify the current "
            "location with perception before moving or grasping."
        )
    for record in lessons[:2]:
        recovery_hint = str(record.metadata.get("recovery_hint") or "").strip()
        if recovery_hint:
            guidance.append(
                f"If a similar failure pattern appears again, prefer {recovery_hint} before retrying."
            )
        else:
            guidance.append(
                f"Use the task lesson '{record.summary}' to avoid repeating the same mistake."
            )
    return guidance


def _prompt_continuity_cues(
    preferences: list[LongTermMemoryRecord],
    task_results: list[LongTermMemoryRecord],
    skill_experiences: list[LongTermMemoryRecord],
) -> list[str]:
    cues: list[str] = []
    for record in preferences[:1]:
        name = str(record.metadata.get("name") or record.key).strip()
        value = str(record.metadata.get("value") or "").strip()
        if value:
            cues.append(f"Known user preference: {name}={value}.")
    for record in task_results[:1]:
        root_goal = str(record.metadata.get("root_goal") or record.key).strip()
        status = str(record.metadata.get("status") or "unknown").strip()
        cues.append(f"Recent related task: {root_goal} ended with status={status}.")
    for record in skill_experiences[:1]:
        skill_name = str(record.metadata.get("skill_name") or record.key).strip()
        success = _skill_success(record)
        if success is True:
            cues.append(
                f"Recent related skill experience: {skill_name} previously worked in a similar context."
            )
        elif success is False:
            recovery_hint = str(record.metadata.get("recovery_hint") or "").strip()
            if recovery_hint:
                cues.append(
                    f"Recent related skill failure: {skill_name} previously needed {recovery_hint} before retry."
                )
            else:
                cues.append(
                    f"Recent related skill failure: {skill_name} previously failed in a similar context."
                )
    return cues
