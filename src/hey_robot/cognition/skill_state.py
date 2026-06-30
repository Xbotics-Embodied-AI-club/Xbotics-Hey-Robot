from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum

from hey_robot.protocol import SkillIntent, SkillResult


class SkillPhase(StrEnum):
    IDLE = "idle"
    ISSUED = "issued"
    ACCEPTED = "accepted"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    FEEDBACK_PENDING = "feedback_pending"
    CONFIRMED = "confirmed"


@dataclass
class SkillSnapshot:
    phase: SkillPhase = SkillPhase.IDLE
    skill_id: str = ""
    objective: str = ""
    issued_at: float | None = None
    updated_at: float | None = None
    feedback_mode: str = "status"
    feedback_summary: str | None = None
    error: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.phase in {
            SkillPhase.IDLE,
            SkillPhase.COMPLETED,
            SkillPhase.FAILED,
            SkillPhase.INTERRUPTED,
            SkillPhase.CONFIRMED,
        }

    @property
    def needs_feedback(self) -> bool:
        return self.phase == SkillPhase.COMPLETED and self.feedback_mode != "none"


class SkillStateMachine:
    def __init__(self) -> None:
        self.snapshot = SkillSnapshot()

    def reset(self) -> SkillSnapshot:
        self.snapshot = SkillSnapshot()
        return self.snapshot

    def submit(self, intent: SkillIntent) -> SkillSnapshot:
        objective = intent.objective.strip()
        if not objective:
            raise ValueError("skill objective must be non-empty")
        now = time.time()
        self.snapshot = SkillSnapshot(
            phase=SkillPhase.ISSUED,
            skill_id=intent.skill_id,
            objective=objective,
            issued_at=now,
            updated_at=now,
            feedback_mode=intent.feedback_mode,
        )
        return self.snapshot

    def observe_result(self, result: SkillResult) -> SkillSnapshot:
        if result.skill_id != self.snapshot.skill_id:
            return self.snapshot
        phase = _phase_for_status(result.status)
        self.snapshot.phase = phase
        self.snapshot.updated_at = time.time()
        self.snapshot.error = result.error
        return self.snapshot

    def mark_feedback_pending(self) -> SkillSnapshot:
        if not self.snapshot.skill_id:
            raise RuntimeError("no skill to evaluate")
        self.snapshot.phase = SkillPhase.FEEDBACK_PENDING
        self.snapshot.updated_at = time.time()
        return self.snapshot

    def mark_feedback_received(self, summary: str) -> SkillSnapshot:
        if not self.snapshot.skill_id:
            raise RuntimeError("no skill to evaluate")
        self.snapshot.phase = SkillPhase.CONFIRMED
        self.snapshot.feedback_summary = summary
        self.snapshot.updated_at = time.time()
        return self.snapshot


def _phase_for_status(status: str) -> SkillPhase:
    normalized = (status or "").lower()
    if normalized == "accepted":
        return SkillPhase.ACCEPTED
    if normalized == "executing":
        return SkillPhase.EXECUTING
    if normalized == "completed":
        return SkillPhase.COMPLETED
    if normalized == "failed":
        return SkillPhase.FAILED
    if normalized == "interrupted":
        return SkillPhase.INTERRUPTED
    return SkillPhase.ACCEPTED
