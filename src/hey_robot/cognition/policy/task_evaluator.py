"""Deterministic all-of evaluator; it never reads model text or summaries."""

from __future__ import annotations

import time

from hey_robot.cognition.task.contract import TaskContract
from hey_robot.protocol import EvidenceFact, TaskEvaluationPayload


class TaskEvaluator:
    def evaluate(
        self,
        contract: TaskContract,
        evidence: tuple[EvidenceFact, ...],
        *,
        now: float | None = None,
    ) -> TaskEvaluationPayload:
        current_time = time.time() if now is None else now
        matched: list[str] = []
        missing: list[str] = []
        for criterion in contract.success_criteria:
            fact = next(
                (
                    item
                    for item in evidence
                    if (
                        item.goal_id == contract.goal_id
                        and item.subject_id == criterion.subject_id
                        and item.predicate == criterion.predicate
                        and item.object_id == criterion.object_id
                        and current_time - item.observed_at <= criterion.max_age_sec
                    )
                ),
                None,
            )
            if fact is None:
                missing.append(criterion.criterion_id)
            else:
                matched.append(fact.evidence_id)
        if not missing:
            return TaskEvaluationPayload(
                "satisfied", "all contract criteria have fresh evidence", tuple(matched)
            )
        return TaskEvaluationPayload(
            "inconclusive",
            "required evidence is missing",
            tuple(matched),
            tuple(missing),
        )
