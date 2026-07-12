from __future__ import annotations

import pytest

from hey_robot.protocol import (
    Envelope,
    EvidenceFact,
    GoalCommand,
    SkillResult,
    SuccessCriterion,
)
from hey_robot.protocol.messages import from_payload, to_payload


def test_goal_command_round_trip_recovers_nested_tuple() -> None:
    command = GoalCommand(
        envelope=Envelope(robot_id="main"),
        command_id="command",
        action="create",
        objective="inspect",
        success_criteria=(
            SuccessCriterion(
                "criterion",
                "evidence_present",
                "room:kitchen",
                "observed",
                "scene",
                30.0,
            ),
        ),
    )
    assert from_payload(GoalCommand, to_payload(command)) == command


def test_new_messages_reject_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown fields"):
        from_payload(
            GoalCommand,
            {"envelope": {}, "command_id": "c", "action": "create", "unknown": True},
        )


def test_goal_contract_rejects_invalid_predicate_combination() -> None:
    with pytest.raises(ValueError, match="invalid for"):
        from_payload(
            GoalCommand,
            {
                "envelope": {"robot_id": "main"},
                "command_id": "c",
                "action": "create",
                "objective": "invalid",
                "success_criteria": [
                    {
                        "criterion_id": "c1",
                        "criterion_type": "robot_state",
                        "subject_id": "robot:main",
                        "predicate": "at",
                        "object_id": "room:lab",
                        "max_age_sec": 10,
                    }
                ],
            },
        )


def test_skill_result_rejects_evidence_from_another_skill() -> None:
    result = SkillResult(
        envelope=Envelope(robot_id="main"),
        skill_id="skill-a",
        status="completed",
        success=True,
        evidence=(
            EvidenceFact(
                "fact",
                "goal",
                "skill_result",
                "skill-b",
                1.0,
                1,
                "object:wand",
                "at",
                "fixture:dock",
            ),
        ),
    )
    with pytest.raises(ValueError, match="result skill_id"):
        from_payload(SkillResult, to_payload(result))
