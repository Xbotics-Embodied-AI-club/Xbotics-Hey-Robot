from hey_robot.protocol import Envelope, SkillIntent
from hey_robot.skill_os.event_sink import _evidence_from_data


def _intent() -> SkillIntent:
    return SkillIntent(
        envelope=Envelope(robot_id="sim_robot"),
        skill_id="skill1",
        task_id="task1",
        intent_kind="observation",
        name="inspect_scene",
        arguments={"question": "is the wand at the dock?"},
        objective="inspect the dock",
    )


def test_skill_result_evidence_requires_explicit_typed_fact() -> None:
    facts = _evidence_from_data(
        _intent(),
        [
            {
                "subject_id": "object:wand",
                "predicate": "at",
                "object_id": "fixture:dock",
            },
            {"summary": "wand appears to be at dock"},
            {
                "subject_id": "object:wand",
                "predicate": "invented",
                "object_id": "fixture:dock",
            },
        ],
        7,
    )

    assert len(facts) == 1
    assert facts[0].task_id == "task1"
    assert facts[0].source_id == "skill1"
    assert facts[0].frame_id == 7
    assert facts[0].subject_id == "object:wand"
    assert facts[0].predicate == "at"
    assert facts[0].object_id == "fixture:dock"


def test_skill_result_summary_cannot_become_evidence() -> None:
    assert _evidence_from_data(_intent(), "wand is at dock", 7) == ()
