from hey_robot.cognition.autonomous.continuation_policy import decide_continuation
from hey_robot.cognition.autonomous.progress import assess_progress


def _action(name: str = "inspect_scene") -> dict:
    return {
        "payload": {
            "intent_kind": "observation",
            "name": name,
            "arguments": {"question": "where is the cup?"},
        }
    }


def test_progress_monitor_detects_repeated_action_without_evidence() -> None:
    assessment = assess_progress(
        goal_status="active", actions=[_action(), _action()], evidence=[]
    )
    assert assessment.state == "no_progress"
    assert assessment.repeated_action_count == 2


def test_progress_monitor_keeps_evidenced_repeat_as_progressing() -> None:
    assessment = assess_progress(
        goal_status="active",
        actions=[_action(), _action()],
        evidence=[{"evidence_id": "e1"}],
    )
    assert assessment.state == "progressing"


def test_continuation_policy_never_schedules_when_waiting_or_gate_blocked() -> None:
    progress = assess_progress(goal_status="active", actions=[], evidence=[])
    waiting = decide_continuation(
        goal_status="waiting_condition",
        progress=progress,
        budget_allowed=True,
        gate_ready=True,
        has_wake_trigger=True,
    )
    blocked = decide_continuation(
        goal_status="active",
        progress=progress,
        budget_allowed=True,
        gate_ready=False,
        has_wake_trigger=True,
    )
    assert waiting.decision == "wait"
    assert blocked.decision == "block"


def test_continuation_policy_marks_no_progress_for_review() -> None:
    progress = assess_progress(
        goal_status="active", actions=[_action(), _action()], evidence=[]
    )
    decision = decide_continuation(
        goal_status="active",
        progress=progress,
        budget_allowed=True,
        gate_ready=True,
        has_wake_trigger=True,
    )
    assert decision.decision == "needs_review"
