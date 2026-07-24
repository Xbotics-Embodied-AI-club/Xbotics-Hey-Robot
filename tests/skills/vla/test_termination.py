from __future__ import annotations

from hey_robot.skills.vla.termination import CompositeTermination, VLAOptionState


def _state(
    stage="after_model",
    *,
    model_data=None,
    environment_data=None,
    action_count=0,
    step_index=0,
    max_steps=2,
) -> VLAOptionState:
    return VLAOptionState(
        stage,
        step_index,
        max_steps,
        model_data or {},
        environment_data or {},
        action_count,
    )


def test_environment_done_has_priority_and_evidence_of_subgoal_success() -> None:
    decision = CompositeTermination().evaluate(
        _state(
            "after_actions",
            model_data={"done": True},
            environment_data={"environment_done": True},
            action_count=1,
        )
    )

    assert decision.terminate is True
    assert decision.reason == "environment_done"
    assert decision.subgoal_succeeded is True


def test_model_done_waits_until_returned_actions_have_executed() -> None:
    policy = CompositeTermination()

    before_action = policy.evaluate(
        _state("after_model", model_data={"done": True}, action_count=1)
    )
    after_action = policy.evaluate(
        _state("after_actions", model_data={"done": True}, action_count=1)
    )

    assert before_action.terminate is False
    assert after_action.reason == "model_done"


def test_budget_completion_does_not_claim_subgoal_success() -> None:
    decision = CompositeTermination().evaluate(
        _state("budget", action_count=1, step_index=1, max_steps=2)
    )

    assert decision.reason == "max_steps"
    assert decision.subgoal_succeeded is None


def test_no_action_is_distinct_from_model_success() -> None:
    decision = CompositeTermination().evaluate(_state("after_model"))

    assert decision.reason == "no_action"
    assert decision.subgoal_succeeded is None
