from hey_robot.skill_os.termination import (
    FixedHorizonTerminationEvaluator,
    TerminationState,
)


def test_fixed_horizon_fails_closed_without_claiming_success() -> None:
    evaluator = FixedHorizonTerminationEvaluator(2)

    first = evaluator.evaluate(
        steps_executed=1,
        policy_result={"valid": True},
        action_result={"success": True, "done": False},
    )
    boundary = evaluator.evaluate(
        steps_executed=2,
        policy_result={"valid": True},
        action_result={"success": True, "done": False},
    )

    assert first.state is TerminationState.CONTINUE
    assert boundary.state is TerminationState.UNKNOWN
    assert boundary.reason == "fixed_horizon_reached"


def test_fixed_horizon_keeps_policy_and_runtime_terminal_states_distinct() -> None:
    evaluator = FixedHorizonTerminationEvaluator(50)

    policy_done = evaluator.evaluate(
        steps_executed=1,
        policy_result={"valid": True, "done": True},
        action_result={"success": True, "done": False},
    )
    episode_done = evaluator.evaluate(
        steps_executed=1,
        policy_result={"valid": True},
        action_result={"success": True, "done": True},
    )

    assert policy_done.state is TerminationState.SUCCESS
    assert episode_done.state is TerminationState.UNKNOWN
