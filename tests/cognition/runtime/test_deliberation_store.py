from pathlib import Path

from hey_robot.cognition.runtime.deliberation_store import DeliberationStore
from hey_robot.protocol import BudgetState, DeliberationRequest, Envelope, GoalSnapshot


def test_incomplete_receipt_keeps_full_request_for_interruption(tmp_path: Path) -> None:
    store = DeliberationStore(tmp_path / "agent.sqlite3")
    request = DeliberationRequest(
        Envelope(robot_id="main"),
        "d",
        "trigger",
        GoalSnapshot("goal", 0, "task", "contract", "hash", "inspect", (), "active"),
        None,
        None,
        (),
        (),
        None,
        BudgetState(0, 0, 0, None),
    )
    assert store.schedule(
        deliberation_id="d",
        request_hash="h",
        goal_id="goal",
        task_id="task",
        request=request,
    )
    interrupted = store.interrupt_incomplete()
    assert interrupted == [request]
