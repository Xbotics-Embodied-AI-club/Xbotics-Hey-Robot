from __future__ import annotations

from hey_robot.persistence import FileRunStore
from hey_robot.protocol import Envelope
from hey_robot.skills import SkillEvent, SkillResult


def test_file_run_store_appends_events_and_writes_terminal_result(tmp_path) -> None:
    store = FileRunStore(tmp_path / "runs")
    accepted = SkillEvent(
        envelope=Envelope(robot_id="mock0"),
        run_id="run-1",
        sequence=1,
        name="inspect",
        phase="accepted",
        timestamp=1.0,
    )
    terminal = SkillEvent(
        envelope=Envelope(robot_id="mock0"),
        run_id="run-1",
        sequence=2,
        name="inspect",
        phase="completed",
        timestamp=2.0,
        result=SkillResult(True, "done", "completed", data={"frame_id": 7}),
    )

    store.append_event(accepted)
    store.append_event(terminal)

    assert [event.sequence for event in store.events("run-1")] == [1, 2]
    assert store.latest_event("run-1") == terminal
    assert store.result("run-1") == terminal.result
