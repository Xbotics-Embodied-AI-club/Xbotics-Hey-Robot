from __future__ import annotations

import numpy as np
import pytest

from hey_robot.persistence import FileRunStore
from hey_robot.protocol import Envelope
from hey_robot.robot_media import LocalMediaStore
from hey_robot.skills import SkillCommand, SkillEvent, SkillResult


def _command(*, arguments: dict | None = None) -> SkillCommand:
    return SkillCommand(
        envelope=Envelope(robot_id="mock0"),
        run_id="run-1",
        task_id="task-1",
        robot_id="mock0",
        name="inspect",
        arguments=arguments or {},
    )


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


def test_file_run_store_persists_idempotent_submission_receipt(tmp_path) -> None:
    store = FileRunStore(tmp_path / "runs")
    command = _command(arguments={"target": "desk"})

    store.record_submission(command)
    store.record_submission(command)

    assert store.submission(command.run_id) == command
    with pytest.raises(ValueError, match="different command"):
        store.record_submission(_command(arguments={"target": "shelf"}))


def test_file_run_store_preserves_and_repairs_corrupt_trailing_event(
    tmp_path, monkeypatch
) -> None:
    warnings: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        "hey_robot.persistence.run_store.logger.warning",
        lambda *args: warnings.append(args),
    )
    store = FileRunStore(tmp_path / "runs")
    accepted = SkillEvent(
        envelope=Envelope(robot_id="mock0"),
        run_id="run-1",
        sequence=1,
        name="inspect",
        phase="accepted",
        timestamp=1.0,
    )
    store.append_event(accepted)
    path = tmp_path / "runs" / "run-1" / "events.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"incomplete":\n\n')

    assert store.events("run-1") == (accepted,)
    assert warnings
    assert "corrupt trailing events.jsonl record" in str(warnings[0][0])

    running = SkillEvent(
        envelope=accepted.envelope,
        run_id="run-1",
        sequence=2,
        name="inspect",
        phase="running",
        timestamp=2.0,
    )
    store.append_event(running)

    assert store.events("run-1") == (accepted, running)


def test_file_run_store_rejects_corruption_before_last_record(tmp_path) -> None:
    store = FileRunStore(tmp_path / "runs")
    path = tmp_path / "runs" / "run-1" / "events.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"broken":\n{}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="corrupt at line 1"):
        store.events("run-1")


def test_file_run_store_externalizes_bounded_execution_trace(tmp_path) -> None:
    media = LocalMediaStore(tmp_path / "media")
    store = FileRunStore(tmp_path / "runs", artifact_store=media)
    terminal = SkillEvent(
        envelope=Envelope(robot_id="mock0"),
        run_id="run-1",
        sequence=1,
        name="manipulate",
        phase="completed",
        timestamp=2.0,
        result=SkillResult(
            True,
            "done",
            "completed",
            data={
                "vla_history": [{"values": list(range(128))}],
                "steps": [{"action": {"values": list(range(32))}}],
                "termination_reason": "model_done",
                "option_completed": True,
                "subgoal_succeeded": True,
                "before_frame_id": 7,
                "after_frame_id": 8,
                "steps_used": 1,
            },
        ),
    )

    persisted = store.append_event(terminal)

    assert persisted.result is not None
    assert persisted.result.data == {
        "termination_reason": "model_done",
        "option_completed": True,
        "subgoal_succeeded": True,
        "before_frame_id": 7,
        "after_frame_id": 8,
        "steps_used": 1,
        "steps_executed": 1,
        "execution_trace": persisted.result.artifacts[0].uri,
    }
    assert (
        media.load_json_artifact(persisted.result.artifacts[0]) == terminal.result.data
    )
    assert store.latest_event("run-1") == persisted
    assert store.recent(1) == (persisted,)
    assert store.append_event(terminal) == persisted


def test_file_run_store_externalizes_numpy_policy_output_as_typed_npz(tmp_path) -> None:
    media = LocalMediaStore(tmp_path / "media")
    store = FileRunStore(tmp_path / "runs", artifact_store=media)
    event = SkillEvent(
        envelope=Envelope(robot_id="mock0"),
        run_id="run-array",
        sequence=1,
        name="manipulate",
        phase="completed",
        timestamp=1.0,
        result=SkillResult(
            True,
            "done",
            "completed",
            data={"model_outputs": {"action": np.arange(64, dtype=np.float32)}},
        ),
    )

    persisted = store.append_event(event)

    assert persisted.result is not None
    artifact = persisted.result.artifacts[0]
    assert artifact.content_type == "application/x.numpy-npz"
    loaded = media.load_npz_artifact(artifact)
    np.testing.assert_array_equal(
        loaded["model_outputs"]["action"], np.arange(64, dtype=np.float32)
    )


def test_file_run_store_pins_observations_outside_transient_image_limit(
    tmp_path,
) -> None:
    media = LocalMediaStore(tmp_path / "media", max_items=2)
    store = FileRunStore(tmp_path / "runs", artifact_store=media)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    transient = media.put_image(image, robot_id="mock0", frame_id=1, camera="front")
    event = SkillEvent(
        envelope=Envelope(robot_id="mock0"),
        run_id="run-evidence",
        sequence=1,
        name="inspect",
        phase="completed",
        timestamp=1.0,
        result=SkillResult(
            True,
            "seen",
            "completed",
            observations=(transient,),
        ),
    )

    persisted = store.append_event(event)
    assert persisted.result is not None
    pinned = persisted.result.observations[0]
    assert pinned.uri.startswith("media://local/evidence/run-evidence/")

    for frame_id in range(2, 8):
        media.put_image(image, robot_id="mock0", frame_id=frame_id, camera="front")

    assert media.resolve_image(pinned).shape == (8, 8, 3)
    assert store.append_event(event) == persisted
