from __future__ import annotations

import json

from hey_robot.agents.task_events import RobotTaskEvent, RobotTaskEventLog


def test_robot_task_event_from_dict_coerces_optional_fields() -> None:
    event = RobotTaskEvent.from_dict(
        {
            "episode_id": "ep1",
            "task_id": 123,
            "kind": "",
            "summary": None,
            "skill_id": 456,
            "frame_id": "7",
            "metadata": {"ok": True},
            "timestamp": "42.5",
        }
    )

    assert event.event_id.startswith("event_")
    assert event.episode_id == "ep1"
    assert event.task_id == "123"
    assert event.kind == "event"
    assert event.summary == ""
    assert event.skill_id == "456"
    assert event.frame_id == 7
    assert event.metadata == {"ok": True}
    assert event.timestamp == 42.5


def test_task_event_log_appends_filters_and_builds_prompt_context(tmp_path) -> None:
    log = RobotTaskEventLog(tmp_path)

    log.append(
        episode_id="ep1",
        task_id="task1",
        kind="task_started",
        summary="pick the cup",
    )
    log.append(
        episode_id="ep1",
        task_id="task1",
        kind="skill_bound",
        summary="move near table",
        skill_id="skill1",
        frame_id=3,
        metadata={"TaskAttempt": 0},
    )
    log.append(
        episode_id="ep1",
        task_id="task1",
        kind="execution_feedback",
        summary="cup reached",
        skill_id="skill1",
    )

    recent = log.recent("ep1", limit=2)
    assert [event.kind for event in recent] == ["skill_bound", "execution_feedback"]
    assert recent[0].metadata == {"TaskAttempt": 0}

    feedback = log.recent("ep1", kind="execution_feedback")
    assert len(feedback) == 1
    assert feedback[0].summary == "cup reached"

    context = log.prompt_context("ep1", limit=3)
    assert context is not None
    assert "Recent task events:" in context
    assert "- skill_bound (skill=skill1, frame=3): move near table" in context
    assert "- execution_feedback (skill=skill1): cup reached" in context


def test_task_event_log_returns_none_for_empty_prompt_context(tmp_path) -> None:
    log = RobotTaskEventLog(tmp_path)

    assert log.recent("missing") == []
    assert log.prompt_context("missing") is None


def test_task_event_log_trims_oldest_events_and_clamps_limit(tmp_path) -> None:
    log = RobotTaskEventLog(tmp_path, max_items_per_episode=2)

    for index in range(4):
        log.append(
            episode_id="ep1",
            task_id="task1",
            kind="step",
            summary=f"step {index}",
        )

    recent = log.recent("ep1", limit=10)
    assert [event.summary for event in recent] == ["step 2", "step 3"]

    clamped = log.recent("ep1", limit=0)
    assert [event.summary for event in clamped] == ["step 3"]


def test_task_event_log_skips_corrupted_lines_without_losing_valid_events(
    tmp_path,
) -> None:
    log = RobotTaskEventLog(tmp_path)
    first = log.append(
        episode_id="ep1", task_id="task1", kind="valid", summary="before"
    )

    path = log.root / "ep1.events.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write("not json\n")
        handle.write(json.dumps(["not", "an", "event"]) + "\n")
        handle.write(json.dumps({"kind": "missing episode"}) + "\n")
        handle.write(json.dumps({"episode_id": "ep1", "frame_id": "bad"}) + "\n")

    last = log.append(episode_id="ep1", task_id="task1", kind="valid", summary="after")

    recent = log.recent("ep1")
    assert [event.event_id for event in recent] == [first.event_id, last.event_id]
    assert [event.summary for event in recent] == ["before", "after"]


def test_task_event_log_sanitizes_episode_id_in_filename(tmp_path) -> None:
    log = RobotTaskEventLog(tmp_path)

    log.append(episode_id="web/user 1", task_id=None, kind="task_started")

    assert (log.root / "web_user_1.events.jsonl").exists()


def test_task_event_log_writes_utf8_json_lines(tmp_path) -> None:
    log = RobotTaskEventLog(tmp_path)
    summary = "\u524d\u65b9\u6709\u684c\u5b50\u548c\u7ea2\u8272\u65b9\u5757"

    log.append(
        episode_id="ep1",
        task_id="task1",
        kind="scene_observed",
        summary=summary,
        metadata={"note": "\u9760\u8fd1\u770b\u7ec6\u8282"},
    )

    path = log.root / "ep1.events.jsonl"
    raw = path.read_text(encoding="utf-8").strip()
    payload = json.loads(raw)

    assert payload["summary"] == summary
    assert payload["metadata"]["note"] == "\u9760\u8fd1\u770b\u7ec6\u8282"
    assert log.recent("ep1")[0].summary == summary
