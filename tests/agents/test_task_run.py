import json

from hey_robot.agents.task_run import TaskRun, TaskRunStore


def test_task_run_store_tracks_loop_attempts(tmp_path):
    store = TaskRunStore(tmp_path)

    task = store.ensure_active(
        episode_id="ep1",
        task="pick up the bottle",
        agent_id="agent1",
        robot_id="robot1",
    )
    assert task.root_task == "pick up the bottle"
    assert task.status == "active"
    assert task.attempts == []

    bound = store.bind_skill("ep1", "skill1", "look at the table")
    assert bound is not None
    assert bound.status == "executing"
    assert bound.active_attempt_id == bound.attempts[-1].attempt_id
    assert bound.attempts[-1].text == "look at the table"

    feedback = store.mark_execution_feedback(
        "ep1", skill_id="skill1", success=False, summary="bottle not found"
    )
    assert feedback is not None
    assert feedback.status == "recovering"
    assert feedback.last_step_success is False
    assert feedback.failure_reason == "bottle not found"

    recovery = store.set_recovery(
        "ep1", strategy="inspect_and_continue", summary="get a fresh observation"
    )
    assert recovery is not None
    assert recovery.recovery is not None
    assert recovery.recovery["strategy"] == "inspect_and_continue"

    rebound = store.bind_skill("ep1", "skill2", "inspect current scene")
    assert rebound is not None
    assert len(rebound.attempts) == 2
    completed = store.mark_task_reported("ep1", success=True, summary="task done")
    assert completed is not None
    assert completed.status == "completed"
    assert completed.task_success is True
    assert store.load_active("ep1") is None


def test_task_run_store_supersedes_changed_task(tmp_path):
    store = TaskRunStore(tmp_path)
    first = store.ensure_active(
        episode_id="ep1", task="clear the desk", agent_id=None, robot_id=None
    )

    second = store.ensure_active(
        episode_id="ep1", task="bring me the bottle", agent_id=None, robot_id=None
    )

    assert second.task_id != first.task_id
    tasks = store.list_for_episode("ep1")
    assert {task.status for task in tasks} == {"active", "cancelled"}
    cancelled = next(task for task in tasks if task.status == "cancelled")
    assert cancelled.metadata["superseded_by"] == "bring me the bottle"


def test_task_run_store_reuses_paused_or_recovering_task_for_same_request(tmp_path):
    store = TaskRunStore(tmp_path)
    first = store.ensure_active(
        episode_id="ep1", task="clear the desk", agent_id="agent1", robot_id="robot1"
    )

    paused = store.pause("ep1", reason="operator requested pause", operator="alice")
    assert paused is not None
    assert paused.status == "paused"
    assert paused.paused_reason == "operator requested pause"

    resumed_by_new_turn = store.ensure_active(
        episode_id="ep1",
        task="clear the desk",
        agent_id="agent1",
        robot_id="robot1",
    )
    assert resumed_by_new_turn.task_id == first.task_id
    assert resumed_by_new_turn.status == "active"
    assert resumed_by_new_turn.paused_reason is None

    failed = store.mark_operator_feedback(
        "ep1", success=False, summary="trash bag not visible", operator="alice"
    )
    assert failed is not None
    assert failed.status == "recovering"
    assert failed.failure_reason == "trash bag not visible"

    resumed_after_recovery = store.ensure_active(
        episode_id="ep1",
        task="clear the desk",
        agent_id="agent1",
        robot_id="robot1",
    )
    assert resumed_after_recovery.task_id == first.task_id
    assert resumed_after_recovery.status == "active"
    assert resumed_after_recovery.failure_reason is None


def test_task_run_store_persists_scene_watchdog_and_operator_transitions(tmp_path):
    store = TaskRunStore(tmp_path)
    store.ensure_active(
        episode_id="ep1", task="bring the bottle", agent_id=None, robot_id="robot1"
    )

    observed = store.record_scene_memory(
        "ep1",
        summary="Bottle is on the front table.",
        frame_id=7,
        metadata={"source": "camera"},
    )
    assert observed is not None
    assert observed.metadata["latest_scene_event"]["frame_id"] == 7

    healthy = store.update_watchdog(
        "ep1", health="ok", summary="skill is making progress"
    )
    assert healthy is not None
    assert healthy.status == "active"
    assert healthy.watchdog["health"] == "ok"

    blocked = store.update_watchdog(
        "ep1", health="blocked", summary="gripper cannot reach bottle"
    )
    assert blocked is not None
    assert blocked.status == "recovering"
    assert blocked.recovery_count == 1
    assert blocked.failure_reason == "gripper cannot reach bottle"

    resumed = store.resume("ep1", operator="alice")
    assert resumed is not None
    assert resumed.status == "active"
    assert resumed.operator_notes[-1]["action"] == "resume"

    aborted = store.abort("ep1", reason="user cancelled", operator="alice")
    assert aborted is not None
    assert aborted.status == "cancelled"
    assert aborted.finished_at is not None
    assert store.load_active("ep1") is None


def test_task_run_store_deduplicates_unchanged_watchdog_events(tmp_path):
    store = TaskRunStore(tmp_path)
    store.ensure_active(
        episode_id="ep1", task="bring the bottle", agent_id=None, robot_id="robot1"
    )

    first = store.update_watchdog(
        "ep1",
        health="blocked",
        summary="gripper cannot reach bottle",
        metadata={"active_skill_id": "cmd1"},
    )
    second = store.update_watchdog(
        "ep1",
        health="blocked",
        summary="gripper cannot reach bottle",
        metadata={"active_skill_id": "cmd1"},
    )
    events = store.events.recent("ep1", limit=10, kind="watchdog")

    assert first is not None
    assert second is not None
    assert len(events) == 1
    assert second.watchdog["health"] == "blocked"
    assert second.watchdog["summary"] == "gripper cannot reach bottle"


def test_task_run_store_appends_watchdog_event_when_summary_changes(tmp_path):
    store = TaskRunStore(tmp_path)
    store.ensure_active(
        episode_id="ep1", task="bring the bottle", agent_id=None, robot_id="robot1"
    )

    store.update_watchdog(
        "ep1",
        health="blocked",
        summary="gripper cannot reach bottle",
        metadata={"active_skill_id": "cmd1"},
    )
    updated = store.update_watchdog(
        "ep1",
        health="blocked",
        summary="target slipped, re-evaluate grasp path",
        metadata={"active_skill_id": "cmd1"},
    )
    events = store.events.recent("ep1", limit=10, kind="watchdog")

    assert updated is not None
    assert len(events) == 2
    assert events[-1].summary == "target slipped, re-evaluate grasp path"


def test_task_run_store_deduplicates_unchanged_recovery_events(tmp_path):
    store = TaskRunStore(tmp_path)
    store.ensure_active(
        episode_id="ep1", task="bring the bottle", agent_id=None, robot_id="robot1"
    )

    first = store.set_recovery(
        "ep1",
        strategy="continue_from_observation",
        summary="gripper cannot reach bottle",
        metadata={"health": "blocked"},
    )
    second = store.set_recovery(
        "ep1",
        strategy="continue_from_observation",
        summary="gripper cannot reach bottle",
        metadata={"health": "blocked"},
    )
    events = store.events.recent("ep1", limit=10, kind="recovery_selected")

    assert first is not None
    assert second is not None
    assert len(events) == 1
    assert second.recovery is not None
    assert second.recovery["strategy"] == "continue_from_observation"


def test_task_run_store_appends_recovery_event_when_strategy_changes(tmp_path):
    store = TaskRunStore(tmp_path)
    store.ensure_active(
        episode_id="ep1", task="bring the bottle", agent_id=None, robot_id="robot1"
    )

    store.set_recovery(
        "ep1",
        strategy="continue_from_observation",
        summary="gripper cannot reach bottle",
        metadata={"health": "blocked"},
    )
    updated = store.set_recovery(
        "ep1",
        strategy="pause_for_operator",
        summary="gripper cannot reach bottle",
        metadata={"health": "stale"},
    )
    events = store.events.recent("ep1", limit=10, kind="recovery_selected")

    assert updated is not None
    assert len(events) == 2
    assert updated.recovery is not None
    assert updated.recovery["strategy"] == "pause_for_operator"


def test_task_run_store_marks_skill_completion_failure_and_deduplicates_skill_ids(
    tmp_path,
):
    store = TaskRunStore(tmp_path)
    store.ensure_active(
        episode_id="ep1", task="pick bottle", agent_id=None, robot_id=None
    )

    first = store.bind_skill("ep1", "skill1", "approach bottle")
    second = store.bind_skill("ep1", "skill1", "retry approach bottle")

    assert first is not None
    assert second is not None
    assert second.skill_ids == ["skill1"]
    assert len(second.attempts) == 2

    failed = store.mark_skill_completed(
        "ep1", skill_id="skill1", summary="base path blocked", success=False
    )
    assert failed is not None
    assert failed.status == "recovering"
    assert failed.last_step_success is False
    assert failed.attempts[-1].status == "failed"
    assert failed.attempts[-1].metadata["completion_summary"] == "base path blocked"


def test_task_run_store_persists_attempt_metadata_for_bound_skill(tmp_path):
    store = TaskRunStore(tmp_path)
    store.ensure_active(
        episode_id="ep1", task="pick bottle", agent_id=None, robot_id=None
    )

    bound = store.bind_skill_with_metadata(
        "ep1",
        "skill1",
        "inspect current scene",
        metadata={"recovery_resume": {"strategy": "reobserve"}},
    )

    assert bound is not None
    assert bound.attempts[-1].metadata["recovery_resume"]["strategy"] == "reobserve"


def test_task_run_store_persists_skill_trace_for_bound_and_completed_skill(tmp_path):
    store = TaskRunStore(tmp_path)
    store.ensure_active(
        episode_id="ep1", task="pick bottle", agent_id=None, robot_id=None
    )

    bound = store.bind_skill_with_metadata(
        "ep1",
        "skill1",
        "pick the cup",
        metadata={
            "skill": "vla_manipulation",
            "backend": "foundation",
            "implementation_name": "vla_manipulation",
            "implementation_kind": "capability_service",
        },
    )
    assert bound is not None
    assert bound.attempts[-1].skill == "vla_manipulation"

    updated = store.bind_skill_trace_metadata(
        "ep1",
        skill_id="skill1",
        status="completed",
        success=True,
        summary="picked the cup",
        metadata={
            "skill": "vla_manipulation",
            "backend": "foundation",
            "implementation_name": "vla_manipulation",
            "implementation_kind": "capability_service",
        },
    )

    assert updated is not None
    assert updated.skill_trace
    assert updated.skill_trace[-1]["skill"] == "vla_manipulation"
    assert updated.skill_trace[-1]["backend"] == "foundation"
    assert updated.skill_trace[-1]["implementation_name"] == "vla_manipulation"
    assert updated.skill_trace[-1]["status"] == "completed"


def test_task_run_skill_trace_keeps_northbound_skill_name_for_composite_skill(tmp_path):
    store = TaskRunStore(tmp_path)
    store.ensure_active(
        episode_id="ep1", task="hand the cup to the user", agent_id=None, robot_id=None
    )

    state = store.bind_skill_with_metadata(
        "ep1",
        "skill1",
        "hand the cup to the user",
        metadata={
            "skill": "vla_manipulation",
            "backend": "foundation",
            "implementation_name": "vla_manipulation",
            "implementation_kind": "skill_composite",
        },
    )

    assert state is not None
    assert state.skill_trace[-1]["skill"] == "vla_manipulation"
    assert state.skill_trace[-1]["implementation_kind"] == "skill_composite"
    assert state.skill_trace[-1]["implementation_name"] == "vla_manipulation"
    assert state.skill_trace[-1]["skill"] not in {
        "open_gripper",
        "reset_posture",
    }


def test_task_run_store_ignores_missing_active_task_and_bad_json_files(tmp_path):
    store = TaskRunStore(tmp_path)

    assert store.bind_skill("ep1", "skill1", "look") is None
    assert (
        store.mark_execution_feedback(
            "ep1", skill_id="skill1", success=True, summary="ok"
        )
        is None
    )
    assert store.append_attempt("ep1", event="note", summary="nothing active") is None

    bad_path = store.root / "bad.task.json"
    bad_path.write_text("{not-json", encoding="utf-8")
    other_path = store.root / "other.task.json"
    other_path.write_text(
        json.dumps(
            TaskRun(
                task_id="task/unsafe:id", episode_id="ep2", root_task="other"
            ).to_dict()
        ),
        encoding="utf-8",
    )

    assert store.list_for_episode("ep1") == []
    assert [task.episode_id for task in store.list_recent(limit=0)] == ["ep2"]


def test_task_run_store_retries_atomic_replace_on_windows_style_permission_error(
    tmp_path, monkeypatch
):
    store = TaskRunStore(tmp_path)
    task = store.ensure_active(
        episode_id="ep1", task="pick bottle", agent_id=None, robot_id=None
    )

    path_type = type(store._path(task.task_id))
    original_replace = path_type.replace
    calls = {"count": 0}

    def flaky_replace(self, target):
        calls["count"] += 1
        if calls["count"] == 1:
            raise PermissionError("simulated temporary lock")
        return original_replace(self, target)

    monkeypatch.setattr(path_type, "replace", flaky_replace)

    task.status = "recovering"
    saved = store.save(task)

    assert saved.status == "recovering"
    assert calls["count"] == 2
    assert store.load_active("ep1") is not None
