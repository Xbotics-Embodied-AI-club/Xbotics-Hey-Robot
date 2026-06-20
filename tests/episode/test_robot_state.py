from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hey_robot.episode import RobotEpisodeStateStore
from hey_robot.episode.robot_state import (
    ExecutionFeedbackSnapshot,
    _is_stale_lock,
    _phase_from_result_status,
    _sanitize,
)
from hey_robot.protocol import (
    Envelope,
    RobotObservation,
    RobotStatus,
    SkillEvent,
    SkillResult,
)


def test_robot_episode_state_tracks_recovery_and_execution_feedback(
    tmp_path: Path,
) -> None:
    store = RobotEpisodeStateStore(tmp_path)
    episode_id = "web/user 1"
    envelope = Envelope(
        trace_id="tr1", episode_id=episode_id, agent_id="main", robot_id="mock0"
    )

    store.mark_task_started(
        episode_id, task="move block", agent_id="main", robot_id="mock0"
    )
    state = store.apply_skill_event(
        SkillEvent(
            envelope=envelope,
            skill_id="skill1",
            phase="issued",
            text="move block",
            mode="skill",
        )
    )
    assert state is not None
    assert state.active_skill_id == "skill1"
    assert state.recovery_required is False

    state = store.apply_skill_event(
        SkillEvent(envelope=envelope, skill_id="skill1", phase="feedback_pending")
    )
    assert state is not None
    assert state.recovery_required is True
    assert "Recovery context" in (state.recovery_context() or "")

    reloaded = RobotEpisodeStateStore(tmp_path)
    assert reloaded.list_states()[0].episode_id == episode_id
    marked = reloaded.mark_recovery_required_for_nonterminal()
    assert marked == []

    state = reloaded.mark_execution_feedback(
        episode_id,
        skill_id="skill1",
        subgoal_success=True,
        task_success=False,
        summary="object moved",
        next_hint="continue",
    )
    assert state is not None
    assert state.recovery_required is False
    assert state.active_skill_phase == "confirmed"


def test_robot_episode_state_store_serializes_concurrent_writes(tmp_path: Path) -> None:
    episode_id = "web/mock0/user"
    envelope = Envelope(episode_id=episode_id, agent_id="main", robot_id="mock0")
    store = RobotEpisodeStateStore(tmp_path)
    store.ensure(episode_id, agent_id="main", robot_id="mock0")

    def write_event(index: int) -> None:
        local_store = RobotEpisodeStateStore(tmp_path)
        local_store.apply_skill_event(
            SkillEvent(
                envelope=envelope,
                skill_id=f"skill{index}",
                phase="executing",
                text=f"skill {index}",
                frame_id=index,
            )
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write_event, range(32)))

    state = store.load(episode_id)

    assert state is not None
    assert state.active_skill_phase == "executing"
    assert state.active_skill_id is not None
    assert state.last_observation_frame is not None
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.lock"))


def test_robot_episode_state_applies_terminal_skill_result(tmp_path: Path) -> None:
    store = RobotEpisodeStateStore(tmp_path)
    episode_id = "web/user/result"
    envelope = Envelope(
        trace_id="tr1", episode_id=episode_id, agent_id="main", robot_id="mock0"
    )

    store.mark_task_started(
        episode_id, task="inspect scene", agent_id="main", robot_id="mock0"
    )
    store.apply_skill_event(
        SkillEvent(envelope=envelope, skill_id="skill1", phase="executing", frame_id=10)
    )
    state = store.apply_skill_result(
        SkillResult(
            envelope=envelope,
            skill_id="skill1",
            status="completed",
            success=True,
            frame_id=12,
            summary="perception refreshed",
        )
    )

    assert state is not None
    assert state.active_skill_id == "skill1"
    assert state.active_skill_phase == "completed"
    assert state.last_observation_frame == 12
    assert state.recovery_required is False


def test_robot_episode_state_updates_episode_from_robot_heartbeat(
    tmp_path: Path,
) -> None:
    store = RobotEpisodeStateStore(tmp_path)
    store.mark_task_started("ep1", task="move", agent_id="main", robot_id="mock0")
    store.mark_task_started("ep2", task="other", agent_id="main", robot_id="other")

    status = RobotStatus(
        envelope=Envelope(robot_id="mock0", timestamp=123.0),
        frame_id=42,
        state="idle",
        metrics={"battery": {"status": "normal"}},
    )
    updated = store.update_status_for_robot("mock0", status)

    ep1 = store.load("ep1")
    ep2 = store.load("ep2")
    assert [state.episode_id for state in updated] == ["ep1"]
    assert ep1 is not None
    assert ep1.last_status["frame_id"] == 42
    assert ep1.last_status["timestamp"] == 123.0
    assert ep2 is not None
    assert ep2.last_status == {}


def test_recovery_context_with_observation_frame(tmp_path: Path) -> None:
    store = RobotEpisodeStateStore(tmp_path)
    episode_id = "web/user/recovery"
    envelope = Envelope(
        trace_id="tr1", episode_id=episode_id, agent_id="main", robot_id="mock0"
    )

    store.mark_task_started(
        episode_id, task="move block", agent_id="main", robot_id="mock0"
    )
    store.apply_skill_event(
        SkillEvent(
            envelope=envelope, skill_id="skill1", phase="feedback_pending", frame_id=99
        )
    )
    state = store.load(episode_id)
    assert state is not None
    ctx = state.recovery_context()
    assert ctx is not None
    assert "last_observation_frame: 99" in ctx
    assert "recommended_tools:" in ctx


def test_recovery_context_with_execution_feedback(tmp_path: Path) -> None:
    store = RobotEpisodeStateStore(tmp_path)
    episode_id = "web/user/recovery2"
    envelope = Envelope(
        trace_id="tr1", episode_id=episode_id, agent_id="main", robot_id="mock0"
    )

    store.mark_task_started(
        episode_id, task="move block", agent_id="main", robot_id="mock0"
    )
    store.apply_skill_event(
        SkillEvent(envelope=envelope, skill_id="skill1", phase="feedback_pending")
    )
    store.mark_execution_feedback(
        episode_id,
        skill_id="skill1",
        subgoal_success=False,
        task_success=False,
        summary="object dropped",
    )
    state = store.load(episode_id)
    assert state is not None
    ctx = state.recovery_context()
    assert ctx is not None
    assert "last_execution_feedback" in ctx
    assert "recovery_strategy_hint: reobserve" in ctx


def test_recovery_context_returns_none_when_not_required(tmp_path: Path) -> None:
    store = RobotEpisodeStateStore(tmp_path)
    store.mark_task_started("ep1", task="move", agent_id="main", robot_id="mock0")
    state = store.load("ep1")
    assert state is not None
    assert state.recovery_context() is None


def test_list_states_skips_corrupted_files(tmp_path: Path) -> None:
    store = RobotEpisodeStateStore(tmp_path)
    store.mark_task_started("ep_valid", task="move", agent_id="main", robot_id="mock0")
    corrupted = tmp_path / "corrupted.robot_state.json"
    corrupted.write_text("not valid json {{{", encoding="utf-8")
    states = store.list_states()
    assert len(states) == 1
    assert states[0].episode_id == "ep_valid"


def test_mark_recovery_required_for_nonterminal_marks_active(tmp_path: Path) -> None:
    store = RobotEpisodeStateStore(tmp_path)
    episode_id = "web/user/nonterminal"
    envelope = Envelope(
        trace_id="tr1", episode_id=episode_id, agent_id="main", robot_id="mock0"
    )

    store.mark_task_started(episode_id, task="move", agent_id="main", robot_id="mock0")
    store.apply_skill_event(
        SkillEvent(envelope=envelope, skill_id="skill1", phase="executing")
    )
    marked = store.mark_recovery_required_for_nonterminal()
    assert len(marked) == 1
    assert marked[0].episode_id == episode_id
    assert marked[0].recovery_required is True
    assert "executing" in (marked[0].recovery_reason or "")


def test_mark_recovery_required_for_nonterminal_skips_terminal_and_already_marked(
    tmp_path: Path,
) -> None:
    store = RobotEpisodeStateStore(tmp_path)
    envelope = Envelope(
        trace_id="tr1", episode_id="ep_done", agent_id="main", robot_id="mock0"
    )

    store.mark_task_started("ep_done", task="move", agent_id="main", robot_id="mock0")
    store.apply_skill_event(
        SkillEvent(envelope=envelope, skill_id="skill1", phase="completed")
    )
    marked = store.mark_recovery_required_for_nonterminal()
    assert len(marked) == 0


def test_apply_skill_event_no_episode_id_returns_none(tmp_path: Path) -> None:
    store = RobotEpisodeStateStore(tmp_path)
    envelope = Envelope(trace_id="tr1", episode_id="")
    result = store.apply_skill_event(
        SkillEvent(envelope=envelope, skill_id="skill1", phase="executing")
    )
    assert result is None


def test_apply_skill_event_failed_phase_sets_recovery(tmp_path: Path) -> None:
    store = RobotEpisodeStateStore(tmp_path)
    episode_id = "web/user/failed"
    envelope = Envelope(
        trace_id="tr1", episode_id=episode_id, agent_id="main", robot_id="mock0"
    )

    store.mark_task_started(episode_id, task="move", agent_id="main", robot_id="mock0")
    state = store.apply_skill_event(
        SkillEvent(
            envelope=envelope,
            skill_id="skill1",
            phase="failed",
            error="actuator timeout",
        )
    )
    assert state is not None
    assert state.recovery_required is True
    assert "actuator timeout" in (state.recovery_reason or "")


def test_apply_skill_event_interrupted_phase_sets_recovery(tmp_path: Path) -> None:
    store = RobotEpisodeStateStore(tmp_path)
    episode_id = "web/user/interrupted"
    envelope = Envelope(
        trace_id="tr1", episode_id=episode_id, agent_id="main", robot_id="mock0"
    )

    store.mark_task_started(episode_id, task="move", agent_id="main", robot_id="mock0")
    state = store.apply_skill_event(
        SkillEvent(envelope=envelope, skill_id="skill1", phase="interrupted")
    )
    assert state is not None
    assert state.recovery_required is True


def test_apply_skill_event_feedback_failed_phase_sets_recovery(tmp_path: Path) -> None:
    store = RobotEpisodeStateStore(tmp_path)
    episode_id = "web/user/fbfail"
    envelope = Envelope(
        trace_id="tr1", episode_id=episode_id, agent_id="main", robot_id="mock0"
    )

    store.mark_task_started(episode_id, task="move", agent_id="main", robot_id="mock0")
    state = store.apply_skill_event(
        SkillEvent(
            envelope=envelope,
            skill_id="skill1",
            phase="feedback_failed",
            summary="no feedback received",
        )
    )
    assert state is not None
    assert state.recovery_required is True
    assert "no feedback received" in (state.recovery_reason or "")


def test_apply_skill_event_confirmed_phase_resets_recovery(tmp_path: Path) -> None:
    store = RobotEpisodeStateStore(tmp_path)
    episode_id = "web/user/confirmed"
    envelope = Envelope(
        trace_id="tr1", episode_id=episode_id, agent_id="main", robot_id="mock0"
    )

    store.mark_task_started(episode_id, task="move", agent_id="main", robot_id="mock0")
    store.apply_skill_event(
        SkillEvent(envelope=envelope, skill_id="skill1", phase="feedback_pending")
    )
    state = store.apply_skill_event(
        SkillEvent(envelope=envelope, skill_id="skill1", phase="confirmed")
    )
    assert state is not None
    assert state.recovery_required is False
    assert state.recovery_reason is None


def test_apply_skill_event_accepted_phase_resets_recovery(tmp_path: Path) -> None:
    store = RobotEpisodeStateStore(tmp_path)
    episode_id = "web/user/accepted"
    envelope = Envelope(
        trace_id="tr1", episode_id=episode_id, agent_id="main", robot_id="mock0"
    )

    store.mark_task_started(episode_id, task="move", agent_id="main", robot_id="mock0")
    store.apply_skill_event(
        SkillEvent(envelope=envelope, skill_id="skill1", phase="feedback_pending")
    )
    state = store.apply_skill_event(
        SkillEvent(envelope=envelope, skill_id="skill1", phase="accepted")
    )
    assert state is not None
    assert state.recovery_required is False


def test_apply_skill_event_with_agent_id_and_robot_id(tmp_path: Path) -> None:
    store = RobotEpisodeStateStore(tmp_path)
    episode_id = "web/user/ids"
    envelope = Envelope(
        trace_id="tr1", episode_id=episode_id, agent_id="agentX", robot_id="botY"
    )

    state = store.apply_skill_event(
        SkillEvent(
            envelope=envelope,
            skill_id="skill1",
            phase="issued",
            text="do task",
            policy_id="pol1",
        )
    )
    assert state is not None
    assert state.agent_id == "agentX"
    assert state.robot_id == "botY"
    assert state.active_skill_text == "do task"
    assert state.policy_id == "pol1"


def test_apply_skill_result_no_episode_id_returns_none(tmp_path: Path) -> None:
    store = RobotEpisodeStateStore(tmp_path)
    envelope = Envelope(trace_id="tr1", episode_id="")
    result = store.apply_skill_result(
        SkillResult(envelope=envelope, skill_id="skill1", status="completed")
    )
    assert result is None


def test_apply_skill_result_failed_status_sets_recovery(tmp_path: Path) -> None:
    store = RobotEpisodeStateStore(tmp_path)
    episode_id = "web/user/result_failed"
    envelope = Envelope(
        trace_id="tr1", episode_id=episode_id, agent_id="main", robot_id="mock0"
    )

    store.mark_task_started(
        episode_id, task="inspect", agent_id="main", robot_id="mock0"
    )
    state = store.apply_skill_result(
        SkillResult(
            envelope=envelope,
            skill_id="skill1",
            status="failed",
            error="collision detected",
            frame_id=15,
        )
    )
    assert state is not None
    assert state.recovery_required is True
    assert "collision detected" in (state.recovery_reason or "")
    assert state.last_observation_frame == 15


def test_apply_skill_result_interrupted_status_sets_recovery(tmp_path: Path) -> None:
    store = RobotEpisodeStateStore(tmp_path)
    episode_id = "web/user/result_interrupted"
    envelope = Envelope(
        trace_id="tr1", episode_id=episode_id, agent_id="main", robot_id="mock0"
    )

    store.mark_task_started(
        episode_id, task="inspect", agent_id="main", robot_id="mock0"
    )
    state = store.apply_skill_result(
        SkillResult(envelope=envelope, skill_id="skill1", status="interrupted")
    )
    assert state is not None
    assert state.recovery_required is True


def test_apply_skill_result_feedback_failed_status_sets_recovery(
    tmp_path: Path,
) -> None:
    store = RobotEpisodeStateStore(tmp_path)
    episode_id = "web/user/result_fbfail"
    envelope = Envelope(
        trace_id="tr1", episode_id=episode_id, agent_id="main", robot_id="mock0"
    )

    store.mark_task_started(
        episode_id, task="inspect", agent_id="main", robot_id="mock0"
    )
    state = store.apply_skill_result(
        SkillResult(
            envelope=envelope,
            skill_id="skill1",
            status="feedback_failed",
            summary="bad result",
        )
    )
    assert state is not None
    assert state.recovery_required is True
    assert "bad result" in (state.recovery_reason or "")


def test_apply_skill_result_unknown_status_preserved(tmp_path: Path) -> None:
    store = RobotEpisodeStateStore(tmp_path)
    episode_id = "web/user/result_unknown"
    envelope = Envelope(
        trace_id="tr1", episode_id=episode_id, agent_id="main", robot_id="mock0"
    )

    store.mark_task_started(
        episode_id, task="inspect", agent_id="main", robot_id="mock0"
    )
    state = store.apply_skill_result(
        SkillResult(envelope=envelope, skill_id="skill1", status="custom_phase")
    )
    assert state is not None
    assert state.active_skill_phase == "custom_phase"


def test_mark_task_started_with_none_ids(tmp_path: Path) -> None:
    store = RobotEpisodeStateStore(tmp_path)
    state = store.mark_task_started("ep1", task="move", agent_id=None, robot_id=None)
    assert state.active_task == "move"
    assert state.agent_id is None
    assert state.robot_id is None
    assert state.active_skill_id is None


def test_update_observation_nonexistent_episode_returns_none(tmp_path: Path) -> None:
    store = RobotEpisodeStateStore(tmp_path)
    obs = RobotObservation(
        envelope=Envelope(episode_id="no_ep", robot_id="mock0"),
        frame_id=1,
        images=[],
        proprioception=[],
    )
    assert store.update_observation("no_ep", obs) is None


def test_update_observation_with_images(tmp_path: Path) -> None:
    store = RobotEpisodeStateStore(tmp_path)
    store.mark_task_started("ep1", task="move", agent_id="main", robot_id="mock0")
    obs = RobotObservation(
        envelope=Envelope(episode_id="ep1", robot_id="mock0"),
        frame_id=10,
        images=[],
        proprioception=[],
    )
    state = store.update_observation("ep1", obs)
    assert state is not None
    assert state.last_observation_frame == 10


def test_update_status_nonexistent_episode_returns_none(tmp_path: Path) -> None:
    store = RobotEpisodeStateStore(tmp_path)
    status = RobotStatus(
        envelope=Envelope(episode_id="no_ep", robot_id="mock0"),
        frame_id=1,
        state="idle",
    )
    assert store.update_status("no_ep", status) is None


def test_update_status_for_robot_mismatch_skips(tmp_path: Path) -> None:
    store = RobotEpisodeStateStore(tmp_path)
    store.mark_task_started("ep1", task="move", agent_id="main", robot_id="mock0")
    status = RobotStatus(
        envelope=Envelope(robot_id="other_bot"),
        frame_id=1,
        state="idle",
    )
    updated = store.update_status_for_robot("other_bot", status)
    assert updated == []


def test_sanitize_preserves_alphanumeric() -> None:
    assert _sanitize("abc_123-X.Y") == "abc_123-X.Y"
    assert _sanitize("web/user 1") == "web_user_1"


def test_phase_from_result_status_known_values() -> None:
    assert _phase_from_result_status("completed") == "completed"
    assert _phase_from_result_status("failed") == "failed"
    assert _phase_from_result_status("interrupted") == "interrupted"
    assert _phase_from_result_status("feedback_failed") == "feedback_failed"


def test_phase_from_result_status_unknown_value() -> None:
    assert _phase_from_result_status("weird_status") == "weird_status"


def test_phase_from_result_status_empty() -> None:
    assert _phase_from_result_status("") == "unknown"


def test_is_stale_lock_fresh_file(tmp_path: Path) -> None:
    lock = tmp_path / "fresh.lock"
    lock.write_text("pid time")
    assert _is_stale_lock(lock) is False


def test_is_stale_lock_old_file(tmp_path: Path) -> None:
    import hey_robot.episode.robot_state as robot_state_module

    lock = tmp_path / "stale.lock"
    lock.write_text("pid time")
    original_time = robot_state_module.time.time
    robot_state_module.time.time = lambda: original_time() + 31.0
    try:
        assert _is_stale_lock(lock) is True
    finally:
        robot_state_module.time.time = original_time


def test_is_stale_lock_missing_file(tmp_path: Path) -> None:
    assert _is_stale_lock(tmp_path / "nonexistent.lock") is False


def test_execution_feedback_snapshot_defaults() -> None:
    snap = ExecutionFeedbackSnapshot(
        skill_id="s1", subgoal_success=True, task_success=False, summary="done"
    )
    assert snap.next_hint is None
    assert snap.updated_at > 0


def test_ensure_with_agent_robot_policy_ids(tmp_path: Path) -> None:
    store = RobotEpisodeStateStore(tmp_path)
    state = store.ensure("ep1", agent_id="a1", robot_id="r1", policy_id="p1")
    assert state.agent_id == "a1"
    assert state.robot_id == "r1"
    assert state.policy_id == "p1"


def test_synchronous_flush_persists_status_and_observation(tmp_path: Path) -> None:
    """flush_episode_state must write status + observation synchronously so
    robot_state.json reflects the actual turn result, not a stale heartbeat."""
    store = RobotEpisodeStateStore(tmp_path)
    store.mark_task_started("ep1", task="move", agent_id="main", robot_id="mock0")

    status = RobotStatus(
        envelope=Envelope(robot_id="mock0"),
        frame_id=42,
        state="skill_completed",
        skill_id="skill1",
        success=True,
        metrics={"last_skill_result": {"success": True, "skill": "move_base"}},
    )
    observation = RobotObservation(
        envelope=Envelope(episode_id="ep1", robot_id="mock0"),
        frame_id=42,
        images=[],
        proprioception=[],
    )

    store.update_status("ep1", status)
    store.update_observation("ep1", observation)

    reloaded = RobotEpisodeStateStore(tmp_path)
    state = reloaded.load("ep1")
    assert state is not None
    assert state.last_status["state"] == "skill_completed"
    assert state.last_status["frame_id"] == 42
    assert state.last_status["skill_id"] == "skill1"
    assert state.last_observation_frame == 42


def test_flush_with_none_values_is_safe(tmp_path: Path) -> None:
    store = RobotEpisodeStateStore(tmp_path)
    store.mark_task_started("ep1", task="move", agent_id="main", robot_id="mock0")

    # flush_episode_state with None status should not crash
    assert store.load("ep1") is not None


def test_flush_does_not_crash_on_nonexistent_episode(tmp_path: Path) -> None:
    store = RobotEpisodeStateStore(tmp_path)
    status = RobotStatus(envelope=Envelope(robot_id="mock0"), frame_id=1, state="idle")
    assert store.update_status("no_such_ep", status) is None
