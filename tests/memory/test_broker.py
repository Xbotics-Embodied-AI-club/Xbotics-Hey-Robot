from __future__ import annotations

import time

from hey_robot.agents.task_events import RobotTaskEventLog
from hey_robot.agents.task_run import TaskAttempt, TaskRun
from hey_robot.memory.broker import MemoryBroker
from hey_robot.memory.long_term import LongTermMemoryStore
from hey_robot.memory.runtime import MemoryRuntime
from hey_robot.memory.scene import SceneMemoryRecord, SceneMemoryStore


def _task(**kw) -> TaskRun:
    defaults = {
        "task_id": "task-1",
        "episode_id": "ep-1",
        "agent_id": "main",
        "robot_id": "mock0",
        "root_task": "pick up the cup",
        "status": "active",
    }
    return TaskRun(**(defaults | kw))


def _task_with_attempts(**kw) -> TaskRun:
    task = _task(**kw)
    task.attempts = [
        TaskAttempt(
            attempt_id="a1",
            text="inspect scene",
            status="completed",
            skill_id="inspect_scene",
            success=True,
        ),
        TaskAttempt(
            attempt_id="a2",
            text="pick up cup",
            status="completed",
            skill_id="vla_manipulation",
            success=True,
        ),
    ]
    return task


def _task_with_failed_attempt(**kw) -> TaskRun:
    task = _task(**kw)
    task.attempts = [
        TaskAttempt(
            attempt_id="a1",
            text="inspect scene",
            status="completed",
            skill_id="inspect_scene",
            success=True,
        ),
        TaskAttempt(
            attempt_id="a2",
            text="pick up cup",
            status="completed",
            skill_id="vla_manipulation",
            success=False,
            metadata={
                "execution_feedback": {
                    "summary": "grasp failed: object slipped from gripper"
                }
            },
        ),
    ]
    return task


def _broker(tmp_path, **kw) -> MemoryBroker:
    defaults = {
        "scene_memory": SceneMemoryStore(tmp_path, max_items=64),
        "task_events": RobotTaskEventLog(tmp_path),
    }
    return MemoryBroker(**(defaults | kw))


def _record(episode_id: str, summary: str, *, frame_id: int = 1) -> SceneMemoryRecord:
    return SceneMemoryRecord(
        record_id=f"rec-{time.time()}",
        episode_id=episode_id,
        robot_id="mock0",
        frame_id=frame_id,
        summary=summary,
        timestamp=time.time(),
    )


class _FakeTaskRuns:
    def record_scene_memory(
        self, episode_id, *, summary, frame_id, metadata=None
    ) -> None:
        pass


# ── routing: active task ────────────────────────────────────────────


def test_broker_active_task_includes_task_state_block(tmp_path) -> None:
    broker = _broker(tmp_path)
    task = _task_with_attempts()

    ctx = broker.build(task=task, task_text="pick up the cup")

    assert ctx is not None
    assert "Current task state:" in ctx
    assert "pick up the cup" in ctx
    assert "status: active" in ctx
    assert "2 completed, 0 failed" in ctx
    assert "vla_manipulation -> completed" in ctx


def test_broker_active_task_includes_scene_evidence(tmp_path) -> None:
    broker = _broker(tmp_path)
    task = _task_with_attempts(episode_id="ep-1")
    broker.append_scene(
        _record("ep-1", "a cup and a bottle on the table", frame_id=1),
        task_runs=_FakeTaskRuns(),
    )

    ctx = broker.build(task=task, task_text="pick up the cup")

    assert ctx is not None
    assert "a cup and a bottle" in ctx


def test_broker_active_task_includes_ltm_when_ltm_configured(tmp_path) -> None:
    ltm_store = LongTermMemoryStore(tmp_path / "ltm.jsonl")
    ltm = MemoryRuntime(ltm_store)
    ltm.write(
        kind="user_preference",
        name="cup_location",
        value="left side of the desk",
        summary="user prefers the cup on the left side of the desk",
    )
    broker = _broker(tmp_path, ltm_runtime=ltm)
    task = _task_with_attempts()

    ctx = broker.build(task=task, task_text="cup location")

    assert ctx is not None
    assert "cup" in ctx


def test_broker_active_task_appends_current_and_perception_context(tmp_path) -> None:
    broker = _broker(tmp_path)
    task = _task_with_attempts()

    ctx = broker.build(
        task=task,
        task_text="pick up the cup",
        current_context="The robot is at position (0,0).",
        perception_context="Camera sees a cup at (1.2, 0.3).",
    )

    assert ctx is not None
    assert "The robot is at position (0,0)." in ctx
    assert "Camera sees a cup at (1.2, 0.3)." in ctx


# ── routing: recovering task ────────────────────────────────────────


def test_broker_recovering_task_includes_recovery_block(tmp_path) -> None:
    broker = _broker(tmp_path)
    task = _task_with_failed_attempt(
        status="recovering",
        failure_reason="grasp failed on attempt a2",
        recovery={"strategy": "reobserve", "summary": "object may have moved"},
    )

    ctx = broker.build(
        task=task,
        task_text="pick up the cup",
        recovery_context="Recovery: object may have moved. Re-observe the scene.",
    )

    assert ctx is not None
    assert "Recovery: object may have moved" in ctx
    assert "Failure reason: grasp failed on attempt a2" in ctx


def test_broker_recovering_task_excludes_scene_dump(tmp_path) -> None:
    broker = _broker(tmp_path)
    task = _task_with_failed_attempt(status="recovering")
    broker.append_scene(
        _record("ep-1", "cluttered desk with many objects", frame_id=1),
        task_runs=_FakeTaskRuns(),
    )

    ctx = broker.build(
        task=task,
        task_text="pick up the cup",
        recovery_context="Re-observe before retry.",
    )

    assert ctx is not None
    assert "cluttered desk" not in ctx


def test_broker_recovering_task_includes_task_state_block(tmp_path) -> None:
    broker = _broker(tmp_path)
    task = _task_with_failed_attempt(status="recovering")

    ctx = broker.build(task=task, task_text="pick up the cup")

    assert ctx is not None
    assert "Current task state:" in ctx
    assert "status: recovering" in ctx
    assert "1 completed, 1 failed" in ctx


# ── routing: feedback_pending ───────────────────────────────────────


def test_broker_feedback_pending_task_includes_last_attempt_block(tmp_path) -> None:
    broker = _broker(tmp_path)
    task = _task_with_failed_attempt(status="feedback_pending")

    ctx = broker.build(task=task, task_text="pick up the cup")

    assert ctx is not None
    assert "Last attempt: vla_manipulation -> failed" in ctx
    assert "grasp failed: object slipped from gripper" in ctx


def test_broker_feedback_pending_excludes_scene_and_ltm(tmp_path) -> None:
    broker = _broker(tmp_path)
    task = _task_with_failed_attempt(status="feedback_pending")
    broker.append_scene(
        _record("ep-1", "irrelevant scene data", frame_id=1),
        task_runs=_FakeTaskRuns(),
    )

    ctx = broker.build(task=task, task_text="pick up the cup")

    assert ctx is not None
    assert "irrelevant scene data" not in ctx


# ── routing: completed / reported ───────────────────────────────────


def test_broker_completed_task_includes_summary_block(tmp_path) -> None:
    broker = _broker(tmp_path)
    task = _task_with_attempts(status="completed", task_success=True)

    ctx = broker.build(task=task)

    assert ctx is not None
    assert "Task completed: pick up the cup" in ctx
    assert "success: True" in ctx
    assert "attempts: 2" in ctx


def test_broker_completed_task_excludes_scene_and_ltm(tmp_path) -> None:
    broker = _broker(tmp_path)
    task = _task_with_attempts(status="completed")
    broker.append_scene(
        _record("ep-1", "stale scene", frame_id=1),
        task_runs=_FakeTaskRuns(),
    )

    ctx = broker.build(task=task)

    assert ctx is not None
    assert "stale scene" not in ctx


# ── routing: idle (no task) ─────────────────────────────────────────


def test_broker_idle_no_task_excludes_scene(tmp_path) -> None:
    broker = _broker(tmp_path)
    broker.append_scene(
        _record("ep-1", "unrelated scene", frame_id=1),
        task_runs=_FakeTaskRuns(),
    )

    ctx = broker.build(task=None, task_text="hello")

    assert ctx is None or "unrelated scene" not in ctx


# ── skill catalog context ───────────────────────────────────────────


def test_broker_always_includes_skill_catalog_when_provided(tmp_path) -> None:
    broker = _broker(tmp_path)

    for status in ("active", "recovering", "completed"):
        task = _task(status=status)
        ctx = broker.build(
            task=task,
            skill_catalog_context="Available skills: inspect_scene, vla_manipulation",
        )
        assert ctx is not None
        assert "Available skills" in ctx


# ── task state block edge cases ─────────────────────────────────────


def test_task_state_block_null_task(tmp_path) -> None:
    broker = _broker(tmp_path)
    assert broker._task_state_block(None) is None


def test_task_state_block_preserves_recovery_strategy(tmp_path) -> None:
    broker = _broker(tmp_path)
    task = _task(
        recovery={"strategy": "retry_with_adjustment", "summary": "adjust grip"}
    )

    block = broker._task_state_block(task)

    assert block is not None
    assert "needed (retry_with_adjustment)" in block


def test_task_state_block_handles_empty_recovery(tmp_path) -> None:
    broker = _broker(tmp_path)
    task = _task(recovery={})

    block = broker._task_state_block(task)

    assert block is not None
    assert "recovery: needed" in block


def test_task_state_block_no_recovery_field(tmp_path) -> None:
    broker = _broker(tmp_path)
    task = _task()
    task.recovery = None

    block = broker._task_state_block(task)

    assert block is not None
    assert "recovery: not_needed" in block


def test_task_state_block_without_attempts(tmp_path) -> None:
    broker = _broker(tmp_path)
    task = _task()

    block = broker._task_state_block(task)

    assert block is not None
    assert "0 completed, 0 failed" in block
