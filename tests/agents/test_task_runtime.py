from __future__ import annotations

from hey_robot.agents.context import RobotContextBuilder
from hey_robot.agents.execution_feedback import ExecutionFeedback
from hey_robot.agents.injection import RobotTurnInjector
from hey_robot.agents.task_run import TaskRunStore
from hey_robot.agents.task_runtime import TaskRunManager
from hey_robot.agents.types import RobotSnapshot
from hey_robot.episode import RobotEpisodeStateStore
from hey_robot.memory import SceneMemoryRecord
from hey_robot.protocol import (
    Envelope,
    RobotObservation,
    RobotStatus,
    SkillResult,
    UserTurn,
)


def _runtime(tmp_path) -> TaskRunManager:
    return TaskRunManager(
        episode_root=tmp_path,
        runtime_dir=tmp_path,
        events_max_items=100,
        robot_states=RobotEpisodeStateStore(tmp_path),
    )


def test_task_runtime_ignores_missing_episode_boundaries(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    turn = UserTurn(envelope=Envelope(trace_id="tr1", robot_id="mock0"), text="hello")

    runtime.mark_task_started(turn, agent_id="main")
    runtime.mark_restore(turn)
    runtime.clear_for_new_turn(None)
    runtime.clear_pending_confirmation(None)
    runtime.store_pending_confirmation(None, {"objective": "x"})

    assert runtime.pending_confirmation(None) is None
    assert runtime.recovery_context(None) is None


def test_task_runtime_status_and_observation_robot_fallbacks(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    episode_id = "ep1"
    runtime.robot_states.ensure(episode_id, agent_id="main", robot_id="mock0")

    status = RobotStatus(envelope=Envelope(robot_id="mock0"), state="idle")
    runtime.observe_status(status)
    state = runtime.robot_states.load(episode_id)
    assert state is not None
    assert state.last_status["state"] == "idle"

    runtime.observe_observation(
        RobotObservation(envelope=Envelope(robot_id="mock0"), frame_id=1)
    )
    state = runtime.robot_states.load(episode_id)
    assert state is not None
    assert state.last_observation_frame is None


def test_task_runtime_follow_up_on_recovering_task_resumes_same_root_task_and_clears_recovery(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    episode_id = "ep1"
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr1", episode_id=episode_id, agent_id="main", robot_id="mock0"
        ),
        text="pick the cup",
    )
    runtime.mark_task_started(turn, agent_id="main")
    runtime.task_runs.ensure_active(
        episode_id=episode_id,
        task="pick the cup",
        agent_id="main",
        robot_id="mock0",
    )
    runtime.task_runs.set_recovery(
        episode_id,
        strategy="reobserve",
        summary="inspect the table again",
        metadata={"needed": True},
    )
    runtime.robot_states.ensure(episode_id, agent_id="main", robot_id="mock0")
    runtime.robot_states.apply_skill_result(
        SkillResult(
            envelope=Envelope(
                trace_id="tr0", episode_id=episode_id, agent_id="main", robot_id="mock0"
            ),
            skill_id="skill1",
            status="failed",
            error="cup not visible",
        )
    )

    follow_up = UserTurn(
        envelope=Envelope(
            trace_id="tr2", episode_id=episode_id, agent_id="main", robot_id="mock0"
        ),
        text="继续",
        metadata={
            "_interaction_intent": {
                "kind": "follow_up",
                "urgency": "safe_boundary",
                "target": "task",
            }
        },
    )
    built = runtime.build_turn(
        turn=follow_up,
        snapshot=RobotSnapshot(robot_id="mock0"),
        history=[],
        recovery_context=runtime.recovery_context(episode_id),
        context_builder=RobotContextBuilder(),
        injector=RobotTurnInjector(),
    )

    task = runtime.task_runs.load_active(episode_id)
    state = runtime.robot_states.load(episode_id)

    assert task is not None
    assert task.root_task == "pick the cup"
    assert task.status == "active"
    assert state is not None
    assert state.recovery_required is False
    assert built.recovery_context is None
    assert "Continue the active robot task: pick the cup" in built.turn.text
    assert (
        "Recovery was active in the previous turn and has just been resumed."
        in built.turn.text
    )
    assert "Recovery strategy: reobserve" in built.turn.text
    assert (
        "Recommended recovery tools first: get_task_context, request_perception"
        in built.turn.text
    )
    assert (
        "First resolve the recovery step, then continue the original task."
        in built.turn.text
    )
    assert built.turn.metadata["_recovery_resume"]["strategy"] == "reobserve"


def test_task_runtime_new_non_quick_turn_still_supersedes_previous_task(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    episode_id = "ep1"
    runtime.task_runs.ensure_active(
        episode_id=episode_id,
        task="clean the desk",
        agent_id="main",
        robot_id="mock0",
    )
    new_turn = UserTurn(
        envelope=Envelope(
            trace_id="tr3", episode_id=episode_id, agent_id="main", robot_id="mock0"
        ),
        text="bring me the bottle",
    )

    built = runtime.build_turn(
        turn=new_turn,
        snapshot=RobotSnapshot(robot_id="mock0"),
        history=[],
        recovery_context=None,
        context_builder=RobotContextBuilder(),
        injector=RobotTurnInjector(),
    )

    tasks = runtime.task_runs.list_for_episode(episode_id)

    assert built.task is not None
    assert built.task.root_task == "bring me the bottle"
    assert {task.status for task in tasks} == {"active", "cancelled"}
    cancelled = next(task for task in tasks if task.status == "cancelled")
    assert cancelled.root_task == "clean the desk"
    assert cancelled.metadata["superseded_by"] == "bring me the bottle"


def test_task_runtime_result_text_for_agent_includes_recovery_continuation_after_successful_resume_step(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    episode_id = "ep1"
    runtime.task_runs.ensure_active(
        episode_id=episode_id,
        task="pick the cup",
        agent_id="main",
        robot_id="mock0",
    )
    runtime.task_runs.bind_skill_with_metadata(
        episode_id,
        "skill1",
        "inspect the table again",
        metadata={
            "recovery_resume": {
                "strategy": "reobserve",
                "summary": "inspect the table again",
                "continuation_guidance": "Collect fresh perception first, then continue the original task.",
            }
        },
    )
    runtime.record_scene_memory(
        SceneMemoryRecord(
            record_id="scene1",
            episode_id=episode_id,
            robot_id="mock0",
            frame_id=42,
            summary="cup is visible near the table edge",
            metadata={
                "source": "scene_captioner",
                "understanding": {
                    "summary": "cup is visible near the table edge",
                    "risks": ["gripper path partly occluded"],
                    "next_observation_hint": "approach from the right side if you actuate again",
                },
            },
        )
    )
    runtime.robot_states.ensure(episode_id, agent_id="main", robot_id="mock0")
    runtime.robot_states.update_status(
        episode_id,
        RobotStatus(
            envelope=Envelope(
                trace_id="st1", episode_id=episode_id, agent_id="main", robot_id="mock0"
            ),
            state="ready",
        ),
    )
    runtime.robot_states.mark_execution_feedback(
        episode_id,
        skill_id="skill0",
        subgoal_success=False,
        task_success=False,
        summary="previous grasp failed because target pose drifted",
        next_hint="reobserve and then continue with grasp",
    )
    runtime.robot_states.update_observation(
        episode_id,
        RobotObservation(
            envelope=Envelope(
                trace_id="obs1",
                episode_id=episode_id,
                agent_id="main",
                robot_id="mock0",
            ),
            frame_id=43,
        ),
    )
    feedback = ExecutionFeedback(
        skill_id="skill1",
        outcome="confirmed",
        task_success=False,
        subgoal_success=True,
        confidence=0.9,
        summary="inspect_scene completed",
        next_hint="continue with the next useful step",
    )

    text = runtime.result_text_for_agent(
        result=SkillResult(
            envelope=Envelope(
                trace_id="tr1", episode_id=episode_id, agent_id="main", robot_id="mock0"
            ),
            skill_id="skill1",
            status="completed",
            summary="inspect_scene completed",
        ),
        feedback=feedback,
    )

    assert text is not None
    assert "Execution feedback for skill skill1:" in text
    assert "Recovery continuation:" in text
    assert "- original_task: pick the cup" in text
    assert "- recovery_strategy: reobserve" in text
    assert "- resolved_recovery_step: inspect the table again" in text
    assert "- latest_scene_frame: 42" in text
    assert "- latest_scene_summary: cup is visible near the table edge" in text
    assert "- latest_scene_risks: gripper path partly occluded" in text
    assert (
        "- latest_scene_hint: approach from the right side if you actuate again" in text
    )
    assert "- feedback_next_hint: continue with the next useful step" in text
    assert "- runtime_last_observation_frame: 43" in text
    assert "- runtime_robot_state: ready" in text
    assert (
        "- runtime_feedback_summary: previous grasp failed because target pose drifted"
        in text
    )
    assert (
        "- runtime_feedback_next_hint: reobserve and then continue with grasp" in text
    )


def test_task_runtime_result_text_for_agent_includes_normal_task_continuation_after_successful_step(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    episode_id = "ep1"
    runtime.task_runs.ensure_active(
        episode_id=episode_id,
        task="pick the cup and place it into the tray",
        agent_id="main",
        robot_id="mock0",
    )
    runtime.task_runs.bind_skill_with_metadata(
        episode_id,
        "skill0",
        "move closer to the cup",
        metadata={},
    )
    runtime.task_runs.mark_execution_feedback(
        episode_id,
        skill_id="skill0",
        success=True,
        summary="robot moved into grasping range",
    )
    runtime.task_runs.bind_skill_with_metadata(
        episode_id,
        "skill1",
        "grasp the cup",
        metadata={
            "skill": "vla_manipulation",
            "backend": "foundation",
            "implementation_name": "vla_manipulation",
            "implementation_kind": "capability_service",
        },
    )
    runtime.record_scene_memory(
        SceneMemoryRecord(
            record_id="scene2",
            episode_id=episode_id,
            robot_id="mock0",
            frame_id=51,
            summary="cup is now inside the gripper",
            metadata={
                "source": "scene_captioner",
                "understanding": {
                    "summary": "cup is now inside the gripper",
                    "risks": ["tray is still behind the robot base"],
                    "next_observation_hint": "turn toward the tray before placing the cup",
                },
            },
        )
    )
    runtime.robot_states.ensure(episode_id, agent_id="main", robot_id="mock0")
    runtime.robot_states.update_status(
        episode_id,
        RobotStatus(
            envelope=Envelope(
                trace_id="st2", episode_id=episode_id, agent_id="main", robot_id="mock0"
            ),
            state="holding_object",
        ),
    )
    runtime.robot_states.update_observation(
        episode_id,
        RobotObservation(
            envelope=Envelope(
                trace_id="obs2",
                episode_id=episode_id,
                agent_id="main",
                robot_id="mock0",
            ),
            frame_id=52,
        ),
    )
    feedback = ExecutionFeedback(
        skill_id="skill1",
        outcome="confirmed",
        task_success=False,
        subgoal_success=True,
        confidence=0.92,
        summary="grasp completed and object is secure",
        next_hint="move toward the tray and place the cup",
    )

    text = runtime.result_text_for_agent(
        result=SkillResult(
            envelope=Envelope(
                trace_id="tr2", episode_id=episode_id, agent_id="main", robot_id="mock0"
            ),
            skill_id="skill1",
            status="completed",
            summary="grasp completed and object is secure",
        ),
        feedback=feedback,
    )

    assert text is not None
    assert "Execution feedback for skill skill1:" in text
    assert "Task continuation:" in text
    assert "- original_task: pick the cup and place it into the tray" in text
    assert "- completed_step: grasp the cup" in text
    assert "- latest_step_summary: grasp completed and object is secure" in text
    assert "- continuation_focus: move toward the tray and place the cup" in text
    assert "- recent_completed_steps: move closer to the cup" in text
    assert "- latest_scene_frame: 51" in text
    assert "- latest_scene_summary: cup is now inside the gripper" in text
    assert "- latest_scene_risks: tray is still behind the robot base" in text
    assert "- latest_scene_hint: turn toward the tray before placing the cup" in text
    assert "- runtime_last_observation_frame: 52" in text
    assert "- runtime_robot_state: holding_object" in text
    assert "Skill trace:" in text
    assert (
        "vla_manipulation; backend=foundation; implementation=vla_manipulation" in text
    )


def test_task_runtime_result_text_for_agent_keeps_composite_skill_northbound_name(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    episode_id = "ep1"
    runtime.task_runs.ensure_active(
        episode_id=episode_id,
        task="把杯子递给用户",
        agent_id="main",
        robot_id="mock0",
    )
    runtime.task_runs.bind_skill_with_metadata(
        episode_id,
        "skill1",
        "把杯子递给用户",
        metadata={
            "skill": "vla_manipulation",
            "backend": "foundation",
            "implementation_name": "vla_manipulation",
            "implementation_kind": "skill_composite",
        },
    )
    runtime.task_runs.mark_execution_feedback(
        episode_id,
        skill_id="skill1",
        success=True,
        summary="已完成递交动作",
    )
    runtime.robot_states.ensure(episode_id, agent_id="main", robot_id="mock0")
    runtime.robot_states.update_status(
        episode_id,
        RobotStatus(
            envelope=Envelope(
                trace_id="st3", episode_id=episode_id, agent_id="main", robot_id="mock0"
            ),
            state="idle",
        ),
    )
    feedback = ExecutionFeedback(
        skill_id="skill1",
        outcome="confirmed",
        task_success=False,
        subgoal_success=True,
        confidence=0.95,
        summary="已完成递交动作",
        next_hint="确认用户已接稳物体",
    )

    text = runtime.result_text_for_agent(
        result=SkillResult(
            envelope=Envelope(
                trace_id="tr3", episode_id=episode_id, agent_id="main", robot_id="mock0"
            ),
            skill_id="skill1",
            status="completed",
            summary="已完成递交动作",
            metadata={
                "skill": "vla_manipulation",
                "backend": "foundation",
                "implementation_name": "vla_manipulation",
                "implementation_kind": "skill_composite",
            },
        ),
        feedback=feedback,
    )

    assert text is not None
    assert "Skill trace:" in text
    assert (
        "vla_manipulation; backend=foundation; implementation=vla_manipulation" in text
    )
    assert "open_gripper; backend=" not in text
    assert "reset_posture; backend=" not in text


def test_task_runtime_enqueue_and_pop_pending_turn_records_task_events(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    runtime.task_runs.ensure_active(
        episode_id="ep1", task="do it", agent_id="main", robot_id="mock0"
    )
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr1", episode_id="ep1", agent_id="main", robot_id="mock0"
        ),
        text="queued follow up",
    )

    runtime.enqueue_pending_turn(
        turn,
        reason="follow_up",
        intent={"kind": "follow_up"},
        active_skill_id="skill1",
    )
    replayed = runtime.pop_pending_turn("ep1")
    events = runtime.task_runs.events.recent("ep1", limit=10)

    assert replayed is not None
    assert replayed.text == "queued follow up"
    kinds = [event.kind for event in events]
    assert "pending_turn_queued" in kinds
    assert "pending_turn_replayed" in kinds


def test_task_runtime_feedback_target_and_commit_paths_without_episode(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    result = SkillResult(
        envelope=Envelope(robot_id="mock0"), skill_id="skill1", name="inspect_scene"
    )

    target = runtime.feedback_target_for(result)

    assert target.task == "inspect_scene"
    assert target.skill_objective == "inspect_scene"
    assert (
        runtime.observe_execution_feedback(
            episode_id=None, feedback=type("F", (), {})()
        )
        is None
    )
    assert (
        runtime.commit_execution_feedback(
            result=SkillResult(envelope=Envelope(robot_id="mock0"), skill_id="skill1"),
            feedback=type(
                "Feedback",
                (),
                {
                    "subgoal_success": True,
                    "task_success": True,
                    "summary": "ok",
                    "next_hint": None,
                    "successful": True,
                    "to_dict": staticmethod(dict),
                },
            )(),
        )
        is None
    )


def test_task_runtime_observe_turn_result_no_episode_is_noop(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    runtime.observe_turn_result(
        turn=UserTurn(envelope=Envelope(trace_id="tr1"), text="hello"),
        result_tool="final_response",
        reply_text="done",
        task_finished=True,
        skill_id=None,
        last_observation_frame=None,
    )
    assert runtime.checkpoints.list_recent() == []


def test_task_runtime_does_not_mark_task_success_for_intermediate_final_response(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr1", episode_id="ep1", agent_id="main", robot_id="mock0"
        ),
        text="what happened",
    )
    runtime.mark_task_started(turn, agent_id="main")
    runtime.task_runs.ensure_active(
        episode_id="ep1",
        task="what happened",
        agent_id="main",
        robot_id="mock0",
    )

    runtime.observe_turn_result(
        turn=turn,
        result_tool="final_response",
        reply_text="camera image unavailable or degraded",
        task_finished=False,
        skill_id=None,
        last_observation_frame=None,
    )

    task = TaskRunStore(tmp_path).load_active("ep1")
    assert task is not None
    assert task.task_success is None
    assert task.status == "active"


def test_task_runtime_marks_task_success_only_on_explicit_task_finished(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr1", episode_id="ep1", agent_id="main", robot_id="mock0"
        ),
        text="finish the task",
    )
    runtime.mark_task_started(turn, agent_id="main")
    runtime.task_runs.ensure_active(
        episode_id="ep1",
        task="finish the task",
        agent_id="main",
        robot_id="mock0",
    )

    runtime.observe_turn_result(
        turn=turn,
        result_tool="final_response",
        reply_text="done",
        task_finished=True,
        skill_id=None,
        last_observation_frame=None,
    )

    tasks = TaskRunStore(tmp_path).list_for_episode("ep1")
    assert tasks
    task = tasks[0]
    assert task.task_success is True
    assert task.status == "completed"


def test_task_runtime_marks_task_complete_when_explicit_final_response_is_bound_to_skill(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr1", episode_id="ep1", agent_id="main", robot_id="mock0"
        ),
        text="move forward a little",
    )
    runtime.mark_task_started(turn, agent_id="main")
    runtime.task_runs.ensure_active(
        episode_id="ep1",
        task="move forward a little",
        agent_id="main",
        robot_id="mock0",
    )

    runtime.observe_turn_result(
        turn=turn,
        result_tool="final_response",
        reply_text="moved forward a little",
        task_finished=True,
        skill_id="skill1",
        last_observation_frame=None,
    )

    task = TaskRunStore(tmp_path).list_for_episode("ep1")[0]
    assert task.task_success is True
    assert task.status == "completed"
    assert task.skill_ids == ["skill1"]
    assert task.attempts[0].skill_id == "skill1"
    assert task.attempts[0].status == "completed"


def test_task_runtime_keeps_task_active_when_latest_feedback_says_task_not_done(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    episode_id = "ep1"
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr1", episode_id=episode_id, agent_id="main", robot_id="mock0"
        ),
        text="move closer and inspect",
    )
    runtime.mark_task_started(turn, agent_id="main")
    runtime.task_runs.ensure_active(
        episode_id=episode_id,
        task="move closer and inspect",
        agent_id="main",
        robot_id="mock0",
    )
    runtime.task_runs.bind_skill(episode_id, "skill1", "move closer")
    result = SkillResult(
        envelope=Envelope(
            trace_id="tr1", episode_id=episode_id, agent_id="main", robot_id="mock0"
        ),
        skill_id="skill1",
        name="move_base",
        status="completed",
        success=True,
        summary="base moved forward 25.0cm",
    )
    runtime.observe_skill_result(result)
    runtime.commit_execution_feedback(
        result=result,
        feedback=ExecutionFeedback(
            skill_id="skill1",
            outcome="confirmed",
            task_success=False,
            subgoal_success=True,
            confidence=0.75,
            summary="base moved forward 25.0cm; robot_state=idle",
            next_hint="continue with the next useful step",
            recommended_action="continue",
        ),
    )
    internal_reply = (
        "Execution feedback for skill skill1:\n"
        "- outcome: confirmed\n"
        "- subgoal_success: True\n"
        "- task_success: False\n"
        "- recommended_action: continue\n"
        "\n"
        "Task continuation:\n"
        "- remaining_goal: Continue advancing the original task until you can report completion."
    )

    runtime.observe_turn_result(
        turn=turn,
        result_tool="final_response",
        reply_text=internal_reply,
        task_finished=True,
        skill_id="skill1",
        last_observation_frame=None,
    )

    task = TaskRunStore(tmp_path).load_active(episode_id)
    assert task is not None
    assert task.status == "active"
    assert task.task_success is None
    attempt = task.attempts[0]
    assert attempt.metadata["completion_summary"] == "base moved forward 25.0cm"
    assert "Execution feedback for skill" not in task.skill_trace[-1]["summary"]


def test_task_runtime_completes_user_final_response_after_successful_step(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    episode_id = "ep1"
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr1", episode_id=episode_id, agent_id="main", robot_id="mock0"
        ),
        text="你看到了什么",
    )
    runtime.mark_task_started(turn, agent_id="main")
    runtime.task_runs.ensure_active(
        episode_id=episode_id,
        task=turn.text,
        agent_id="main",
        robot_id="mock0",
    )
    runtime.task_runs.bind_skill(episode_id, "skill1", "查看前方场景")
    result = SkillResult(
        envelope=turn.envelope,
        skill_id="skill1",
        name="inspect_scene",
        status="completed",
        success=True,
        summary="scene inspected",
    )
    runtime.observe_skill_result(result)
    runtime.commit_execution_feedback(
        result=result,
        feedback=ExecutionFeedback(
            skill_id="skill1",
            outcome="confirmed",
            task_success=False,
            subgoal_success=True,
            confidence=0.75,
            summary="scene inspected; robot_state=idle",
            recommended_action="continue",
        ),
    )

    runtime.observe_turn_result(
        turn=turn,
        result_tool="final_response",
        reply_text="前方是一张桌子。",
        task_finished=True,
        skill_id="skill1",
        last_observation_frame=None,
    )

    task = TaskRunStore(tmp_path).list_for_episode(episode_id)[0]
    assert task.status == "completed"
    assert task.task_success is True


def test_task_runtime_does_not_create_second_attempt_for_completed_skill_turn(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    episode_id = "ep1"
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr1", episode_id=episode_id, agent_id="main", robot_id="mock0"
        ),
        text="stop now",
    )
    runtime.mark_task_started(turn, agent_id="main")
    runtime.task_runs.ensure_active(
        episode_id=episode_id, task="stop now", agent_id="main", robot_id="mock0"
    )
    runtime.task_runs.bind_skill(episode_id, "skill1", "stop now")
    runtime.task_runs.mark_execution_feedback(
        episode_id, skill_id="skill1", success=True, summary="emergency stop active"
    )

    runtime.observe_turn_result(
        turn=turn,
        result_tool="request_capability",
        reply_text="stopped",
        task_finished=True,
        skill_id="skill1",
        last_observation_frame=None,
    )

    task = TaskRunStore(tmp_path).list_for_episode(episode_id)[0]
    assert len(task.attempts) == 1
    assert task.attempts[0].skill_id == "skill1"
    assert task.attempts[0].success is True
    assert task.status == "completed"
    assert task.task_success is True


# _has_unresolved_recovery


def test_has_unresolved_recovery_returns_true_when_robot_state_recovery_required(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    episode_id = "ep1"
    runtime.robot_states.ensure(episode_id, agent_id="main", robot_id="mock0")
    state = runtime.robot_states.load(episode_id)
    assert state is not None
    state.recovery_required = True
    state.recovery_reason = "camera black frame"
    runtime.robot_states.save(state)

    assert runtime._has_unresolved_recovery(episode_id) is True


def test_has_unresolved_recovery_returns_true_when_last_attempt_failed(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    episode_id = "ep1"
    runtime.task_runs.ensure_active(
        episode_id=episode_id, task="move", agent_id="main", robot_id="mock0"
    )
    runtime.task_runs.bind_skill(episode_id, "skill1", "move_base")
    runtime.task_runs.mark_execution_feedback(
        episode_id, skill_id="skill1", success=False, summary="grasp failed"
    )

    assert runtime._has_unresolved_recovery(episode_id) is True


def test_has_unresolved_recovery_returns_false_when_all_healthy(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    episode_id = "ep1"
    runtime.robot_states.ensure(episode_id, agent_id="main", robot_id="mock0")
    runtime.task_runs.ensure_active(
        episode_id=episode_id, task="move", agent_id="main", robot_id="mock0"
    )
    runtime.task_runs.bind_skill(episode_id, "skill1", "move_base")
    runtime.task_runs.mark_execution_feedback(
        episode_id, skill_id="skill1", success=True, summary="moved forward"
    )

    assert runtime._has_unresolved_recovery(episode_id) is False


def test_has_unresolved_recovery_returns_false_without_active_task_or_state(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    assert runtime._has_unresolved_recovery("nonexistent") is False


# observe_turn_result with recovery-aware success


def test_observe_turn_result_marks_success_false_when_recovery_required(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    episode_id = "ep1"
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr1", episode_id=episode_id, agent_id="main", robot_id="mock0"
        ),
        text="finish",
    )
    runtime.mark_task_started(turn, agent_id="main")
    runtime.task_runs.ensure_active(
        episode_id=episode_id, task="finish", agent_id="main", robot_id="mock0"
    )
    runtime.robot_states.ensure(episode_id, agent_id="main", robot_id="mock0")
    state = runtime.robot_states.load(episode_id)
    assert state is not None
    state.recovery_required = True
    state.recovery_reason = "camera black frame"
    runtime.robot_states.save(state)

    runtime.observe_turn_result(
        turn=turn,
        result_tool="final_response",
        reply_text="done despite camera issue",
        task_finished=True,
        skill_id=None,
        last_observation_frame=None,
    )

    tasks = TaskRunStore(tmp_path).list_for_episode(episode_id)
    assert tasks
    assert tasks[0].task_success is False
    assert tasks[0].status == "failed"


def test_observe_turn_result_marks_success_false_when_last_attempt_failed(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    episode_id = "ep1"
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr1", episode_id=episode_id, agent_id="main", robot_id="mock0"
        ),
        text="finish",
    )
    runtime.mark_task_started(turn, agent_id="main")
    runtime.task_runs.ensure_active(
        episode_id=episode_id, task="finish", agent_id="main", robot_id="mock0"
    )
    runtime.task_runs.bind_skill(episode_id, "skill1", "move_base")
    runtime.task_runs.mark_execution_feedback(
        episode_id, skill_id="skill1", success=False, summary="collision"
    )

    runtime.observe_turn_result(
        turn=turn,
        result_tool="final_response",
        reply_text="done",
        task_finished=True,
        skill_id=None,
        last_observation_frame=None,
    )

    tasks = TaskRunStore(tmp_path).list_for_episode(episode_id)
    assert tasks
    assert tasks[0].task_success is False


def test_observe_turn_result_still_marks_success_true_when_healthy(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    episode_id = "ep1"
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr1", episode_id=episode_id, agent_id="main", robot_id="mock0"
        ),
        text="finish healthy",
    )
    runtime.mark_task_started(turn, agent_id="main")
    runtime.task_runs.ensure_active(
        episode_id=episode_id, task="finish healthy", agent_id="main", robot_id="mock0"
    )
    runtime.task_runs.bind_skill(episode_id, "skill1", "move_base")
    runtime.task_runs.mark_execution_feedback(
        episode_id, skill_id="skill1", success=True, summary="ok"
    )

    runtime.observe_turn_result(
        turn=turn,
        result_tool="final_response",
        reply_text="all done",
        task_finished=True,
        skill_id=None,
        last_observation_frame=None,
    )

    tasks = TaskRunStore(tmp_path).list_for_episode(episode_id)
    assert tasks
    assert tasks[0].task_success is True


def test_task_runtime_keeps_skill_bound_final_response_active_without_task_finished(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    turn = UserTurn(
        envelope=Envelope(
            trace_id="tr1", episode_id="ep1", agent_id="main", robot_id="mock0"
        ),
        text="move forward a little",
    )
    runtime.mark_task_started(turn, agent_id="main")
    runtime.task_runs.ensure_active(
        episode_id="ep1",
        task="move forward a little",
        agent_id="main",
        robot_id="mock0",
    )

    runtime.observe_turn_result(
        turn=turn,
        result_tool="final_response",
        reply_text="moved forward a little",
        task_finished=False,
        skill_id="skill1",
        last_observation_frame=None,
    )

    task = TaskRunStore(tmp_path).load_active("ep1")
    assert task is not None
    assert task.task_success is None
    assert task.status == "active"
