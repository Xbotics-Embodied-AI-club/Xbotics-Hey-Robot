from __future__ import annotations

import asyncio

import numpy as np

from hey_robot.agents.execution_feedback import (
    DefaultExecutionFeedbackEvaluator,
    ExecutionFeedback,
    VisionExecutionFeedbackEvaluator,
    _feedback_from_parsed,
    _resolve_images,
    status_feedback_from_result,
)
from hey_robot.agents.robot_agent import RobotAgentService
from hey_robot.agents.types import RobotSnapshot
from hey_robot.config import DeploymentConfig
from hey_robot.protocol import (
    Envelope,
    ImageRef,
    RobotObservation,
    RobotStatus,
    SkillResult,
)
from hey_robot.providers import ReasoningMessage, ReasoningResponse


class FakeVisionProvider:
    def __init__(self) -> None:
        self.messages: list[ReasoningMessage] = []

    async def chat(self, **kwargs):
        self.messages = list(kwargs.get("messages") or [])
        return ReasoningResponse(
            content=(
                '{"subgoal_success":true,"task_success":false,"summary":"gripper opened",'
                '"failure_reason":null,"next_hint":"move closer to the target","confidence":0.76}'
            )
        )

    def get_default_model(self) -> str:
        return "fake-vision"


class FakeImageResolver:
    def resolve_images(self, _images):
        return [np.zeros((8, 8, 3), dtype=np.uint8)]


def test_status_feedback_uses_robot_success_metric() -> None:
    result = SkillResult(
        envelope=Envelope(robot_id="mock0"), skill_id="cmd1", status="completed"
    )
    status = RobotStatus(
        envelope=Envelope(robot_id="mock0"), state="terminated", success=True
    )

    feedback = status_feedback_from_result(result, status)

    assert feedback.successful is True
    assert feedback.task_success is True
    assert feedback.outcome == "confirmed"
    assert feedback.recommended_action == "report_or_continue"


def test_agent_service_commits_execution_feedback(tmp_path) -> None:
    config = DeploymentConfig.from_dict(
        {
            "agents": {
                "main": {
                    "type": "robot_agent",
                    "robot_id": "mock0",
                    "settings": {
                        "mode": "direct",
                        "execution_feedback": {"backend": "status"},
                    },
                }
            },
            "robots": {"mock0": {"type": "mock"}},
        }
    )
    service = RobotAgentService(config, agent_id="main", episode_dir=tmp_path)
    service.latest_status["mock0"] = RobotStatus(
        envelope=Envelope(robot_id="mock0"),
        state="terminated",
        success=True,
    )
    service.task_runtime.task_runs.ensure_active(
        episode_id="s1", task="move", agent_id="main", robot_id="mock0"
    )
    service.task_runtime.task_runs.bind_skill("s1", "cmd1", "move")
    result = SkillResult(
        envelope=Envelope(
            trace_id="tr1", episode_id="s1", agent_id="main", robot_id="mock0"
        ),
        skill_id="cmd1",
        status="completed",
    )

    feedback = asyncio.run(service._evaluate_execution_feedback(result))
    task = asyncio.run(service._commit_execution_feedback(result, feedback))

    assert isinstance(feedback, ExecutionFeedback)
    assert task is not None
    assert task.status == "active"
    assert task.task_success is None


async def test_vision_execution_feedback_evaluator_parses_progress_json() -> None:
    provider = FakeVisionProvider()
    evaluator = VisionExecutionFeedbackEvaluator(
        provider, image_resolver=FakeImageResolver()
    )  # type: ignore[arg-type]
    result = SkillResult(
        envelope=Envelope(robot_id="xlerobot"), skill_id="cmd1", status="completed"
    )
    snapshot = RobotSnapshot(
        robot_id="xlerobot",
        observation=RobotObservation(
            envelope=Envelope(episode_id="s1", robot_id="xlerobot"),
            frame_id=2,
            images=[ImageRef(uri="media://image")],
        ),
    )

    feedback = await evaluator.evaluate(
        task="pick up the cup",
        skill_objective="open the gripper",
        result=result,
        snapshot=snapshot,
        mode="vision",
    )

    assert feedback.subgoal_success is True
    assert feedback.task_success is False
    assert feedback.summary == "gripper opened"
    assert feedback.next_hint == "move closer to the target"
    assert feedback.confidence == 0.76
    assert "embodied execution feedback" in provider.messages[0].content
    assert "Overall task: pick up the cup" in provider.messages[1].content


# ExecutionFeedback: successful property.


def test_execution_feedback_successful_true() -> None:
    fb = ExecutionFeedback(
        skill_id="s1",
        outcome="confirmed",
        task_success=True,
        subgoal_success=True,
        confidence=1.0,
        summary="ok",
    )
    assert fb.successful is True


def test_execution_feedback_successful_skipped_with_subgoal_success() -> None:
    fb = ExecutionFeedback(
        skill_id="s1",
        outcome="skipped",
        task_success=False,
        subgoal_success=True,
        confidence=1.0,
        summary="skipped",
    )
    assert fb.successful is True


def test_execution_feedback_successful_false_on_failed_outcome() -> None:
    fb = ExecutionFeedback(
        skill_id="s1",
        outcome="failed",
        task_success=False,
        subgoal_success=False,
        confidence=1.0,
        summary="fail",
    )
    assert fb.successful is False


def test_execution_feedback_successful_false_when_subgoal_not_success() -> None:
    fb = ExecutionFeedback(
        skill_id="s1",
        outcome="confirmed",
        task_success=False,
        subgoal_success=False,
        confidence=1.0,
        summary="hmm",
    )
    assert fb.successful is False


# ExecutionFeedback: for_agent.


def test_execution_feedback_for_agent_minimal() -> None:
    fb = ExecutionFeedback(
        skill_id="cmd1",
        outcome="confirmed",
        task_success=True,
        subgoal_success=True,
        confidence=1.0,
        summary="done",
    )
    text = fb.for_agent()
    assert "cmd1" in text
    assert "confirmed" in text
    assert "subgoal_success: True" in text
    assert "task_success: True" in text
    assert "done" in text


def test_execution_feedback_for_agent_with_confidence_and_failure() -> None:
    fb = ExecutionFeedback(
        skill_id="cmd2",
        outcome="failed",
        task_success=False,
        subgoal_success=False,
        confidence=0.42,
        summary="bad",
        failure_reason="collision",
        next_hint="retry with offset",
    )
    text = fb.for_agent()
    assert "confidence: 0.42" in text
    assert "failure_reason: collision" in text
    assert "next_hint: retry with offset" in text
    assert "recommended_action" not in text


# status_feedback_from_result: edge cases.


def test_status_feedback_non_completed_status() -> None:
    result = SkillResult(
        envelope=Envelope(robot_id="mock0"),
        skill_id="cmd1",
        status="interrupted",
        error="timeout",
    )
    feedback = status_feedback_from_result(result, None)

    assert feedback.outcome == "failed"
    assert feedback.task_success is False
    assert feedback.subgoal_success is False
    assert feedback.confidence == 1.0
    assert feedback.summary == "timeout"


def test_status_feedback_completed_without_status() -> None:
    result = SkillResult(
        envelope=Envelope(robot_id="mock0"),
        skill_id="cmd1",
        status="completed",
        summary="ok",
    )
    feedback = status_feedback_from_result(result, None)

    assert feedback.outcome == "confirmed"
    assert feedback.task_success is False
    assert feedback.subgoal_success is True
    assert feedback.confidence == 0.6
    assert feedback.next_hint == "continue with the next useful step"


def test_status_feedback_completed_with_unsuccessful_get_robot_status() -> None:
    result = SkillResult(
        envelope=Envelope(robot_id="mock0"), skill_id="cmd1", status="completed"
    )
    status = RobotStatus(
        envelope=Envelope(robot_id="mock0"),
        state="error",
        success=False,
        error="arm stuck",
    )
    feedback = status_feedback_from_result(result, status)

    assert feedback.outcome == "failed"
    assert feedback.task_success is False
    assert feedback.subgoal_success is False
    assert feedback.confidence == 0.9
    assert "arm stuck" in feedback.summary
    assert feedback.next_hint == "inspect the scene and choose the next action"
    assert feedback.recommended_action == "reobserve"


def test_status_feedback_completed_status_with_state_text_appends_robot_state() -> None:
    result = SkillResult(
        envelope=Envelope(robot_id="mock0"),
        skill_id="cmd1",
        status="completed",
        summary="done",
    )
    status = RobotStatus(
        envelope=Envelope(robot_id="mock0"), state="idle", success=True
    )
    feedback = status_feedback_from_result(result, status)

    assert "robot_state=idle" in feedback.summary
    assert feedback.task_success is True


def test_status_feedback_completed_status_success_none() -> None:
    result = SkillResult(
        envelope=Envelope(robot_id="mock0"), skill_id="cmd1", status="completed"
    )
    status = RobotStatus(envelope=Envelope(robot_id="mock0"), state="idle")
    feedback = status_feedback_from_result(result, status)

    assert feedback.task_success is False
    assert feedback.confidence == 0.75


# _feedback_from_parsed.


def test_feedback_from_parsed_subgoal_failure() -> None:
    result = SkillResult(
        envelope=Envelope(robot_id="mock0"), skill_id="cmd1", status="completed"
    )
    fb = _feedback_from_parsed(
        result,
        False,
        False,
        "grasp failed",
        "retry",
        failure_reason="slipped",
        metadata={"m": 1},
    )

    assert fb.outcome == "failed"
    assert fb.subgoal_success is False
    assert fb.failure_reason == "slipped"
    assert fb.next_hint == "retry"
    assert fb.recommended_action == "recover"


def test_feedback_from_parsed_subgoal_success_clears_failure_reason() -> None:
    result = SkillResult(
        envelope=Envelope(robot_id="mock0"), skill_id="cmd1", status="completed"
    )
    fb = _feedback_from_parsed(
        result, True, True, "ok", None, failure_reason="ignored", metadata={}
    )

    assert fb.outcome == "confirmed"
    assert fb.failure_reason is None
    assert fb.recommended_action == "continue"


# _resolve_images.


def test_resolve_images_with_none_resolver() -> None:
    from hey_robot.agents.types import RobotSnapshot

    snapshot = RobotSnapshot(
        robot_id="r1",
        observation=RobotObservation(
            envelope=Envelope(episode_id="s1", robot_id="r1"),
            frame_id=1,
            images=[ImageRef(uri="media://img")],
        ),
    )
    assert _resolve_images(snapshot, None) == []


def test_resolve_images_with_none_observation() -> None:
    from hey_robot.agents.types import RobotSnapshot

    snapshot = RobotSnapshot(robot_id="r1")
    assert _resolve_images(snapshot, FakeImageResolver()) == []


def test_resolve_images_with_valid_resolver() -> None:
    from hey_robot.agents.types import RobotSnapshot

    snapshot = RobotSnapshot(
        robot_id="r1",
        observation=RobotObservation(
            envelope=Envelope(episode_id="s1", robot_id="r1"),
            frame_id=1,
            images=[ImageRef(uri="media://img")],
        ),
    )
    result = _resolve_images(snapshot, FakeImageResolver())
    assert len(result) == 1


# DefaultExecutionFeedbackEvaluator: evaluate modes.


async def test_default_evaluator_mode_none() -> None:
    evaluator = DefaultExecutionFeedbackEvaluator()
    result = SkillResult(
        envelope=Envelope(robot_id="mock0"),
        skill_id="cmd1",
        status="completed",
        summary="done",
    )
    snapshot = RobotSnapshot(robot_id="mock0")

    feedback = await evaluator.evaluate(
        task="t", skill_objective="o", result=result, snapshot=snapshot, mode="none"
    )

    assert feedback.outcome == "skipped"
    assert feedback.subgoal_success is True
    assert feedback.task_success is False
    assert feedback.confidence == 1.0


async def test_default_evaluator_non_completed_result() -> None:
    evaluator = DefaultExecutionFeedbackEvaluator()
    result = SkillResult(
        envelope=Envelope(robot_id="mock0"),
        skill_id="cmd1",
        status="failed",
        summary="crash",
        error="motor failure",
    )
    snapshot = RobotSnapshot(robot_id="mock0")

    feedback = await evaluator.evaluate(
        task="t", skill_objective="o", result=result, snapshot=snapshot, mode="status"
    )

    assert feedback.outcome == "failed"
    assert feedback.subgoal_success is False
    assert "motor failure" in feedback.failure_reason  # type: ignore[operator]


async def test_default_evaluator_mode_operator() -> None:
    evaluator = DefaultExecutionFeedbackEvaluator()
    result = SkillResult(
        envelope=Envelope(robot_id="mock0"), skill_id="cmd1", status="completed"
    )
    snapshot = RobotSnapshot(robot_id="mock0")

    feedback = await evaluator.evaluate(
        task="t", skill_objective="o", result=result, snapshot=snapshot, mode="operator"
    )

    assert feedback.outcome == "unknown"
    assert feedback.subgoal_success is None
    assert feedback.confidence is None
    assert "operator confirmation" in feedback.summary


# _camera_quality_issue_from_status and _camera_quality_issue_from_snapshot


def test_camera_quality_issue_from_status_detects_black_frame() -> None:
    from hey_robot.agents.execution_feedback import _camera_quality_issue_from_status

    status = RobotStatus(
        envelope=Envelope(robot_id="mock0"),
        state="terminated",
        success=True,
        metrics={
            "camera": {
                "ok": False,
                "frame_available": False,
                "valid_image_count": 0,
                "image_quality_issues": ["black_frame"],
                "age_ms": 28969,
            }
        },
    )
    issue = _camera_quality_issue_from_status(status)
    assert issue is not None
    assert "ok=False" in issue
    assert "black_frame" in issue


def test_camera_quality_issue_from_status_detects_no_valid_images() -> None:
    from hey_robot.agents.execution_feedback import _camera_quality_issue_from_status

    status = RobotStatus(
        envelope=Envelope(robot_id="mock0"),
        state="idle",
        metrics={
            "camera": {
                "ok": True,
                "frame_available": True,
                "valid_image_count": 0,
                "image_quality_issues": [],
            }
        },
    )
    issue = _camera_quality_issue_from_status(status)
    assert issue is not None
    assert "valid_image_count=0" in issue


def test_camera_quality_issue_from_status_detects_frame_not_available() -> None:
    from hey_robot.agents.execution_feedback import _camera_quality_issue_from_status

    status = RobotStatus(
        envelope=Envelope(robot_id="mock0"),
        state="idle",
        metrics={
            "camera": {"ok": True, "frame_available": False, "valid_image_count": 5}
        },
    )
    issue = _camera_quality_issue_from_status(status)
    assert issue is not None
    assert "frame not available" in issue


def test_camera_quality_issue_from_status_returns_none_for_healthy_camera() -> None:
    from hey_robot.agents.execution_feedback import _camera_quality_issue_from_status

    status = RobotStatus(
        envelope=Envelope(robot_id="mock0"),
        state="idle",
        metrics={
            "camera": {"ok": True, "frame_available": True, "valid_image_count": 3}
        },
    )
    assert _camera_quality_issue_from_status(status) is None


def test_camera_quality_issue_from_status_returns_none_without_camera_metrics() -> None:
    from hey_robot.agents.execution_feedback import _camera_quality_issue_from_status

    status = RobotStatus(
        envelope=Envelope(robot_id="mock0"),
        state="idle",
        metrics={"battery": {"percentage": 85}},
    )
    assert _camera_quality_issue_from_status(status) is None


def test_camera_quality_issue_from_status_returns_none_for_none_status() -> None:
    from hey_robot.agents.execution_feedback import _camera_quality_issue_from_status

    assert _camera_quality_issue_from_status(None) is None


def test_camera_quality_issue_from_snapshot_delegates_to_status() -> None:
    from hey_robot.agents.execution_feedback import _camera_quality_issue_from_snapshot

    status = RobotStatus(
        envelope=Envelope(robot_id="mock0"),
        state="idle",
        metrics={
            "camera": {
                "ok": False,
                "frame_available": False,
                "valid_image_count": 0,
                "image_quality_issues": ["black_frame"],
            }
        },
    )
    snapshot = RobotSnapshot(robot_id="mock0", status=status)
    issue = _camera_quality_issue_from_snapshot(snapshot)
    assert issue is not None
    assert "black_frame" in issue


def test_camera_quality_issue_from_snapshot_returns_none_without_status() -> None:
    from hey_robot.agents.execution_feedback import _camera_quality_issue_from_snapshot

    snapshot = RobotSnapshot(robot_id="mock0")
    assert _camera_quality_issue_from_snapshot(snapshot) is None


# status_feedback_from_result: camera quality


def test_status_feedback_fails_on_camera_quality_issue() -> None:
    result = SkillResult(
        envelope=Envelope(robot_id="mock0"),
        skill_id="cmd1",
        status="completed",
        summary="ok",
    )
    status = RobotStatus(
        envelope=Envelope(robot_id="mock0"),
        state="terminated",
        success=True,
        metrics={
            "camera": {
                "ok": False,
                "frame_available": False,
                "valid_image_count": 0,
                "image_quality_issues": ["black_frame"],
                "age_ms": 30000,
            }
        },
    )
    feedback = status_feedback_from_result(result, status)
    assert feedback.outcome == "failed"
    assert feedback.subgoal_success is False
    assert "perception degraded" in feedback.summary
    assert "black_frame" in feedback.failure_reason  # type: ignore[operator]
    assert feedback.recommended_action == "reobserve"


def test_status_feedback_succeeds_when_camera_is_healthy() -> None:
    result = SkillResult(
        envelope=Envelope(robot_id="mock0"),
        skill_id="cmd1",
        status="completed",
        summary="ok",
    )
    status = RobotStatus(
        envelope=Envelope(robot_id="mock0"),
        state="idle",
        success=True,
        metrics={
            "camera": {"ok": True, "frame_available": True, "valid_image_count": 3}
        },
    )
    feedback = status_feedback_from_result(result, status)
    assert feedback.outcome == "confirmed"
    assert feedback.subgoal_success is True


# DefaultExecutionFeedbackEvaluator: mode="none" with camera issues


async def test_default_evaluator_mode_none_blocks_on_camera_quality_issue() -> None:
    evaluator = DefaultExecutionFeedbackEvaluator()
    result = SkillResult(
        envelope=Envelope(robot_id="mock0"),
        skill_id="cmd1",
        status="completed",
        summary="camera ok",
    )
    status = RobotStatus(
        envelope=Envelope(robot_id="mock0"),
        state="idle",
        metrics={
            "camera": {
                "ok": False,
                "frame_available": False,
                "valid_image_count": 0,
                "image_quality_issues": ["black_frame"],
            }
        },
    )
    snapshot = RobotSnapshot(robot_id="mock0", status=status)

    feedback = await evaluator.evaluate(
        task="t", skill_objective="o", result=result, snapshot=snapshot, mode="none"
    )

    assert feedback.outcome == "failed"
    assert feedback.subgoal_success is False
    assert "perception degraded" in feedback.summary
    assert feedback.recommended_action == "reobserve"


async def test_default_evaluator_mode_none_returns_skipped_when_camera_healthy() -> (
    None
):
    evaluator = DefaultExecutionFeedbackEvaluator()
    result = SkillResult(
        envelope=Envelope(robot_id="mock0"),
        skill_id="cmd1",
        status="completed",
        summary="done",
    )
    status = RobotStatus(
        envelope=Envelope(robot_id="mock0"),
        state="idle",
        metrics={
            "camera": {"ok": True, "frame_available": True, "valid_image_count": 3}
        },
    )
    snapshot = RobotSnapshot(robot_id="mock0", status=status)

    feedback = await evaluator.evaluate(
        task="t", skill_objective="o", result=result, snapshot=snapshot, mode="none"
    )

    assert feedback.outcome == "skipped"
    assert feedback.subgoal_success is True


async def test_default_evaluator_mode_status_falls_through() -> None:
    evaluator = DefaultExecutionFeedbackEvaluator()
    result = SkillResult(
        envelope=Envelope(robot_id="mock0"),
        skill_id="cmd1",
        status="completed",
        summary="ok",
    )
    snapshot = RobotSnapshot(robot_id="mock0")

    feedback = await evaluator.evaluate(
        task="t", skill_objective="o", result=result, snapshot=snapshot, mode="status"
    )

    assert feedback.outcome == "confirmed"
    assert feedback.subgoal_success is True
