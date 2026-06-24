from __future__ import annotations

from hey_robot.agents.task_run import TaskRun
from hey_robot.agents.task_runtime import RecoveryManager
from hey_robot.protocol import Envelope, RobotStatus, SkillResult
from hey_robot.tasks.recovery import TaskRecoveryPlanner


def test_task_recovery_planner_maps_critical_battery_to_safe_abort() -> None:
    planner = TaskRecoveryPlanner()
    decision = planner.decide(
        task=None,
        result=SkillResult(
            envelope=Envelope(robot_id="mock0"), skill_id="cmd1", status="failed"
        ),
        status=RobotStatus(
            envelope=Envelope(robot_id="mock0"),
            state="unsafe",
            metrics={"battery": {"status": "critical", "percentage": 4}},
        ),
    )

    assert decision.needed is True
    assert decision.strategy == "safe_abort"
    assert decision.operator_required is True
    assert "stop_motion" in decision.actions


def test_task_recovery_planner_maps_target_not_visible_to_reposition() -> None:
    planner = TaskRecoveryPlanner()
    decision = planner.decide(
        task=None,
        result=SkillResult(
            envelope=Envelope(robot_id="mock0"),
            skill_id="cmd1",
            status="failed",
            failure_mode="target_not_visible",
            summary="cup not visible",
        ),
        status=None,
    )

    assert decision.needed is True
    assert decision.strategy == "reposition"
    assert "base_reposition" in decision.actions
    assert "inspect_scene" in decision.actions


def test_task_recovery_planner_maps_resource_busy_to_clarify() -> None:
    planner = TaskRecoveryPlanner()
    decision = planner.decide(
        task=None,
        result=SkillResult(
            envelope=Envelope(robot_id="mock0"),
            skill_id="cmd1",
            status="failed",
            failure_mode="resource_busy",
            error="arm controller busy",
        ),
        status=None,
    )

    assert decision.needed is True
    assert decision.strategy == "clarify"
    assert decision.operator_required is True
    assert "request_clarification" in decision.actions


def test_task_recovery_planner_marks_unknown_pose_as_non_retryable_parameter_error() -> (
    None
):
    planner = TaskRecoveryPlanner()
    decision = planner.decide(
        task=None,
        result=SkillResult(
            envelope=Envelope(robot_id="mock0"),
            skill_id="cmd1",
            status="failed",
            failure_mode="execution_failed",
            summary="unknown named pose: pre_grasp",
        ),
        status=None,
    )

    assert decision.needed is True
    assert decision.strategy == "clarify"
    assert decision.retryable is False
    assert decision.metadata["failure_class"] == "parameter_error"
    assert "预抓取" in decision.summary
    assert "直接重试" in decision.summary


def test_task_recovery_planner_marks_invalid_joint_as_parameter_error() -> None:
    planner = TaskRecoveryPlanner()
    decision = planner.decide(
        task=None,
        result=SkillResult(
            envelope=Envelope(robot_id="mock0"),
            skill_id="cmd1",
            status="failed",
            failure_mode="invalid_joint",
            error="unknown joint: wrist_yaw",
        ),
        status=None,
    )

    assert decision.needed is True
    assert decision.strategy == "clarify"
    assert decision.retryable is False
    assert decision.metadata["failure_class"] == "parameter_error"
    assert "wrist_yaw" in decision.summary


def test_task_recovery_planner_marks_capability_unavailable_as_non_retryable() -> None:
    planner = TaskRecoveryPlanner()
    decision = planner.decide(
        task=None,
        result=SkillResult(
            envelope=Envelope(robot_id="mock0"),
            skill_id="cmd1",
            name="human_follow",
            status="failed",
            failure_mode="capability_not_available",
            error="capability not available: human_follow",
        ),
        status=None,
    )

    assert decision.needed is True
    assert decision.strategy == "clarify"
    assert decision.retryable is False
    assert decision.metadata["failure_class"] == "capability_unavailable"
    assert "human_follow" in decision.summary


def test_task_recovery_planner_uses_scene_risk_for_reobserve() -> None:
    planner = TaskRecoveryPlanner()
    task = TaskRun(
        task_id="task1",
        episode_id="ep1",
        root_task="inspect the cup",
        metadata={
            "latest_scene_event": {
                "metadata": {
                    "understanding": {
                        "risks": ["missing image from front camera"],
                        "next_observation_hint": "Get a fresh observation.",
                    }
                }
            }
        },
    )

    decision = planner.decide(
        task=task,
        result=SkillResult(
            envelope=Envelope(robot_id="mock0"), skill_id="cmd1", status="failed"
        ),
        status=None,
    )

    assert decision.needed is True
    assert decision.strategy == "reobserve"
    assert "inspect_scene" in decision.actions


def test_task_recovery_planner_references_matching_health_report() -> None:
    planner = TaskRecoveryPlanner()
    health_report = {
        "component": "robot.mock0.camera",
        "status": "failed",
        "severity": "warning",
        "evidence": "OpenCV device opened but no frame",
        "impacted_skills": ["human_follow"],
        "fix_hint": "Run camera scan.",
    }

    decision = planner.decide(
        task=None,
        result=SkillResult(
            envelope=Envelope(robot_id="mock0"),
            skill_id="cmd1",
            name="human_follow",
            status="failed",
            failure_mode="camera_unavailable",
        ),
        status=None,
        health_reports=(health_report,),
    )

    assert decision.needed is True
    assert decision.strategy == "reobserve"
    assert "robot.mock0.camera" in decision.summary
    assert decision.metadata["health_report"] == health_report


class TestRecoveryManagerEscalation:
    @staticmethod
    def _failed_result(failure_mode: str = "camera_unavailable") -> SkillResult:
        return SkillResult(
            envelope=Envelope(robot_id="mock0"),
            skill_id="cmd1",
            status="failed",
            failure_mode=failure_mode,
            summary="skill failed",
        )

    @staticmethod
    def _task() -> TaskRun:
        return TaskRun(
            task_id="task1",
            episode_id="ep1",
            root_task="向后移动",
        )

    def test_first_failure_returns_planner_decision(self) -> None:
        mgr = RecoveryManager()
        task = self._task()
        result = self._failed_result()

        decision = mgr.decide_for_skill_result(task=task, result=result, status=None)

        assert decision.needed is True
        assert decision.strategy == "reobserve"

    def test_second_same_failure_returns_planner_decision(self) -> None:
        mgr = RecoveryManager()
        task = self._task()
        result = self._failed_result()

        mgr.decide_for_skill_result(task=task, result=result, status=None)
        decision = mgr.decide_for_skill_result(task=task, result=result, status=None)

        assert decision.needed is True
        assert decision.strategy == "reobserve"

    def test_third_same_failure_escalates_to_ask_operator(self) -> None:
        mgr = RecoveryManager()
        task = self._task()
        result = self._failed_result()

        for _ in range(2):
            mgr.decide_for_skill_result(task=task, result=result, status=None)
        decision = mgr.decide_for_skill_result(task=task, result=result, status=None)

        assert decision.needed is True
        assert decision.strategy == "ask_operator"
        assert decision.operator_required is True
        assert decision.retryable is True
        assert decision.metadata["escalated_from"] == "reobserve"
        assert decision.metadata["recovery_attempts"] == 3

    def test_fifth_same_failure_escalates_to_safe_abort(self) -> None:
        mgr = RecoveryManager()
        task = self._task()
        result = self._failed_result()

        for _ in range(4):
            mgr.decide_for_skill_result(task=task, result=result, status=None)
        decision = mgr.decide_for_skill_result(task=task, result=result, status=None)

        assert decision.needed is True
        assert decision.strategy == "safe_abort"
        assert decision.operator_required is True
        assert decision.retryable is False
        assert "stop_motion" in decision.actions

    def test_different_failure_resets_counter(self) -> None:
        mgr = RecoveryManager()
        task = self._task()

        mgr.decide_for_skill_result(
            task=task, result=self._failed_result("camera_unavailable"), status=None
        )
        mgr.decide_for_skill_result(
            task=task, result=self._failed_result("camera_unavailable"), status=None
        )
        decision = mgr.decide_for_skill_result(
            task=task, result=self._failed_result("base_motion_failed"), status=None
        )

        assert decision.strategy != "ask_operator"
        assert mgr._attempts["ep1"] == 1

    def test_clear_resets_counter(self) -> None:
        mgr = RecoveryManager()
        task = self._task()
        result = self._failed_result()

        mgr.decide_for_skill_result(task=task, result=result, status=None)
        mgr.decide_for_skill_result(task=task, result=result, status=None)
        assert mgr._attempts.get("ep1") == 2

        mgr.clear("ep1")
        assert mgr._attempts.get("ep1") is None

        decision = mgr.decide_for_skill_result(task=task, result=result, status=None)
        assert decision.strategy == "reobserve"
        assert mgr._attempts["ep1"] == 1

    def test_non_needed_decision_does_not_count(self) -> None:
        mgr = RecoveryManager()
        task = self._task()
        result = SkillResult(
            envelope=Envelope(robot_id="mock0"),
            skill_id="cmd1",
            status="completed",
            summary="all good",
        )

        mgr.decide_for_skill_result(task=task, result=result, status=None)
        mgr.decide_for_skill_result(task=task, result=result, status=None)

        assert mgr._attempts.get("ep1") is None

    def test_no_task_no_escalation_tracking(self) -> None:
        mgr = RecoveryManager()
        result = self._failed_result()

        for _ in range(5):
            decision = mgr.decide_for_skill_result(
                task=None, result=result, status=None
            )

        assert decision.strategy == "reobserve"
        assert not mgr._attempts
