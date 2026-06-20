from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hey_robot.agents.task_run import TaskRun
from hey_robot.protocol import RobotStatus, SkillResult


class RecoveryStrategy(StrEnum):
    NONE = "none"
    AWAIT_EXECUTION_FEEDBACK = "await_execution_feedback"
    CLARIFY = "clarify"
    REOBSERVE = "reobserve"
    REPOSITION = "reposition"
    RETRY_WITH_ADJUSTMENT = "retry_with_adjustment"
    SAFE_ABORT = "safe_abort"
    ASK_OPERATOR = "ask_operator"
    DEGRADED_CONTINUE = "degraded_continue"


class RecoveryAction(StrEnum):
    HOLD_TASK = "hold_task"
    BASE_STOP = "stop_motion"
    INSPECT_SCENE = "inspect_scene"
    REPOSITION_BASE = "base_reposition"
    VERIFY_RESULT = "inspect_scene"
    DECIDE_NEXT_STEP = "decide_next_step"
    ASK_OPERATOR = "ask_operator"
    REQUEST_CLARIFICATION = "request_clarification"
    RETRY_SKILL = "retry_skill"
    DEGRADE_RESOURCE = "degrade_resource"


@dataclass(frozen=True)
class RecoveryPlaybook:
    strategy: RecoveryStrategy
    summary: str
    severity: str = "info"
    actions: tuple[RecoveryAction, ...] = ()
    operator_required: bool = False
    retryable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def decision(self, *, needed: bool = True) -> TaskRecoveryDecision:
        return TaskRecoveryDecision(
            needed=needed,
            strategy=self.strategy.value,
            summary=self.summary,
            severity=self.severity,
            actions=tuple(action.value for action in self.actions),
            operator_required=self.operator_required,
            retryable=self.retryable,
            metadata={
                **self.metadata,
                "playbook": {
                    "strategy": self.strategy.value,
                    "actions": [action.value for action in self.actions],
                    "operator_required": self.operator_required,
                    "retryable": self.retryable,
                },
            },
        )


@dataclass(frozen=True)
class TaskRecoveryDecision:
    needed: bool
    strategy: str
    summary: str
    severity: str = "info"
    actions: tuple[str, ...] = ()
    operator_required: bool = False
    retryable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def block_actuation(self) -> bool:
        """Whether this recovery decision blocks new actuation skills.

        Most recovery strategies block actuation until resolved. Only
        ``degraded_continue`` and ``none`` allow the agent to proceed
        without first resolving the recovery condition.
        """
        if not self.needed:
            return False
        return self.strategy not in {
            RecoveryStrategy.DEGRADED_CONTINUE.value,
            RecoveryStrategy.NONE.value,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "needed": self.needed,
            "strategy": self.strategy,
            "summary": self.summary,
            "severity": self.severity,
            "actions": list(self.actions),
            "operator_required": self.operator_required,
            "retryable": self.retryable,
            "block_actuation": self.block_actuation,
            "metadata": self.metadata,
        }


class TaskRecoveryPlanner:
    """Deterministic recovery classifier for long-running robot tasks."""

    def decide(
        self,
        *,
        task: TaskRun | None,
        result: SkillResult,
        status: RobotStatus | None = None,
        health_reports: tuple[dict[str, Any], ...] = (),
    ) -> TaskRecoveryDecision:
        health_decision = self._health_recovery(result, health_reports)
        if health_decision is not None:
            return health_decision
        battery = status.metrics.get("battery") if status is not None else None
        if isinstance(battery, dict) and battery.get("status") == "critical":
            return RecoveryPlaybook(
                RecoveryStrategy.SAFE_ABORT,
                "Battery is critical; stop autonomous execution and ask the operator to recover power first.",
                severity="operator_required",
                actions=(
                    RecoveryAction.HOLD_TASK,
                    RecoveryAction.BASE_STOP,
                    RecoveryAction.ASK_OPERATOR,
                ),
                operator_required=True,
                retryable=False,
                metadata={"battery": battery},
            ).decision()
        if result.status in {"failed", "interrupted"}:
            scene_decision = self._scene_recovery(task)
            if scene_decision is not None:
                return scene_decision
            return self._skill_failure_recovery(result)
        if (
            result.status == "completed"
            and task is not None
            and task.status == "feedback_pending"
        ):
            return RecoveryPlaybook(
                RecoveryStrategy.AWAIT_EXECUTION_FEEDBACK,
                "Skill completed; execution feedback is being committed before the next skill.",
                severity="info",
                actions=(RecoveryAction.VERIFY_RESULT,),
                metadata={"skill_id": result.skill_id},
            ).decision(needed=False)
        return RecoveryPlaybook(
            RecoveryStrategy.NONE, "No recovery required.", retryable=True
        ).decision(needed=False)

    def _health_recovery(
        self,
        result: SkillResult,
        health_reports: tuple[dict[str, Any], ...],
    ) -> TaskRecoveryDecision | None:
        report = _matching_health_report(result, health_reports)
        if report is None:
            return None
        component = str(report.get("component") or "health")
        evidence = str(report.get("evidence") or report.get("status") or "")
        fix_hint = report.get("fix_hint")
        impacted = tuple(str(item) for item in report.get("impacted_skills") or ())
        component_lower = component.lower()
        metadata = {
            "skill_id": result.skill_id,
            "status": result.status,
            "failure_mode": result.failure_mode,
            "health_report": report,
        }
        if "camera" in component_lower or "audio" in component_lower:
            summary = f"{component} is not ready: {evidence}"
            if fix_hint:
                summary = f"{summary} Suggested fix: {fix_hint}"
            return RecoveryPlaybook(
                RecoveryStrategy.REOBSERVE,
                summary,
                severity="recoverable",
                actions=(RecoveryAction.INSPECT_SCENE, RecoveryAction.DECIDE_NEXT_STEP),
                retryable=True,
                metadata=metadata,
            ).decision()
        if any(token in component_lower for token in ("servo", "base", "arm")):
            summary = f"{component} requires operator recovery: {evidence}"
            if fix_hint:
                summary = f"{summary} Suggested fix: {fix_hint}"
            actions: tuple[RecoveryAction, ...] = (
                RecoveryAction.HOLD_TASK,
                RecoveryAction.ASK_OPERATOR,
            )
            if "base" in component_lower:
                actions = (
                    RecoveryAction.HOLD_TASK,
                    RecoveryAction.BASE_STOP,
                    RecoveryAction.ASK_OPERATOR,
                )
            return RecoveryPlaybook(
                RecoveryStrategy.ASK_OPERATOR,
                summary,
                severity="operator_required",
                actions=actions,
                operator_required=True,
                retryable=False,
                metadata={**metadata, "impacted_skills": impacted},
            ).decision()
        return None

    def _scene_recovery(self, task: TaskRun | None) -> TaskRecoveryDecision | None:
        if task is None:
            return None
        latest = task.metadata.get("latest_scene_event")
        metadata = latest.get("metadata") if isinstance(latest, dict) else None
        understanding = (
            metadata.get("understanding") if isinstance(metadata, dict) else None
        )
        if not isinstance(understanding, dict):
            return None
        risks = [str(item).lower() for item in understanding.get("risks", []) or []]
        hint = understanding.get("next_observation_hint") or understanding.get(
            "next_hint"
        )
        if any(
            "no camera" in item or "no image" in item or "missing image" in item
            for item in risks
        ):
            return RecoveryPlaybook(
                RecoveryStrategy.REOBSERVE,
                str(
                    hint
                    or "Get a fresh visual observation before choosing the next action."
                ),
                severity="recoverable",
                actions=(RecoveryAction.INSPECT_SCENE, RecoveryAction.DECIDE_NEXT_STEP),
                metadata={"scene": understanding},
            ).decision()
        if any(
            "occluded" in item or "blocked" in item or "bad view" in item
            for item in risks
        ):
            return RecoveryPlaybook(
                RecoveryStrategy.REPOSITION,
                str(hint or "Change viewpoint slightly, then inspect the scene again."),
                severity="recoverable",
                actions=(
                    RecoveryAction.REPOSITION_BASE,
                    RecoveryAction.INSPECT_SCENE,
                    RecoveryAction.DECIDE_NEXT_STEP,
                ),
                metadata={"scene": understanding},
            ).decision()
        if hint:
            return RecoveryPlaybook(
                RecoveryStrategy.REOBSERVE,
                str(hint),
                severity="recoverable",
                actions=(RecoveryAction.INSPECT_SCENE, RecoveryAction.DECIDE_NEXT_STEP),
                metadata={"scene": understanding},
            ).decision()
        return None

    @staticmethod
    def _skill_failure_recovery(result: SkillResult) -> TaskRecoveryDecision:
        contract = result.metadata.get("contract")
        recovery_hints = (
            contract.get("recovery_hints") if isinstance(contract, dict) else None
        )
        hints = tuple(str(item) for item in recovery_hints or () if str(item).strip())
        failure_mode = str(
            result.failure_mode or result.metadata.get("failure_mode") or ""
        ).lower()
        if "battery" in failure_mode:
            return RecoveryPlaybook(
                RecoveryStrategy.ASK_OPERATOR,
                result.error
                or result.summary
                or "Battery failure requires operator confirmation.",
                severity="operator_required",
                actions=(RecoveryAction.HOLD_TASK, RecoveryAction.ASK_OPERATOR),
                operator_required=True,
                retryable=False,
                metadata={
                    "skill_id": result.skill_id,
                    "status": result.status,
                    "failure_mode": failure_mode,
                },
            ).decision()
        if failure_mode in {
            "target_not_visible",
            "object_not_visible",
            "not_visible",
            "occluded_target",
        }:
            return RecoveryPlaybook(
                RecoveryStrategy.REPOSITION,
                result.error
                or result.summary
                or "Target is not visible; change viewpoint and inspect again.",
                severity="recoverable",
                actions=(
                    RecoveryAction.REPOSITION_BASE,
                    RecoveryAction.INSPECT_SCENE,
                    RecoveryAction.DECIDE_NEXT_STEP,
                ),
                metadata={
                    "skill_id": result.skill_id,
                    "status": result.status,
                    "failure_mode": failure_mode,
                    "recovery_hints": hints,
                },
            ).decision()
        if failure_mode in {"resource_busy", "busy", "conflict"}:
            return RecoveryPlaybook(
                RecoveryStrategy.CLARIFY,
                (
                    result.error
                    or result.summary
                    or "The requested action conflicts with current execution; clarify whether to wait or interrupt."
                ),
                severity="operator_required",
                actions=(
                    RecoveryAction.HOLD_TASK,
                    RecoveryAction.REQUEST_CLARIFICATION,
                    RecoveryAction.ASK_OPERATOR,
                ),
                operator_required=True,
                metadata={
                    "skill_id": result.skill_id,
                    "status": result.status,
                    "failure_mode": failure_mode,
                    "recovery_hints": hints,
                },
            ).decision()
        if (
            failure_mode
            in {
                "grasp_failed",
                "grip_failed",
                "precision_error",
                "approach_offset",
                "reach_limit",
                "position_error",
            }
            or "retry" in hints
        ):
            return RecoveryPlaybook(
                RecoveryStrategy.RETRY_WITH_ADJUSTMENT,
                result.error
                or result.summary
                or f"Skill {result.skill_id} failed; retry with adjusted parameters.",
                severity="recoverable",
                actions=(
                    RecoveryAction.RETRY_SKILL,
                    RecoveryAction.INSPECT_SCENE,
                    RecoveryAction.DECIDE_NEXT_STEP,
                ),
                retryable=True,
                metadata={
                    "skill_id": result.skill_id,
                    "status": result.status,
                    "failure_mode": failure_mode or None,
                    "recovery_hints": hints,
                },
            ).decision()
        if (
            failure_mode
            in {
                "camera_degraded",
                "single_camera_only",
                "reduced_fps",
                "partial_observation",
            }
            or "degraded" in hints
        ):
            return RecoveryPlaybook(
                RecoveryStrategy.DEGRADED_CONTINUE,
                result.error
                or result.summary
                or f"Skill {result.skill_id} completed with degraded resources; continue cautiously.",
                severity="info",
                actions=(
                    RecoveryAction.DEGRADE_RESOURCE,
                    RecoveryAction.DECIDE_NEXT_STEP,
                ),
                retryable=True,
                metadata={
                    "skill_id": result.skill_id,
                    "status": result.status,
                    "failure_mode": failure_mode or None,
                    "recovery_hints": hints,
                },
            ).decision()
        if "stop_motion" in hints:
            actions = [RecoveryAction.HOLD_TASK]
            actions.append(RecoveryAction.BASE_STOP)
            actions.extend(
                [RecoveryAction.INSPECT_SCENE, RecoveryAction.DECIDE_NEXT_STEP]
            )
            return RecoveryPlaybook(
                RecoveryStrategy.REOBSERVE,
                result.error
                or result.summary
                or f"Skill {result.skill_id} failed; stop, reobserve, and continue.",
                severity="recoverable",
                actions=tuple(actions),
                metadata={
                    "skill_id": result.skill_id,
                    "status": result.status,
                    "failure_mode": failure_mode or None,
                    "recovery_hints": hints,
                },
            ).decision()
        return RecoveryPlaybook(
            RecoveryStrategy.REOBSERVE,
            result.error
            or result.summary
            or f"Skill {result.skill_id} ended with status {result.status}.",
            severity="recoverable",
            actions=(RecoveryAction.INSPECT_SCENE, RecoveryAction.DECIDE_NEXT_STEP),
            metadata={
                "skill_id": result.skill_id,
                "status": result.status,
                "failure_mode": failure_mode or None,
                "recovery_hints": hints,
            },
        ).decision()


def _matching_health_report(
    result: SkillResult,
    health_reports: tuple[dict[str, Any], ...],
) -> dict[str, Any] | None:
    if not health_reports:
        return None
    skill_name = str(result.name or result.metadata.get("skill") or "").strip()
    failure_mode = str(result.failure_mode or result.metadata.get("failure_mode") or "")
    failure_mode = failure_mode.lower()
    candidates = [
        report
        for report in health_reports
        if isinstance(report, dict)
        and str(report.get("status") or "").lower()
        not in {"ok", "available", "configured", "ready_check_required"}
    ]
    if not candidates:
        return None
    if skill_name:
        for report in candidates:
            impacted = {str(item) for item in report.get("impacted_skills") or ()}
            if skill_name in impacted:
                return report
    component_tokens = {
        "camera": ("camera", "image", "observation", "target_not_visible"),
        "audio": ("audio", "voice", "microphone", "speaker"),
        "base": ("base", "motion", "move", "turn"),
        "servo": ("servo", "joint", "arm", "gripper"),
        "arm": ("arm", "gripper", "reach", "grasp"),
    }
    for component, tokens in component_tokens.items():
        if not any(token in failure_mode for token in tokens):
            continue
        for report in candidates:
            report_component = str(report.get("component") or "").lower()
            if component in report_component:
                return report
    return None
