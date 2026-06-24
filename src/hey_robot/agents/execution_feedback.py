from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from hey_robot.agents.runtime.execution_feedback import (
    parse_execution_feedback_response,
)
from hey_robot.agents.runtime.grounding import is_perception_skill_name
from hey_robot.agents.types import RobotSnapshot
from hey_robot.media import LocalMediaStore, MediaResolver
from hey_robot.protocol import ImageRef, RobotStatus, SkillResult
from hey_robot.providers import ReasoningImage, ReasoningMessage, ReasoningProvider
from hey_robot.templates.loader import TemplateStore

FeedbackMode = Literal["none", "status", "vision", "operator"]
FeedbackOutcome = Literal["confirmed", "failed", "partial", "unknown", "skipped"]


@dataclass(frozen=True)
class ExecutionFeedback:
    skill_id: str
    outcome: FeedbackOutcome
    task_success: bool | None
    subgoal_success: bool | None
    confidence: float | None
    summary: str
    next_hint: str | None = None
    failure_reason: str | None = None
    recommended_action: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def successful(self) -> bool:
        return self.outcome in {"confirmed", "skipped"} and bool(self.subgoal_success)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "outcome": self.outcome,
            "task_success": self.task_success,
            "subgoal_success": self.subgoal_success,
            "confidence": self.confidence,
            "summary": self.summary,
            "next_hint": self.next_hint,
            "failure_reason": self.failure_reason,
            "recommended_action": self.recommended_action,
            "metadata": self.metadata,
        }

    def for_agent(self) -> str:
        lines = [
            f"Execution feedback for skill {self.skill_id}:",
            f"- outcome: {self.outcome}",
            f"- subgoal_success: {self.subgoal_success}",
            f"- task_success: {self.task_success}",
            f"- summary: {self.summary}",
        ]
        if self.confidence is not None:
            lines.append(f"- confidence: {self.confidence:.2f}")
        if self.failure_reason:
            lines.append(f"- failure_reason: {self.failure_reason}")
        if self.next_hint:
            lines.append(f"- next_hint: {self.next_hint}")
        if self.recommended_action:
            lines.append(f"- recommended_action: {self.recommended_action}")
        return "\n".join(lines)


class ImageResolver(Protocol):
    def resolve_images(self, refs: list[ImageRef]) -> list: ...


class ExecutionFeedbackEvaluator(Protocol):
    async def evaluate(
        self,
        *,
        task: str,
        skill_objective: str,
        result: SkillResult,
        snapshot: RobotSnapshot,
        mode: FeedbackMode,
    ) -> ExecutionFeedback: ...


class DefaultExecutionFeedbackEvaluator:
    def __init__(
        self,
        *,
        status_backend: str = "status",
        vision_backend: ExecutionFeedbackEvaluator | None = None,
    ) -> None:
        self.status_backend = status_backend
        self.vision_backend = vision_backend

    async def evaluate(
        self,
        *,
        task: str,
        skill_objective: str,
        result: SkillResult,
        snapshot: RobotSnapshot,
        mode: FeedbackMode,
    ) -> ExecutionFeedback:
        if result.status != "completed" or result.success is False:
            return ExecutionFeedback(
                skill_id=result.skill_id,
                outcome="failed",
                task_success=False,
                subgoal_success=False,
                confidence=1.0,
                summary=result.error or result.summary or f"skill {result.status}",
                failure_reason=result.error
                or result.summary
                or f"skill {result.status}",
                next_hint="recover or inspect the scene before issuing another actuation skill",
                recommended_action="inspect_or_recover",
                metadata={"mode": mode, "controller_status": result.status},
            )
        if mode == "none":
            camera_issue = _camera_quality_issue_from_snapshot(snapshot)
            if camera_issue:
                return ExecutionFeedback(
                    skill_id=result.skill_id,
                    outcome="failed",
                    task_success=False,
                    subgoal_success=False,
                    confidence=0.9,
                    summary=f"perception degraded: {camera_issue}",
                    failure_reason=camera_issue,
                    next_hint="inspect camera status and retry perception before any motion",
                    recommended_action="reobserve",
                    metadata={"mode": mode, "camera_issue": camera_issue},
                )
            return ExecutionFeedback(
                skill_id=result.skill_id,
                outcome="skipped",
                task_success=False,
                subgoal_success=True,
                confidence=1.0,
                summary=result.summary
                or "skill completed; no execution feedback required",
                next_hint="continue with the next useful step",
                recommended_action="continue",
                metadata={"mode": mode},
            )
        if mode == "vision" and self.vision_backend is not None:
            return await self.vision_backend.evaluate(
                task=task,
                skill_objective=skill_objective,
                result=result,
                snapshot=snapshot,
                mode=mode,
            )
        if mode == "operator":
            return ExecutionFeedback(
                skill_id=result.skill_id,
                outcome="unknown",
                task_success=False,
                subgoal_success=None,
                confidence=None,
                summary="skill completed; operator confirmation is required",
                next_hint="ask the operator to confirm the physical result",
                recommended_action="ask_operator",
                metadata={"mode": mode},
            )
        return status_feedback_from_result(
            result, snapshot.status, backend=self.status_backend
        )


class VisionExecutionFeedbackEvaluator:
    def __init__(
        self,
        provider: ReasoningProvider,
        *,
        image_resolver: ImageResolver | None = None,
        templates: TemplateStore | None = None,
    ) -> None:
        self.provider = provider
        self.image_resolver = image_resolver
        self.templates = templates or TemplateStore()

    async def evaluate(
        self,
        *,
        task: str,
        skill_objective: str,
        result: SkillResult,
        snapshot: RobotSnapshot,
        mode: FeedbackMode,
    ) -> ExecutionFeedback:
        images = [
            ReasoningImage(data=image)
            for image in _resolve_images(snapshot, self.image_resolver)
        ]
        response = await self.provider.chat(
            messages=[
                ReasoningMessage(
                    role="system",
                    content=self.templates.render("robot/execution_feedback/SYSTEM.md"),
                ),
                ReasoningMessage(
                    role="user",
                    content=self.templates.render(
                        "robot/execution_feedback/USER.md",
                        task=task,
                        skill_objective=skill_objective,
                        current_state=snapshot.summary(),
                    ),
                    images=images,
                ),
            ],
            tools=None,
        )
        parsed = parse_execution_feedback_response(response.content or "")
        return _feedback_from_parsed(
            result,
            parsed.subgoal_success,
            parsed.task_success,
            parsed.summary,
            parsed.next_hint,
            failure_reason=parsed.failure_reason,
            confidence=parsed.confidence,
            metadata={"mode": mode, "backend": "vision"},
        )


def status_feedback_from_result(
    result: SkillResult,
    status: RobotStatus | None,
    *,
    backend: str = "status",
) -> ExecutionFeedback:
    if result.status != "completed":
        return ExecutionFeedback(
            skill_id=result.skill_id,
            outcome="failed",
            task_success=False,
            subgoal_success=False,
            confidence=1.0,
            summary=result.error or result.summary or f"skill {result.status}",
            failure_reason=result.error or result.summary or f"skill {result.status}",
            recommended_action="inspect_or_recover",
            metadata={"backend": backend, "controller_status": result.status},
        )
    matching_status = status if status_matches_skill(status, result) else None
    status_for_context = matching_status or status
    if matching_status is not None and matching_status.success is False:
        summary = (
            matching_status.error
            or result.summary
            or "robot status reports unsuccessful execution"
        )
        return ExecutionFeedback(
            skill_id=result.skill_id,
            outcome="failed",
            task_success=False,
            subgoal_success=False,
            confidence=0.9,
            summary=summary,
            failure_reason=summary,
            next_hint="inspect the scene and choose the next action",
            recommended_action="reobserve",
            metadata={"backend": backend, "robot_state": matching_status.state},
        )
    camera_issue = _camera_quality_issue_from_status(status_for_context)
    if camera_issue:
        return ExecutionFeedback(
            skill_id=result.skill_id,
            outcome="failed",
            task_success=False,
            subgoal_success=False,
            confidence=0.9,
            summary=f"perception degraded: {camera_issue}",
            failure_reason=camera_issue,
            next_hint="retry camera inspection to get valid frames before any motion",
            recommended_action="reobserve",
            metadata={
                "backend": backend,
                "camera_issue": camera_issue,
                "robot_state": status_for_context.state
                if status_for_context is not None
                else None,
            },
        )
    if is_perception_skill_name(result.name):
        summary = result.summary or "perception skill completed"
        if status_for_context is not None and status_for_context.state:
            summary = f"{summary}; robot_state={status_for_context.state}"
        return ExecutionFeedback(
            skill_id=result.skill_id,
            outcome="confirmed",
            task_success=None,
            subgoal_success=True,
            confidence=0.9 if status_for_context is not None else 0.75,
            summary=summary,
            next_hint="report the observed scene; do not repeat the same perception skill unless the user asks or images are invalid",
            recommended_action="report_or_continue",
            metadata={
                "backend": backend,
                "robot_state": status_for_context.state
                if status_for_context is not None
                else None,
                "perception_completed": True,
            },
        )
    task_success = True
    if matching_status is not None and matching_status.success is not None:
        task_success = bool(matching_status.success)
    elif result.success is not None:
        task_success = bool(result.success)
    summary = result.summary or "controller completed the skill"
    if status_for_context is not None and status_for_context.state:
        summary = f"{summary}; robot_state={status_for_context.state}"
    return ExecutionFeedback(
        skill_id=result.skill_id,
        outcome="confirmed",
        task_success=task_success,
        subgoal_success=True,
        confidence=0.75 if status_for_context is not None else 0.6,
        summary=summary,
        next_hint="continue with the next useful step"
        if not task_success
        else "report task result if done",
        recommended_action="report_or_continue" if task_success else "continue",
        metadata={
            "backend": backend,
            "robot_state": status_for_context.state
            if status_for_context is not None
            else None,
            "status_matched_skill": matching_status is not None,
        },
    )


def status_matches_skill(status: RobotStatus | None, result: SkillResult) -> bool:
    return bool(
        status is not None
        and status.skill_id
        and result.skill_id
        and status.skill_id == result.skill_id
    )


def image_resolver_from_root(media_root: str = "runtime/media") -> ImageResolver:
    return MediaResolver(LocalMediaStore(media_root))


def _feedback_from_parsed(
    result: SkillResult,
    subgoal_success: bool,
    task_success: bool,
    summary: str,
    next_hint: str | None,
    *,
    failure_reason: str | None = None,
    confidence: float | None = None,
    metadata: dict[str, Any],
) -> ExecutionFeedback:
    outcome: FeedbackOutcome = "confirmed" if subgoal_success else "failed"
    return ExecutionFeedback(
        skill_id=result.skill_id,
        outcome=outcome,
        task_success=task_success,
        subgoal_success=subgoal_success,
        confidence=confidence,
        summary=summary,
        next_hint=next_hint,
        failure_reason=failure_reason if not subgoal_success else None,
        recommended_action="continue" if subgoal_success else "recover",
        metadata=metadata,
    )


def _resolve_images(snapshot: RobotSnapshot, resolver: ImageResolver | None):
    if resolver is None or snapshot.observation is None:
        return []
    return resolver.resolve_images(snapshot.observation.images)


def _camera_quality_issue_from_snapshot(snapshot: RobotSnapshot) -> str | None:
    return _camera_quality_issue_from_status(snapshot.status)


def _camera_quality_issue_from_status(status: RobotStatus | None) -> str | None:
    if status is None:
        return None
    metrics = getattr(status, "metrics", None)
    if not isinstance(metrics, dict):
        return None
    camera = metrics.get("camera")
    if not isinstance(camera, dict):
        return None
    ok = camera.get("ok")
    frame_available = camera.get("frame_available")
    valid_count = camera.get("valid_image_count")
    quality_issues = camera.get("image_quality_issues") or []
    if ok is False:
        issues_str = (
            ", ".join(map(str, quality_issues)) if quality_issues else "camera not ok"
        )
        return f"camera unhealthy: ok=False, frame_available={frame_available}, issues=[{issues_str}]"
    if frame_available is False:
        return f"camera frame not available, valid_image_count={valid_count}"
    if isinstance(valid_count, (int, float)) and valid_count <= 0:
        return f"no valid camera images (valid_image_count={valid_count})"
    if quality_issues and all(str(i).strip() for i in quality_issues):
        return f"camera quality issues: {', '.join(map(str, quality_issues))}"
    return None
