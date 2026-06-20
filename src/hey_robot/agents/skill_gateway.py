from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from hey_robot.agents.recovery_capabilities import is_recovery_safe_capability
from hey_robot.agents.task_safety import evaluate_skill_request
from hey_robot.protocol import Envelope, SkillIntent
from hey_robot.skills.base import SkillCatalog

WaitPolicy = Literal["wait_result", "wait_acceptance", "return_handle"]

_MOTION_CAMERA_MAX_AGE_MS = 15_000
_MOTION_CAMERA_MAX_VALID_AGE_MS = 30_000


@dataclass(frozen=True)
class SkillGatewayRequest:
    capability: str
    objective: str
    slots: dict[str, Any] | None = None
    interrupt: bool = False
    wait_policy: WaitPolicy = "wait_result"
    metadata: dict[str, Any] | None = None
    result_prefix: str = "skill"
    enforce_motion_guards: bool = True
    confirmed: bool = False


class SkillGateway:
    """Single Agent-layer gateway for submitting physical skill intents."""

    def __init__(
        self,
        *,
        io: Any,
        spec: Any,
        skill_catalog: SkillCatalog,
        runtime_state: Any,
        pending_skills: dict[str, asyncio.Future[str]],
        current_envelope: Callable[[], Envelope],
        get_task: Callable[[], str],
        on_submit: Callable[[SkillIntent], None] | None = None,
        recovery_required: Callable[[], bool] | None = None,
        task_runtime: Any = None,
    ) -> None:
        self.io = io
        self.spec = spec
        self.skill_catalog = skill_catalog
        self.runtime_state = runtime_state
        self.pending_skills = pending_skills
        self.current_envelope = current_envelope
        self.get_task = get_task
        self.on_submit = on_submit
        self.recovery_required = recovery_required
        self._task_runtime = task_runtime

    async def submit(self, request: SkillGatewayRequest) -> str:
        task_text = self.get_task()
        capability = (request.capability or "").strip()
        if not capability:
            raise ValueError("capability must not be empty")

        objective = (request.objective or "").replace("__TASK__", task_text).strip()
        if not objective:
            raise ValueError("objective must not be empty")

        slots = dict(request.slots or {})
        if self._recovery_required() and not is_recovery_safe_capability(
            capability, slots
        ):
            raise RuntimeError(
                "recovery required; inspect, stop, reset, or open the gripper before issuing another skill"
            )

        contract = self.skill_catalog.get(capability)
        envelope = self.current_envelope()
        safety_decision = evaluate_skill_request(
            capability=capability,
            objective=objective,
            contract=contract,
            task=task_text,
            channel=envelope.channel,
            settings=self.spec.settings,
            confirmed=bool(request.confirmed),
        )
        if not safety_decision.allowed:
            raise RuntimeError(safety_decision.reply or safety_decision.reason)

        if request.enforce_motion_guards and contract.safety_level in {
            "motion",
            "actuate",
        }:
            camera_block = _check_camera_health_for_motion(
                skill_name=capability,
                episode_id=envelope.episode_id,
                task_runtime=self.task_runtime,
            )
            if camera_block:
                raise RuntimeError(camera_block)
            consecutive_block = _check_consecutive_motion(
                skill_name=capability,
                runtime_state=self.runtime_state,
            )
            if consecutive_block:
                raise RuntimeError(consecutive_block)

        intent = SkillIntent(
            envelope=envelope,
            name=capability,
            objective=objective,
            arguments=slots,
            interrupt=bool(request.interrupt),
            timeout_sec=contract.timeout_sec,
            feedback_mode=contract.feedback_mode,
            metadata=dict(request.metadata or {}),
        )
        if self.on_submit is not None:
            self.on_submit(intent)

        wait_policy = request.wait_policy or "wait_result"
        future: asyncio.Future[str] | None = None
        if wait_policy == "wait_result":
            future = asyncio.Future()
            self.pending_skills[intent.skill_id] = future
        try:
            await self.io.submit_skill(intent)
        except Exception:
            if future is not None:
                self.pending_skills.pop(intent.skill_id, None)
            raise

        if wait_policy == "return_handle":
            return (
                f"{request.result_prefix}_submitted: "
                f"skill_id={intent.skill_id} capability={intent.name}"
            )
        if wait_policy == "wait_acceptance":
            return (
                f"{request.result_prefix}_accepted: "
                f"skill_id={intent.skill_id} capability={intent.name}"
            )
        if wait_policy != "wait_result":
            raise ValueError(f"unknown wait_policy: {request.wait_policy}")
        assert future is not None
        try:
            skill_timeout = getattr(self.spec, "skill_timeout_sec", 60) or 60
            return await asyncio.wait_for(future, timeout=skill_timeout)
        finally:
            self.pending_skills.pop(intent.skill_id, None)

    async def submit_direct(
        self,
        *,
        objective: str,
        slots: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        interrupt: bool = False,
    ) -> SkillIntent:
        """Submit a legacy/direct-mode intent through the Agent gateway boundary."""
        normalized_objective = (objective or "").strip()
        if not normalized_objective:
            raise ValueError("objective must not be empty")
        intent = SkillIntent(
            envelope=self.current_envelope(),
            name="",
            objective=normalized_objective,
            arguments=dict(slots or {}),
            interrupt=bool(interrupt),
            metadata=dict(metadata or {}),
        )
        if self.on_submit is not None:
            self.on_submit(intent)
        await self.io.submit_skill(intent)
        return intent

    @staticmethod
    def build_interrupt_intent(
        *,
        envelope: Envelope,
        active_skill_id: str,
        objective: str,
        metadata: dict[str, Any] | None = None,
    ) -> SkillIntent:
        """Build the interrupt intent used to stop/reconsider an active skill."""
        return SkillIntent(
            envelope=envelope,
            skill_id=active_skill_id,
            name="interrupt",
            objective=(objective or "interrupt active skill").strip(),
            interrupt=True,
            metadata=dict(metadata or {}),
        )

    def _recovery_required(self) -> bool:
        return bool(self.recovery_required and self.recovery_required())

    @property
    def task_runtime(self) -> Any:
        if callable(self._task_runtime):
            return self._task_runtime()
        return self._task_runtime


def _check_camera_health_for_motion(
    *,
    skill_name: str,
    episode_id: str | None,
    task_runtime: object,
) -> str | None:
    if not episode_id or task_runtime is None:
        return None
    robot_states = getattr(task_runtime, "robot_states", None)
    if robot_states is None:
        return None
    state = (
        robot_states.load(episode_id)
        if callable(getattr(robot_states, "load", None))
        else None
    )
    if state is None:
        return None
    last_status = state.last_status if isinstance(state.last_status, dict) else {}
    metrics = (
        last_status.get("metrics")
        if isinstance(last_status.get("metrics"), dict)
        else {}
    )
    camera = metrics.get("camera") if isinstance(metrics, dict) else None
    if not isinstance(camera, dict):
        return None
    age_ms = camera.get("age_ms")
    ok = camera.get("ok")
    if (
        ok is True
        and isinstance(age_ms, (int, float))
        and age_ms <= _MOTION_CAMERA_MAX_AGE_MS
    ):
        return None
    if ok is False or (
        isinstance(age_ms, (int, float)) and age_ms > _MOTION_CAMERA_MAX_VALID_AGE_MS
    ):
        issues = camera.get("image_quality_issues") or []
        issue_text = ", ".join(map(str, issues)) if issues else "no valid frames"
        return (
            f"CameraUnsafe: cannot execute motion skill {skill_name!r} with unhealthy camera: "
            f"ok={ok}, age_ms={age_ms}, issues=[{issue_text}]. "
            "Run inspect_scene first to collect fresh perception before any motion command."
        )
    if not isinstance(age_ms, (int, float)):
        return None
    if age_ms > _MOTION_CAMERA_MAX_AGE_MS:
        return (
            f"CameraStale: last camera frame is {age_ms:.0f}ms old (max {_MOTION_CAMERA_MAX_AGE_MS}ms) "
            f"for motion skill {skill_name!r}. Run inspect_scene first to refresh perception."
        )
    return None


def _check_consecutive_motion(
    *,
    skill_name: str,
    runtime_state: object,
) -> str | None:
    last_level = getattr(runtime_state, "last_capability_safety_level", None)
    last_name = getattr(runtime_state, "last_capability_name", None)
    if last_level not in {"motion", "actuate"}:
        return None
    if skill_name == "stop_motion":
        return None
    return (
        f"ConsecutiveMotionBlocked: last capability {last_name!r} was also a motion/actuation skill. "
        "Run inspect_scene or request_perception to collect fresh visual evidence "
        "before issuing another motion command."
    )
