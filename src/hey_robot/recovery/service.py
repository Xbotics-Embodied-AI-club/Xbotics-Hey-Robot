from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from hey_robot.episode import RobotEpisodeState, RobotEpisodeStateStore
from hey_robot.protocol import Envelope, SkillEvent
from hey_robot.skills import SkillPhase, SkillRecord, SkillStore


@dataclass(frozen=True)
class RecoveryAction:
    name: str
    label: str
    description: str
    destructive: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecoveryDecision:
    episode_id: str
    robot_id: str | None
    agent_id: str | None
    active_task: str | None
    skill_id: str | None
    skill_phase: str | None
    recovery_required: bool
    reason: str | None
    severity: str
    strategy: str | None = None
    user_message: str | None = None
    next_step: str | None = None
    last_observation_frame: int | None = None
    last_status: dict[str, Any] = field(default_factory=dict)
    last_execution_feedback: dict[str, Any] | None = None
    skill: dict[str, Any] | None = None
    actions: list[RecoveryAction] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["actions"] = [asdict(action) for action in self.actions]
        return data


class RecoveryService:
    """Operator-facing recovery decision service for robot episodes.

    It intentionally operates on materialized stores, not live services. That
    makes it usable from the web console during robot validation and after
    partial service restarts.
    """

    def __init__(
        self, robot_states: RobotEpisodeStateStore, skills: SkillStore
    ) -> None:
        self.robot_states = robot_states
        self.skills = skills

    def list_recoveries(self, *, include_clean: bool = False) -> list[RecoveryDecision]:
        recoveries = [
            self.recovery_for_state(state) for state in self.robot_states.list_states()
        ]
        if not include_clean:
            recoveries = [item for item in recoveries if item.recovery_required]
        return sorted(
            recoveries,
            key=lambda item: (item.recovery_required, item.skill_id or ""),
            reverse=True,
        )

    def recovery_for_episode(self, episode_id: str) -> RecoveryDecision | None:
        state = self.robot_states.load(episode_id)
        if state is None:
            return None
        return self.recovery_for_state(state)

    def recovery_for_state(self, state: RobotEpisodeState) -> RecoveryDecision:
        skill = (
            self.skills.get(state.active_skill_id) if state.active_skill_id else None
        )
        phase = state.active_skill_phase or (skill.phase if skill else None)
        recovery_required = bool(state.recovery_required or _requires_recovery(phase))
        reason = state.recovery_reason or _reason_for_phase(phase)
        return RecoveryDecision(
            episode_id=state.episode_id,
            robot_id=state.robot_id,
            agent_id=state.agent_id,
            active_task=state.active_task,
            skill_id=state.active_skill_id,
            skill_phase=phase,
            recovery_required=recovery_required,
            reason=reason,
            severity=_severity(phase, recovery_required),
            strategy=_strategy_for_state(state, phase, recovery_required),
            user_message=_user_message_for_state(state, phase, recovery_required),
            next_step=_next_step_for_state(state, phase, recovery_required),
            last_observation_frame=state.last_observation_frame,
            last_status=dict(state.last_status),
            last_execution_feedback=(
                dict(state.last_execution_feedback)
                if state.last_execution_feedback
                else None
            ),
            skill=_skill_to_dict(skill),
            actions=_actions_for_phase(state, phase, recovery_required),
        )

    def mark_confirmed(
        self,
        episode_id: str,
        *,
        summary: str = "operator confirmed skill result",
        operator: str | None = None,
    ) -> RecoveryDecision:
        state = self._require_state(episode_id)
        skill_id = self._require_skill_id(state)
        event = SkillEvent(
            envelope=self._envelope_for_state(state),
            skill_id=skill_id,
            phase=SkillPhase.CONFIRMED.value,
            text=state.active_skill_text,
            summary=summary,
            metadata={"operator": operator, "source": "recovery.operator"},
        )
        self.skills.append(event)
        self.robot_states.apply_skill_event(event)
        self.robot_states.mark_execution_feedback(
            episode_id,
            skill_id=skill_id,
            subgoal_success=True,
            task_success=False,
            summary=summary,
            next_hint=None,
        )
        decision = self.recovery_for_episode(episode_id)
        assert decision is not None
        return decision

    def abort(
        self,
        episode_id: str,
        *,
        reason: str = "operator aborted recovery",
        operator: str | None = None,
    ) -> RecoveryDecision:
        state = self._require_state(episode_id)
        skill_id = self._require_skill_id(state)
        event = SkillEvent(
            envelope=self._envelope_for_state(state),
            skill_id=skill_id,
            phase=SkillPhase.INTERRUPTED.value,
            text=state.active_skill_text,
            error=reason,
            summary=reason,
            metadata={"operator": operator, "source": "recovery.operator"},
        )
        self.skills.append(event)
        self.robot_states.apply_skill_event(event)
        decision = self.recovery_for_episode(episode_id)
        assert decision is not None
        return decision

    def _require_state(self, episode_id: str) -> RobotEpisodeState:
        state = self.robot_states.load(episode_id)
        if state is None:
            raise KeyError(f"unknown episode: {episode_id}")
        return state

    def _require_skill_id(self, state: RobotEpisodeState) -> str:
        if not state.active_skill_id:
            raise ValueError(f"episode {state.episode_id} has no active skill")
        return state.active_skill_id

    def _envelope_for_state(self, state: RobotEpisodeState) -> Envelope:
        skill = (
            self.skills.get(state.active_skill_id) if state.active_skill_id else None
        )
        trace_id = (
            skill.trace_id
            if skill and skill.trace_id
            else f"recovery_{int(time.time() * 1000)}"
        )
        return Envelope(
            trace_id=trace_id,
            episode_id=state.episode_id,
            agent_id=state.agent_id,
            robot_id=state.robot_id,
            channel=skill.channel if skill else None,
        )


def _requires_recovery(phase: str | None) -> bool:
    return phase in {
        SkillPhase.FEEDBACK_PENDING.value,
        SkillPhase.FAILED.value,
        SkillPhase.INTERRUPTED.value,
        SkillPhase.FEEDBACK_FAILED.value,
    }


def _reason_for_phase(phase: str | None) -> str | None:
    if phase == SkillPhase.FEEDBACK_PENDING.value:
        return "skill completed and requires execution feedback"
    if phase in {
        SkillPhase.FAILED.value,
        SkillPhase.INTERRUPTED.value,
        SkillPhase.FEEDBACK_FAILED.value,
    }:
        return f"skill is in recovery phase {phase}"
    return None


def _severity(phase: str | None, recovery_required: bool) -> str:
    if not recovery_required:
        return "ok"
    if phase in {
        SkillPhase.FAILED.value,
        SkillPhase.INTERRUPTED.value,
        SkillPhase.FEEDBACK_FAILED.value,
    }:
        return "operator_required"
    return "feedback_required"


def _actions_for_phase(
    state: RobotEpisodeState,
    phase: str | None,
    recovery_required: bool,
) -> list[RecoveryAction]:
    if not recovery_required:
        return []
    strategy = _strategy_for_state(state, phase, recovery_required)
    actions = [
        RecoveryAction(
            name="mark_confirmed",
            label="Mark Confirmed",
            description="Clear recovery after operator confirms the last skill is safe and complete.",
        )
    ]
    if strategy == "clarify":
        actions.insert(
            0,
            RecoveryAction(
                name="clarify",
                label="Ask Clarifying Question",
                description="Explain the conflict and ask whether to wait, interrupt, or adjust the task.",
                metadata={"strategy": strategy},
            ),
        )
    elif strategy == "reposition":
        actions.insert(
            0,
            RecoveryAction(
                name="reposition",
                label="Reposition And Reobserve",
                description="Move to a better viewpoint, then inspect the scene again before continuing.",
                metadata={"strategy": strategy},
            ),
        )
    elif strategy == "reobserve":
        actions.insert(
            0,
            RecoveryAction(
                name="reobserve",
                label="Reobserve Scene",
                description="Get a fresh observation before deciding the next robot action.",
                metadata={"strategy": strategy},
            ),
        )
    elif strategy == "safe_abort":
        actions.insert(
            0,
            RecoveryAction(
                name="safe_abort",
                label="Safe Abort",
                description="Stop autonomous execution and require operator intervention before continuing.",
                destructive=True,
                metadata={"strategy": strategy},
            ),
        )
    if phase != SkillPhase.CONFIRMED.value:
        actions.append(
            RecoveryAction(
                name="abort",
                label="Abort Skill",
                description=(
                    "Mark the active skill interrupted and keep recovery required for a fresh task decision."
                ),
                destructive=True,
            )
        )
    return actions


def _skill_to_dict(skill: SkillRecord | None) -> dict[str, Any] | None:
    if skill is None:
        return None
    return asdict(skill)


def _strategy_for_state(
    state: RobotEpisodeState, phase: str | None, recovery_required: bool
) -> str | None:
    if not recovery_required:
        return None
    status = state.last_status if isinstance(state.last_status, dict) else {}
    battery = (
        status.get("metrics", {}).get("battery")
        if isinstance(status.get("metrics"), dict)
        else None
    )
    if isinstance(battery, dict) and battery.get("status") == "critical":
        return "safe_abort"
    feedback = (
        state.last_execution_feedback
        if isinstance(state.last_execution_feedback, dict)
        else {}
    )
    next_hint = str(feedback.get("next_hint") or "").lower()
    reason = str(state.recovery_reason or "").lower()
    if "busy" in reason or "conflict" in reason:
        return "clarify"
    if any(
        token in next_hint
        for token in ("reposition", "change viewpoint", "move closer")
    ):
        return "reposition"
    if any(token in reason for token in ("not visible", "occluded", "camera", "image")):
        return "reobserve"
    if phase in {
        SkillPhase.FAILED.value,
        SkillPhase.INTERRUPTED.value,
        SkillPhase.FEEDBACK_FAILED.value,
    }:
        return "reobserve"
    return "clarify" if phase == SkillPhase.FEEDBACK_PENDING.value else "reobserve"


def _user_message_for_state(
    state: RobotEpisodeState, phase: str | None, recovery_required: bool
) -> str | None:
    if not recovery_required:
        return None
    strategy = _strategy_for_state(state, phase, recovery_required)
    feedback = (
        state.last_execution_feedback
        if isinstance(state.last_execution_feedback, dict)
        else {}
    )
    summary = str(
        feedback.get("summary")
        or state.recovery_reason
        or "The last robot step needs recovery."
    )
    if strategy == "safe_abort":
        return f"{summary} I will stop autonomous execution and wait for operator help."
    if strategy == "reposition":
        return f"{summary} I will change viewpoint first, then inspect again before continuing."
    if strategy == "reobserve":
        return (
            f"{summary} I will inspect the scene again before choosing the next step."
        )
    if strategy == "clarify":
        return f"{summary} I need a quick clarification before I continue."
    return summary


def _next_step_for_state(
    state: RobotEpisodeState, phase: str | None, recovery_required: bool
) -> str | None:
    if not recovery_required:
        return None
    strategy = _strategy_for_state(state, phase, recovery_required)
    if strategy is None:
        return None
    mapping = {
        "safe_abort": "Stop the robot and ask the operator to recover the system state.",
        "reposition": "Reposition slightly, run perception again, then decide whether to continue.",
        "reobserve": "Run perception again and use the new observation to choose the next tool call.",
        "clarify": "Ask the user whether to wait, interrupt, or restate the task goal.",
    }
    return mapping.get(strategy)
