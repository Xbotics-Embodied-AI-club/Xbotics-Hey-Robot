from __future__ import annotations

from pathlib import Path

from hey_robot.episode import RobotEpisodeStateStore
from hey_robot.protocol import Envelope, SkillEvent
from hey_robot.recovery.service import (
    RecoveryService,
    _actions_for_phase,
    _next_step_for_state,
    _reason_for_phase,
    _requires_recovery,
    _severity,
    _skill_to_dict,
    _strategy_for_state,
    _user_message_for_state,
)
from hey_robot.skills import SkillStore


class TestRecoveryHelpers:
    def test_requires_recovery_true_for_relevant_phases(self) -> None:
        assert _requires_recovery("feedback_pending") is True
        assert _requires_recovery("failed") is True
        assert _requires_recovery("interrupted") is True
        assert _requires_recovery("feedback_failed") is True

    def test_requires_recovery_false_for_other_phases(self) -> None:
        assert _requires_recovery("executing") is False
        assert _requires_recovery("completed") is False
        assert _requires_recovery(None) is False
        assert _requires_recovery("unknown") is False

    def test_reason_for_phase_feedback_pending(self) -> None:
        assert "execution feedback" in (_reason_for_phase("feedback_pending") or "")

    def test_reason_for_phase_failed_interrupted_feedback_failed(self) -> None:
        for phase in ("failed", "interrupted", "feedback_failed"):
            result = _reason_for_phase(phase)
            assert result is not None
            assert "recovery phase" in result

    def test_reason_for_phase_other_returns_none(self) -> None:
        assert _reason_for_phase("executing") is None
        assert _reason_for_phase(None) is None

    def test_severity_ok_when_not_required(self) -> None:
        assert _severity("executing", False) == "ok"
        assert _severity(None, False) == "ok"

    def test_severity_operator_required_for_terminal_phases(self) -> None:
        for phase in ("failed", "interrupted", "feedback_failed"):
            assert _severity(phase, True) == "operator_required"

    def test_severity_feedback_required_default(self) -> None:
        assert _severity("feedback_pending", True) == "feedback_required"
        assert _severity(None, True) == "feedback_required"

    def test_actions_for_phase_empty_when_not_required(self) -> None:
        from hey_robot.episode.robot_state import RobotEpisodeState

        assert (
            _actions_for_phase(RobotEpisodeState(episode_id="ep1"), "executing", False)
            == []
        )

    def test_actions_for_phase_includes_abort_when_not_confirmed(self) -> None:
        from hey_robot.episode.robot_state import RobotEpisodeState

        actions = _actions_for_phase(
            RobotEpisodeState(episode_id="ep1"), "failed", True
        )
        names = [a.name for a in actions]
        assert "abort" in names
        assert "mark_confirmed" in names

    def test_actions_for_phase_excludes_abort_when_confirmed(self) -> None:
        from hey_robot.episode.robot_state import RobotEpisodeState

        actions = _actions_for_phase(
            RobotEpisodeState(episode_id="ep1"), "confirmed", True
        )
        names = [a.name for a in actions]
        assert "mark_confirmed" in names
        assert "abort" not in names

    def test_strategy_helpers_map_busy_to_clarify(self) -> None:
        from hey_robot.episode.robot_state import RobotEpisodeState

        state = RobotEpisodeState(
            episode_id="ep1",
            recovery_required=True,
            recovery_reason="resource busy conflict",
        )

        assert _strategy_for_state(state, "failed", True) == "clarify"
        assert (
            "clarification"
            in (_user_message_for_state(state, "failed", True) or "").lower()
        )
        assert (
            "wait, interrupt"
            in (_next_step_for_state(state, "failed", True) or "").lower()
        )

    def test_strategy_helpers_map_hint_to_reposition(self) -> None:
        from hey_robot.episode.robot_state import RobotEpisodeState

        state = RobotEpisodeState(
            episode_id="ep1",
            recovery_required=True,
            recovery_reason="last execution feedback failed",
            last_execution_feedback={
                "next_hint": "move closer to the target and reposition"
            },
        )

        assert _strategy_for_state(state, "failed", True) == "reposition"
        actions = _actions_for_phase(state, "failed", True)
        assert actions[0].name == "reposition"
        assert (
            "change viewpoint"
            in (_user_message_for_state(state, "failed", True) or "").lower()
        )

    def test_skill_to_dict_none(self) -> None:
        assert _skill_to_dict(None) is None

    def test_skill_to_dict_returns_dict(self) -> None:
        from dataclasses import dataclass

        @dataclass
        class FakeSkill:
            skill_id: str = "s1"
            phase: str = "executing"

        result = _skill_to_dict(FakeSkill())  # type: ignore[arg-type]
        assert result is not None
        assert result.get("skill_id") == "s1"


def test_recovery_service_plans_and_marks_confirmed(tmp_path: Path) -> None:
    skills = SkillStore(tmp_path / "skills")
    states = RobotEpisodeStateStore(tmp_path / "episodes")
    service = RecoveryService(states, skills)
    envelope = Envelope(
        trace_id="tr1", episode_id="s1", agent_id="main", robot_id="mock0"
    )

    states.mark_task_started("s1", task="move bowl", agent_id="main", robot_id="mock0")
    skills.append(
        SkillEvent(envelope=envelope, skill_id="cmd1", phase="issued", text="move bowl")
    )
    states.apply_skill_event(
        SkillEvent(envelope=envelope, skill_id="cmd1", phase="issued", text="move bowl")
    )
    skills.append(
        SkillEvent(envelope=envelope, skill_id="cmd1", phase="feedback_pending")
    )
    states.apply_skill_event(
        SkillEvent(envelope=envelope, skill_id="cmd1", phase="feedback_pending")
    )

    recovery = service.recovery_for_episode("s1")

    assert recovery is not None
    assert recovery.recovery_required is True
    assert recovery.severity == "feedback_required"
    assert recovery.strategy == "clarify"
    assert recovery.user_message is not None
    assert [action.name for action in recovery.actions] == [
        "clarify",
        "mark_confirmed",
        "abort",
    ]

    confirmed = service.mark_confirmed("s1", operator="tester")

    assert confirmed.recovery_required is False
    assert confirmed.skill_phase == "confirmed"
    assert confirmed.last_execution_feedback is not None
    assert skills.get("cmd1") is not None
    assert skills.get("cmd1").phase == "confirmed"  # type: ignore[union-attr]


def test_recovery_service_abort_keeps_operator_required(tmp_path: Path) -> None:
    skills = SkillStore(tmp_path / "skills")
    states = RobotEpisodeStateStore(tmp_path / "episodes")
    service = RecoveryService(states, skills)
    envelope = Envelope(
        trace_id="tr1", episode_id="s1", agent_id="main", robot_id="mock0"
    )

    states.mark_task_started("s1", task="move bowl", agent_id="main", robot_id="mock0")
    skills.append(
        SkillEvent(
            envelope=envelope, skill_id="cmd1", phase="executing", text="move bowl"
        )
    )
    states.apply_skill_event(
        SkillEvent(
            envelope=envelope, skill_id="cmd1", phase="executing", text="move bowl"
        )
    )

    recovery = service.abort("s1", reason="operator stopped skill", operator="tester")

    assert recovery.recovery_required is True
    assert recovery.skill_phase == "interrupted"
    assert recovery.severity == "operator_required"
    assert recovery.strategy == "reobserve"
    assert skills.get("cmd1") is not None
    assert skills.get("cmd1").phase == "interrupted"  # type: ignore[union-attr]


def test_recovery_for_episode_returns_none_when_state_missing(tmp_path: Path) -> None:
    skills = SkillStore(tmp_path / "skills")
    states = RobotEpisodeStateStore(tmp_path / "episodes")
    service = RecoveryService(states, skills)
    assert service.recovery_for_episode("nonexistent") is None
