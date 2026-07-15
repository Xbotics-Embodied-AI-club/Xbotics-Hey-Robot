"""Cover all branches of SkillGateway.validate()."""

from __future__ import annotations

from hey_robot.cognition.autonomous.policy import PolicyDecision
from hey_robot.cognition.autonomous.skill_gateway import DispatchPreflight, SkillGateway
from hey_robot.protocol import Envelope, RobotExecutionGate, RobotStatus
from hey_robot.skill_os.base import SkillCatalog, SkillSpec


def _gateway(catalog_skills=None) -> SkillGateway:
    if catalog_skills is None:
        catalog_skills = [
            SkillSpec(
                name="navigate_to",
                description="navigate",
                category="navigation",
                safety_level="normal",
            ),
            SkillSpec(
                name="inspect_scene",
                description="look",
                category="observe",
                safety_level="normal",
            ),
            SkillSpec(
                name="critical_skill",
                description="critical",
                category="manipulation",
                safety_level="critical",
            ),
            SkillSpec(
                name="move_base",
                description="move",
                category="navigation",
                safety_level="normal",
            ),
        ]
    return SkillGateway(SkillCatalog(tuple(catalog_skills)))


def test_skill_gateway_is_named_dispatch_preflight() -> None:
    assert SkillGateway is DispatchPreflight


def _ready_gate() -> RobotExecutionGate:
    return RobotExecutionGate("r1", 0, "ready")


def _online_status() -> RobotStatus:
    return RobotStatus(Envelope(robot_id="r1"), state="idle", battery_percentage=80)


# ── Happy path ───────────────────────────────────────────────────────────────


def test_validate_allows_valid_skill() -> None:
    gw = _gateway()
    decision = gw.validate(
        skill_name="navigate_to",
        objective="go",
        arguments={},
        intent_kind="skill",
        gate=_ready_gate(),
        status=_online_status(),
    )
    assert decision == PolicyDecision(True)


def test_validate_allows_observation_intent() -> None:
    gw = _gateway()
    decision = gw.validate(
        skill_name="inspect_scene",
        objective="look",
        arguments={"question": "x"},
        intent_kind="observation",
        gate=_ready_gate(),
        status=_online_status(),
    )
    assert decision == PolicyDecision(True)


# ── Rejections ──────────────────────────────────────────────────────────────


def test_rejects_empty_objective() -> None:
    gw = _gateway()
    decision = gw.validate(
        skill_name="navigate_to",
        objective="",
        arguments={},
        intent_kind="skill",
        gate=_ready_gate(),
        status=_online_status(),
    )
    assert decision == PolicyDecision(False, "SAFETY_REJECTED")


def test_rejects_whitespace_objective() -> None:
    gw = _gateway()
    decision = gw.validate(
        skill_name="navigate_to",
        objective="   ",
        arguments={},
        intent_kind="skill",
        gate=_ready_gate(),
        status=_online_status(),
    )
    assert decision == PolicyDecision(False, "SAFETY_REJECTED")


def test_rejects_unknown_skill() -> None:
    gw = _gateway()
    decision = gw.validate(
        skill_name="nonexistent",
        objective="go",
        arguments={},
        intent_kind="skill",
        gate=_ready_gate(),
        status=_online_status(),
    )
    assert decision == PolicyDecision(False, "SKILL_REJECTED")


def test_rejects_observe_skill_via_request_skill() -> None:
    gw = _gateway()
    decision = gw.validate(
        skill_name="inspect_scene",
        objective="look",
        arguments={},
        intent_kind="skill",
        gate=_ready_gate(),
        status=_online_status(),
    )
    assert decision == PolicyDecision(False, "SAFETY_REJECTED")


def test_rejects_perception_category_via_request_skill() -> None:
    gw = _gateway(
        [
            SkillSpec(
                name="look_around", description="look around", category="perception"
            )
        ]
    )
    decision = gw.validate(
        skill_name="look_around",
        objective="look",
        arguments={},
        intent_kind="skill",
        gate=_ready_gate(),
        status=_online_status(),
    )
    assert decision == PolicyDecision(False, "SAFETY_REJECTED")


def test_rejects_when_gate_not_ready() -> None:
    gw = _gateway()
    gate = RobotExecutionGate("r1", 0, "stop_pending")
    decision = gw.validate(
        skill_name="navigate_to",
        objective="go",
        arguments={},
        intent_kind="skill",
        gate=gate,
        status=_online_status(),
    )
    assert decision == PolicyDecision(False, "ROBOT_EXECUTION_UNCERTAIN")

    gate2 = RobotExecutionGate("r1", 0, "uncertain")
    decision2 = gw.validate(
        skill_name="navigate_to",
        objective="go",
        arguments={},
        intent_kind="skill",
        gate=gate2,
        status=_online_status(),
    )
    assert decision2 == PolicyDecision(False, "ROBOT_EXECUTION_UNCERTAIN")


def test_rejects_when_robot_offline() -> None:
    gw = _gateway()
    offline = RobotStatus(Envelope(robot_id="r1"), state="offline")
    decision = gw.validate(
        skill_name="navigate_to",
        objective="go",
        arguments={},
        intent_kind="skill",
        gate=_ready_gate(),
        status=offline,
    )
    assert decision == PolicyDecision(False, "ROBOT_OFFLINE")


def test_rejects_when_robot_unknown_state() -> None:
    gw = _gateway()
    unknown = RobotStatus(Envelope(robot_id="r1"), state="unknown")
    decision = gw.validate(
        skill_name="navigate_to",
        objective="go",
        arguments={},
        intent_kind="skill",
        gate=_ready_gate(),
        status=unknown,
    )
    assert decision == PolicyDecision(False, "ROBOT_OFFLINE")


def test_rejects_when_robot_in_error_state() -> None:
    gw = _gateway()
    error_status = RobotStatus(Envelope(robot_id="r1"), state="error")
    decision = gw.validate(
        skill_name="navigate_to",
        objective="go",
        arguments={},
        intent_kind="skill",
        gate=_ready_gate(),
        status=error_status,
    )
    assert decision == PolicyDecision(False, "ROBOT_OFFLINE")


def test_rejects_when_status_is_none() -> None:
    gw = _gateway()
    decision = gw.validate(
        skill_name="navigate_to",
        objective="go",
        arguments={},
        intent_kind="skill",
        gate=_ready_gate(),
        status=None,
    )
    assert decision == PolicyDecision(False, "ROBOT_OFFLINE")


def test_rejects_critical_skill_on_low_battery() -> None:
    gw = _gateway()
    low_battery = RobotStatus(
        Envelope(robot_id="r1"), state="idle", battery_percentage=20.0
    )
    decision = gw.validate(
        skill_name="critical_skill",
        objective="go",
        arguments={},
        intent_kind="skill",
        gate=_ready_gate(),
        status=low_battery,
    )
    assert decision == PolicyDecision(False, "SAFETY_REJECTED")


def test_allows_critical_skill_on_sufficient_battery() -> None:
    gw = _gateway()
    high_battery = RobotStatus(
        Envelope(robot_id="r1"), state="idle", battery_percentage=50.0
    )
    decision = gw.validate(
        skill_name="critical_skill",
        objective="go",
        arguments={},
        intent_kind="skill",
        gate=_ready_gate(),
        status=high_battery,
    )
    assert decision == PolicyDecision(True)


def test_validates_input_schema_required_fields() -> None:
    gw = _gateway(
        [
            SkillSpec(
                name="pick",
                description="pick up",
                category="manipulation",
                input_schema={
                    "required": ["target_object"],
                    "properties": {"target_object": {"type": "string"}},
                },
            )
        ]
    )
    # Missing required field
    decision = gw.validate(
        skill_name="pick",
        objective="grab",
        arguments={},
        intent_kind="skill",
        gate=_ready_gate(),
        status=_online_status(),
    )
    assert decision == PolicyDecision(False, "INVALID_TOOL_ARGUMENTS")
    # With required field
    decision2 = gw.validate(
        skill_name="pick",
        objective="grab",
        arguments={"target_object": "cup"},
        intent_kind="skill",
        gate=_ready_gate(),
        status=_online_status(),
    )
    assert decision2 == PolicyDecision(True)
