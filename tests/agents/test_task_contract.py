from __future__ import annotations

from hey_robot.agents.task_contract import (
    EvidenceLedger,
    build_task_contract,
    capability_type_for_name,
    default_skill_semantics,
    evidence_type_for_capability_type,
)


def test_task_contract_uses_skill_spec_semantics_for_evidence_mapping() -> None:
    semantics = default_skill_semantics()

    assert semantics["turn_base"].capability_type == "base_turn"
    assert semantics["turn_base"].evidence_outputs == ("base_turn_action_result",)
    assert capability_type_for_name("turn_base", semantics) == "base_turn"
    assert (
        evidence_type_for_capability_type("base_turn", semantics)
        == "base_turn_action_result"
    )


def test_evidence_ledger_distinguishes_caption_from_marker_detector() -> None:
    ledger = EvidenceLedger(default_skill_semantics())

    ledger.add_tool_result(
        tool="request_capability",
        args={"capability": "inspect_scene"},
        result="red marker-like region visible",
        success=True,
    )
    ledger.add_tool_result(
        tool="request_capability",
        args={"capability": "detect_marker"},
        result="marker detector timed out",
        success=False,
    )

    assert ledger.has_successful_evidence("weak_scene_observation") is True
    assert ledger.has_successful_evidence("marker_detection_result") is False
    assert ledger.has_failed_capability_type("marker_detection") is True


def test_task_contract_does_not_treat_marker_object_manipulation_as_detection() -> None:
    contract = build_task_contract("pick up the marker and put it in the bin")

    assert contract.required_capability is None
    assert contract.task_type == "general"


def test_task_contract_requires_detector_for_marker_check() -> None:
    contract = build_task_contract("check whether there is a workspace marker")

    assert contract.required_capability is not None
    assert contract.required_capability.type == "marker_detection"
    assert contract.completion_evidence_required == ("marker_detection_result",)


def test_task_contract_treats_arm_raise_as_arm_joint_delta() -> None:
    contract = build_task_contract("机械臂末端抬高一些")

    assert contract.required_capability is not None
    assert contract.required_capability.type == "arm_joint_delta"
    assert contract.completion_evidence_required == ("arm_joint_action_result",)
