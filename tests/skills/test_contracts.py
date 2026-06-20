from __future__ import annotations

from hey_robot.protocol import Envelope, RobotStatus
from hey_robot.skills import (
    RobotSkillAction,
    SkillContractRuntime,
    load_skill_registry,
)

SKILL_CONTRACTS = load_skill_registry().robot_skill_catalog()


def test_skill_contract_runtime_blocks_missing_required_arguments() -> None:
    runtime = SkillContractRuntime()
    contract = SKILL_CONTRACTS.get("move_base")

    decision = runtime.acceptance_decision(contract, arguments={})

    assert decision.allowed is False
    assert decision.failure_mode == "invalid_arguments"
    assert decision.metadata["missing_arguments"] == ["direction", "distance_cm"]


def test_skill_contract_runtime_checks_readiness_by_required_resource() -> None:
    runtime = SkillContractRuntime()
    contract = SKILL_CONTRACTS.get("human_follow")

    decision = runtime.acceptance_decision(
        contract,
        arguments={"duration_sec": 10},
        readiness={
            "base": {"ok": True},
            "camera": {"ok": False},
            "battery": {"status": "normal"},
        },
    )

    assert decision.allowed is False
    assert decision.failure_mode == "readiness_failed"
    assert "camera is not ready" in decision.reason


def test_skill_contract_runtime_accepts_human_follow_contract() -> None:
    runtime = SkillContractRuntime()
    contract = SKILL_CONTRACTS.get("human_follow")

    decision = runtime.acceptance_decision(
        contract,
        arguments={"duration_sec": 10},
        readiness={
            "base": {"ok": True},
            "camera": {"ok": True},
            "battery": {"status": "normal"},
        },
    )

    assert decision.allowed is True


def test_skill_contract_runtime_preserves_stop_path_when_readiness_is_bad() -> None:
    runtime = SkillContractRuntime()
    contract = SKILL_CONTRACTS.get("stop_motion")

    decision = runtime.acceptance_decision(
        contract,
        readiness={
            "base": {"ok": False},
            "battery": {"status": "critical"},
            "emergency_stop": True,
        },
    )

    assert decision.allowed is True


def test_skill_contract_runtime_keeps_existing_status_precondition_policy() -> None:
    runtime = SkillContractRuntime()
    contract = SKILL_CONTRACTS.get("move_base")

    decision = runtime.acceptance_decision(
        contract,
        arguments={"direction": "forward", "distance_cm": 5},
        status=RobotStatus(
            envelope=Envelope(robot_id="xlerobot"),
            state="idle",
            metrics={"battery": {"status": "low"}},
        ),
    )

    assert decision.allowed is False
    assert decision.failure_mode == "precondition_failed"
    assert "battery low" in decision.reason


def test_skill_contract_runtime_validates_robot_action_against_catalog() -> None:
    runtime = SkillContractRuntime()

    contract, decision = runtime.validate_action(
        RobotSkillAction("move_base", {"direction": "forward", "distance_cm": 10}),
        robot_type="xlerobot",
        readiness={"base": {"ok": True}, "battery": {"status": "normal"}},
    )

    assert contract.name == "move_base"
    assert decision.allowed is True


def test_skill_contract_runtime_instantiates_targeted_resources_from_arguments() -> (
    None
):
    runtime = SkillContractRuntime()
    contract = SKILL_CONTRACTS.get("inspect_scene")

    resources = runtime.normalized_resources(
        contract, arguments={"arm": "left", "camera": "left_wrist"}
    )

    assert resources == {"left_wrist_camera"}


def test_skill_contract_runtime_allows_dual_arm_parallel_resources() -> None:
    runtime = SkillContractRuntime()
    contract = SKILL_CONTRACTS.get("set_gripper")

    assert (
        runtime.resources_conflict(
            contract,
            contract,
            left_arguments={"arm": "left", "action": "open"},
            right_arguments={"arm": "right", "action": "open"},
        )
        is False
    )
    assert contract.timeout_sec >= 10.0


def test_skill_contract_runtime_blocks_same_targeted_arm_resource_conflict() -> None:
    runtime = SkillContractRuntime()
    contract = SKILL_CONTRACTS.get("set_arm_pose")

    shared = runtime.shared_or_global_resources(
        contract,
        contract,
        left_arguments={"arm": "left", "pose_name": "home"},
        right_arguments={"arm": "left", "pose_name": "pregrasp"},
    )

    assert shared == {"left_arm"}


# ── bug-finding tests ────────────────────────────────────────────────────────


class TestResourceReadyFallback:
    """_resource_ready has multi-level fallback logic with subtle precedence."""

    def test_dict_with_ok_false_available_true_is_not_ready(self) -> None:
        runtime = SkillContractRuntime()
        assert (
            runtime._resource_ready("arm", {"arm": {"ok": False, "available": True}})
            is False
        )  # type: ignore[arg-type]

    def test_dict_with_available_true_no_ok_is_ready(self) -> None:
        runtime = SkillContractRuntime()
        assert runtime._resource_ready("arm", {"arm": {"available": True}}) is True  # type: ignore[arg-type]

    def test_dict_with_ready_false_no_ok_no_available_is_not_ready(self) -> None:
        runtime = SkillContractRuntime()
        assert runtime._resource_ready("arm", {"arm": {"ready": False}}) is False  # type: ignore[arg-type]

    def test_dict_without_any_known_key_defaults_ready(self) -> None:
        """A resource dict like {'moving': True} has no ok/available/ready key,
        but the resource is present and reporting — it should default to ready."""
        runtime = SkillContractRuntime()
        assert runtime._resource_ready("arm", {"arm": {"moving": True}}) is True  # type: ignore[arg-type]

    def test_non_dict_truthy_value_is_ready(self) -> None:
        runtime = SkillContractRuntime()
        assert runtime._resource_ready("arm", {"arm": True}) is True  # type: ignore[arg-type]

    def test_non_dict_falsy_value_is_not_ready(self) -> None:
        runtime = SkillContractRuntime()
        assert runtime._resource_ready("arm", {"arm": 0}) is False  # type: ignore[arg-type]

    def test_missing_resource_falls_back_to_suffix_keys(self) -> None:
        runtime = SkillContractRuntime()
        assert runtime._resource_ready("arm", {"arm_available": True}) is True  # type: ignore[arg-type]
        assert runtime._resource_ready("arm", {"arm_ready": True}) is True  # type: ignore[arg-type]
        assert (
            runtime._resource_ready("arm", {"arm_available": False, "arm_ready": False})
            is False
        )  # type: ignore[arg-type]


# ── camera shared-resource tests ─────────────────────────────────────────────


class TestCameraSharedResource:
    def test_camera_is_in_shared_resources(self) -> None:
        assert "camera" in SkillContractRuntime.SHARED_RESOURCES

    def test_human_follow_and_inspect_scene_no_longer_conflict(self) -> None:
        runtime = SkillContractRuntime()
        human_follow = SKILL_CONTRACTS.get("human_follow")
        inspect_scene = SKILL_CONTRACTS.get("inspect_scene")

        assert runtime.resources_conflict(human_follow, inspect_scene) is False

    def test_human_follow_shared_camera_with_look_around(self) -> None:
        runtime = SkillContractRuntime()
        human_follow = SKILL_CONTRACTS.get("human_follow")
        look_around = SKILL_CONTRACTS.get("look_around")

        # human_follow {camera, base} vs look_around {camera, base}
        # camera is shared but base still conflicts → conflict detected
        assert runtime.resources_conflict(human_follow, look_around) is True

    def test_human_follow_conflicts_on_base_but_not_camera(self) -> None:
        runtime = SkillContractRuntime()
        human_follow = SKILL_CONTRACTS.get("human_follow")
        look_around = SKILL_CONTRACTS.get("look_around")

        shared = runtime.shared_or_global_resources(human_follow, look_around)
        # Only exclusive resources (base) appear in the intersection
        assert shared == {"base"}
        assert "camera" not in shared

    def test_two_camera_only_skills_never_conflict(self) -> None:
        runtime = SkillContractRuntime()
        inspect_scene = SKILL_CONTRACTS.get("inspect_scene")
        detect_marker = SKILL_CONTRACTS.get("detect_marker")

        assert runtime.resources_conflict(inspect_scene, detect_marker) is False

    def test_instance_specific_camera_also_shared(self) -> None:
        runtime = SkillContractRuntime()
        inspect_scene = SKILL_CONTRACTS.get("inspect_scene")

        left = runtime.normalized_resources(
            inspect_scene, arguments={"camera": "front"}
        )
        right = runtime.normalized_resources(
            inspect_scene, arguments={"camera": "left_wrist"}
        )

        assert left == {"front_camera"}
        assert right == {"left_wrist_camera"}

        # Both are camera instances — no conflict
        exclusive_left = runtime._exclusive_resources(left)
        exclusive_right = runtime._exclusive_resources(right)
        assert exclusive_left & exclusive_right == set()
        assert runtime._exclusive_resources({"base", "front_camera"}) == {"base"}
        assert runtime._exclusive_resources({"camera"}) == set()
        assert runtime._exclusive_resources({"arm", "camera", "base"}) == {
            "arm",
            "base",
        }

    def test_move_base_still_self_conflicts(self) -> None:
        runtime = SkillContractRuntime()
        move_base = SKILL_CONTRACTS.get("move_base")

        assert runtime.resources_conflict(move_base, move_base) is True

    def test_set_arm_pose_self_conflict_on_same_arm(self) -> None:
        runtime = SkillContractRuntime()
        set_arm_pose = SKILL_CONTRACTS.get("set_arm_pose")

        assert (
            runtime.resources_conflict(
                set_arm_pose,
                set_arm_pose,
                left_arguments={"arm": "arm", "pose_name": "home"},
                right_arguments={"arm": "arm", "pose_name": "observe_front"},
            )
            is True
        )
