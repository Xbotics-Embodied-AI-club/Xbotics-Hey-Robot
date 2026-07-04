from hey_robot.cognition.recovery_capabilities import is_recovery_safe_skill


def test_recovery_safe_skills_allow_observe_stop_and_open_gripper() -> None:
    assert is_recovery_safe_skill("inspect_scene", None)
    assert is_recovery_safe_skill("stop_motion", {})
    assert is_recovery_safe_skill("reset_posture", {})
    assert is_recovery_safe_skill("set_gripper", {"action": "open"})
    assert is_recovery_safe_skill("set_gripper", {"opening_pct": 80})
    assert is_recovery_safe_skill("set_gripper", {"opening_pct": "95"})


def test_recovery_safe_skills_reject_actuation_and_invalid_gripper_slots() -> None:
    assert not is_recovery_safe_skill("move_base", {"direction": "forward"})
    assert not is_recovery_safe_skill("set_gripper", {"action": "close"})
    assert not is_recovery_safe_skill("set_gripper", {})
    assert not is_recovery_safe_skill("set_gripper", {"opening_pct": 40})
    assert not is_recovery_safe_skill("set_gripper", {"opening_pct": "wide"})
