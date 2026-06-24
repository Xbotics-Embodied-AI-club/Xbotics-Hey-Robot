from hey_robot.agents.command_router import CommandRouter


def test_stop_command_routes_without_provider() -> None:
    routed = CommandRouter().route("停下所有动作")

    assert routed is not None
    assert routed.capability == "stop_motion"
    assert routed.slots == {"emergency": True}
    assert routed.interrupt is True
    assert routed.wait_policy == "wait_acceptance"


def test_reset_command_routes_without_provider() -> None:
    routed = CommandRouter().route("复位")

    assert routed is not None
    assert routed.capability == "reset_posture"
    assert routed.interrupt is True


def test_home_pose_routes_to_set_arm_pose() -> None:
    routed = CommandRouter().route("机械臂回到 home 位姿")

    assert routed is not None
    assert routed.capability == "set_arm_pose"
    assert routed.slots == {"pose_name": "home"}


def test_gripper_open_routes_to_set_gripper() -> None:
    routed = CommandRouter().route("夹爪完全张开")

    assert routed is not None
    assert routed.capability == "set_gripper"
    assert routed.slots == {"action": "open"}
