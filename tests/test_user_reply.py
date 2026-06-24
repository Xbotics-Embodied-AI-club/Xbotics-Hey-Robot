from hey_robot.user_reply import (
    looks_like_internal_user_reply,
    present_tool_result_for_user,
)


def test_status_tool_reply_preserves_multiline_metrics() -> None:
    reply = present_tool_result_for_user(
        tool="get_robot_status",
        args={},
        result="机器人当前异常。\n电池约 66.7%，电压 11.4V，状态 normal。",
        success=True,
    )

    assert reply is not None
    assert "机器人当前异常" in reply
    assert "电池约 66.7%" in reply


def test_execution_feedback_summary_strips_robot_state_suffix() -> None:
    reply = present_tool_result_for_user(
        tool="request_capability",
        args={"capability": "move_base"},
        result=(
            "Execution feedback for skill skill1:\n"
            "- task_success: True\n"
            "- summary: base turned right 15.0deg; robot_state=idle"
        ),
        success=True,
    )

    assert reply == "已经向右转了约 15 度。"


def test_unknown_named_pose_reply_uses_user_facing_pose_name() -> None:
    reply = present_tool_result_for_user(
        tool="request_capability",
        args={"capability": "set_arm_pose"},
        result="unknown named pose: pre_grasp",
        success=False,
    )

    assert reply == "当前没有名为“预抓取”的已验证姿态，所以我没有移动机械臂。"


def test_invalid_joint_reply_is_user_facing() -> None:
    reply = present_tool_result_for_user(
        tool="request_capability",
        args={"capability": "move_arm_joints"},
        result="unknown joint: wrist_yaw",
        success=False,
    )

    assert reply == "当前没有名为“wrist_yaw”的已验证关节，所以我没有移动机械臂。"


def test_tool_unavailable_reply_does_not_leak_internal_protocol() -> None:
    reply = present_tool_result_for_user(
        tool="request_capability",
        args={"capability": "human_follow"},
        result="ToolUnavailable: request_capability is not available in this execution context",
        success=False,
    )

    assert reply == "当前运行环境不支持这个工具或能力，所以我没有继续执行动作。"
    assert "ToolUnavailable" not in reply


def test_consecutive_motion_blocked_reply_is_user_facing() -> None:
    reply = present_tool_result_for_user(
        tool="request_capability",
        args={"capability": "move_base"},
        result=(
            "ConsecutiveMotionBlocked: last capability 'move_base' was also a "
            "motion/actuation skill. Run inspect_scene first."
        ),
        success=False,
    )

    assert reply == "为了避免连续动作带来风险，我需要先重新观察当前画面，再继续执行。"
    assert "ConsecutiveMotionBlocked" not in reply


def test_arm_pregrasp_reply_is_user_facing() -> None:
    reply = present_tool_result_for_user(
        tool="request_capability",
        args={"capability": "set_arm_pose"},
        result="arm moved to pregrasp",
        success=True,
    )

    assert reply == "机械臂已经切换到预抓取位姿。"


def test_model_self_narration_is_internal_reply() -> None:
    assert looks_like_internal_user_reply(
        '用户说"继续"，回顾一下之前的进展，然后决定下一步。'
    )


def test_task_watchdog_alert_reply_is_rephrased() -> None:
    reply = present_tool_result_for_user(
        tool="request_capability",
        args={"capability": "move_base"},
        result="任务监督告警：robot status stale for 34.6s",
        success=False,
    )

    assert (
        reply
        == "任务监督发现异常：robot status stale for 34.6s。我会先暂停继续动作，避免扩大问题。"
    )
    assert "任务监督告警" not in reply
