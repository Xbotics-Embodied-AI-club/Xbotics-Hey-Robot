from hey_robot.user_reply import present_tool_result_for_user


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
