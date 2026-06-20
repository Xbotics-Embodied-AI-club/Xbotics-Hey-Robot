from hey_robot.agents.runtime.state import AgentState


def test_state_reset_clears_loop_context():
    state = AgentState(
        task="pick the bottle",
        last_observation_summary="bottle on table",
        last_error="failed",
    )
    state.add_message("user", "pick the bottle")
    state.add_tool_call("request_capability", {"name": "inspect_scene"}, "ok")

    state.reset()

    assert state.task == ""
    assert state.messages == []
    assert state.tool_calls == []
    assert state.last_observation_summary is None
    assert state.last_error is None


def test_add_message_ignores_blank_content():
    state = AgentState()

    state.add_message("assistant", "  ")
    state.add_message("user", " look ")

    assert state.messages == [{"role": "user", "content": "look"}]


def test_recent_tool_context_reports_attempt_history():
    state = AgentState()
    state.add_tool_call(
        "request_perception", {"question": "what is on the table"}, "bottle visible"
    )
    state.add_tool_call(
        "request_capability",
        {"objective": "grasp bottle"},
        "grasp failed",
        success=False,
    )

    context = state.recent_tool_context()

    assert "Recent tool calls:" in context
    assert "request_perception" in context
    assert "request_capability" in context
    assert "error: grasp failed" in context


def test_loop_warning_context_reports_repeated_failures_and_no_progress():
    state = AgentState()
    for _ in range(3):
        state.add_tool_call(
            "request_capability",
            {"capability": "set_gripper", "objective": "pick cup"},
            "target still not reachable",
            success=False,
        )

    context = state.loop_warning_context()

    assert "Loop warning:" in context
    assert "repeated_attempt" in context
    assert "repeated_failure" in context
    assert "no_progress_signal" in context
    assert "do not repeat the same action without new evidence" in context


def test_last_capability_safety_level_initialized_as_none():
    state = AgentState()
    assert state.last_capability_safety_level is None
    assert state.last_capability_name is None


def test_last_capability_fields_reset_clears_tracking():
    state = AgentState(
        last_capability_safety_level="motion",
        last_capability_name="move_base",
    )
    state.reset()
    assert state.last_capability_safety_level is None
    assert state.last_capability_name is None


def test_last_capability_fields_persist_independent_of_tool_calls():
    state = AgentState(
        last_capability_safety_level="motion", last_capability_name="move_base"
    )
    state.add_tool_call("request_capability", {"capability": "inspect_scene"}, "ok")
    # tool_calls record is separate from capability tracking — runner sets these
    assert state.last_capability_safety_level == "motion"
    assert state.last_capability_name == "move_base"
    assert len(state.tool_calls) == 1
