from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from hey_robot.agents.execution_feedback import DefaultExecutionFeedbackEvaluator
from hey_robot.agents.runtime.agent_run import AgentRunReader, AgentRunRecorder
from hey_robot.agents.runtime.hooks import AgentRuntimeHook
from hey_robot.agents.runtime.runner import AgentRuntime, AgentRuntimeInput
from hey_robot.agents.types import RobotSnapshot
from hey_robot.protocol import Envelope, SkillResult
from hey_robot.providers import ReasoningResponse, ReasoningToolCall
from tests.conftest import FakeProvider


def _payload(task: str = "pick up the bottle") -> AgentRuntimeInput:
    return AgentRuntimeInput(
        task=task,
        images=[],
        robot_state="robot is idle",
        robot_status={"state": "idle"},
    )


def test_agent_runtime_returns_wait_on_provider_error() -> None:
    provider = FakeProvider(
        ReasoningResponse(
            content="provider unavailable", finish_reason="error", error_kind="api"
        )
    )
    runtime = AgentRuntime(provider, max_iterations=2)

    result = asyncio.run(runtime.step(_payload()))

    assert result.tool == "wait"
    assert result.stop_reason == "provider_error"
    assert result.reason == "api"
    assert result.result == "provider unavailable"


def test_agent_runtime_executes_tool_then_uses_required_final_answer_policy() -> None:
    provider = FakeProvider(
        {
            "tool": "request_perception",
            "args": {"question": "what do you see", "freshness": "fresh"},
            "reason": "need fresh visual evidence",
        }
    )
    runtime = AgentRuntime(provider, max_iterations=1)
    runtime.register_tool(
        "request_perception",
        lambda question, freshness="fresh": json.dumps(
            {
                "tool": "request_perception",
                "evidence": {
                    "status": "ok",
                    "summary": f"{question}: bottle on the table",
                },
                "freshness": freshness,
            }
        ),
        read_only=True,
        result_policy="require_final_answer",
    )

    result = asyncio.run(runtime.step(_payload("what do you see")))

    assert result.tool == "final_response"
    assert result.stop_reason == "text_response"
    assert result.result == "what do you see: bottle on the table"
    assert runtime.state.tool_calls[-1].name == "request_perception"


def test_agent_runtime_can_continue_to_second_robot_skill_after_skill_result() -> None:
    provider = FakeProvider(
        [
            {
                "tool": "request_capability",
                "args": {
                    "capability": "set_gripper",
                    "objective": "pick up the marker",
                    "slots": {"task": "pick up the marker"},
                },
                "reason": "first grasp the target object",
            },
            {
                "tool": "request_capability",
                "args": {
                    "capability": "set_gripper",
                    "objective": "place the held object into the bin",
                    "slots": {"task": "place the held object into the bin"},
                },
                "reason": "the first skill succeeded, so place the held object",
            },
            "Task complete: marker is in the bin.",
        ]
    )
    runtime = AgentRuntime(provider, max_iterations=4)
    submitted: list[dict[str, Any]] = []

    def submit_capability(
        capability: str, objective: str, slots: dict[str, Any] | None = None
    ) -> str:
        submitted.append(
            {"capability": capability, "objective": objective, "slots": slots or {}}
        )
        return f"{capability} completed: {objective}"

    runtime.register_tool(
        "request_capability", submit_capability, safety_level="actuate"
    )

    result = asyncio.run(
        runtime.step(_payload("clean the table by putting the marker into the bin"))
    )

    assert result.tool == "final_response"
    assert result.result == "Task complete: marker is in the bin."
    assert [item["objective"] for item in submitted] == [
        "pick up the marker",
        "place the held object into the bin",
    ]
    assert [record.name for record in runtime.state.tool_calls] == [
        "request_capability",
        "request_capability",
    ]


def test_agent_runtime_grounding_guard_inserts_perception_before_visual_answer() -> (
    None
):
    provider = FakeProvider(
        ["There is a bottle in front of me.", "The camera evidence shows a bottle."]
    )
    runtime = AgentRuntime(provider, max_iterations=3)

    def scene_evidence_tool(question: str, freshness: str = "fresh") -> str:
        return json.dumps(
            {
                "question": question,
                "freshness": freshness,
                "evidence": {
                    "status": "ok",
                    "summary": "fresh camera: bottle on front table",
                },
            }
        )

    runtime.register_tool(
        "request_perception",
        scene_evidence_tool,
        read_only=True,
    )

    result = asyncio.run(runtime.step(_payload("what do you see on the table?")))

    assert result.tool == "final_response"
    assert result.result == "The camera evidence shows a bottle."
    assert runtime.state.tool_calls[-1].name == "request_perception"
    assert runtime.state.tool_calls[-1].arguments == {
        "question": "what do you see on the table?",
        "freshness": "fresh",
    }


def test_agent_runtime_grounding_guard_handles_missing_perception_tool() -> None:
    provider = FakeProvider(
        ["There is a bottle in front of me.", "I do not have fresh visual perception."]
    )
    runtime = AgentRuntime(provider, max_iterations=3)

    result = asyncio.run(runtime.step(_payload("what do you see?")))

    assert result.tool == "final_response"
    assert result.result == "I do not have fresh visual perception."
    assert runtime.state.tool_calls == []
    assert provider.last_messages is not None
    assert "PerceptionUnavailable" in provider.last_messages[-1].content


def test_agent_runtime_reports_allowed_tool_block_as_failed_tool_result() -> None:
    provider = FakeProvider(
        {
            "tool": "request_capability",
            "args": {"capability": "set_gripper", "objective": "close gripper"},
            "reason": "try to grasp",
        }
    )
    runtime = AgentRuntime(provider, max_iterations=1)

    def submit_capability(capability: str, objective: str) -> str:
        return f"submitted {capability}: {objective}"

    runtime.register_tool(
        "request_capability", submit_capability, safety_level="actuate"
    )

    payload = _payload()
    payload.allowed_tools = {"get_robot_status"}

    result = asyncio.run(runtime.step(payload))

    assert result.tool == "final_response"
    assert result.reason == "failed_tool_fallback"
    assert result.tool_success is False
    assert "ToolUnavailable: request_capability is not available" in result.result


def test_agent_runtime_records_tool_decision_and_skill_memory_fallback(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(
        [
            {
                "tool": "request_capability",
                "args": {
                    "capability": "set_gripper",
                    "objective": "close gripper",
                    "slots": {"width": 0.02},
                },
                "reason": "single safe next action",
            },
            "The gripper has been closed.",
        ]
    )
    recorder = AgentRunRecorder(tmp_path, agent_run_id="run1")
    runtime = AgentRuntime(provider, max_iterations=1, agent_run_recorder=recorder)

    def submit_capability(
        capability: str, objective: str, slots: dict[str, Any] | None = None
    ) -> str:
        return f"skill submitted: {capability} {objective} {slots}"

    runtime.register_tool("request_capability", submit_capability)

    class Memory:
        def __init__(self) -> None:
            self.records: list[dict[str, Any]] = []

        def record_tool_result(
            self,
            tool: str,
            args: dict[str, Any],
            result: str,
            success: bool,
            *,
            context_summary: str = "",
        ) -> None:
            self.records.append(
                {
                    "tool": tool,
                    "args": args,
                    "result": result,
                    "success": success,
                    "context_summary": context_summary,
                }
            )

    memory = Memory()
    runtime.memory = memory

    result = asyncio.run(runtime.step(_payload()))

    assert result.tool == "final_response"
    assert result.stop_reason == "text_response"
    assert result.result == "The gripper has been closed."
    assert memory.records == [
        {
            "tool": "request_capability",
            "args": {
                "capability": "set_gripper",
                "objective": "close gripper",
                "slots": {"width": 0.02},
            },
            "result": "skill submitted: set_gripper close gripper {'width': 0.02}",
            "success": True,
            "context_summary": "",
        }
    ]
    latest = AgentRunReader(tmp_path, agent_run_id="run1").latest_agent_step()
    assert latest is not None
    assert latest["decision"]["tool"] == "request_capability"
    assert latest["result"]["success"] is True


def test_agent_runtime_emits_runtime_hooks_in_expected_order() -> None:
    provider = FakeProvider(
        [
            {
                "tool": "request_capability",
                "args": {
                    "capability": "set_gripper",
                    "objective": "close gripper",
                },
                "reason": "actuate once",
            },
            "The gripper is closed.",
        ]
    )
    events: list[tuple[str, int]] = []

    class RecordingHook(AgentRuntimeHook):
        async def before_iteration(self, context) -> None:
            events.append(("before_iteration", context.iteration))

        async def after_model_response(self, context) -> None:
            events.append(("after_model_response", context.iteration))

        async def before_execute_tools(self, context) -> None:
            events.append(("before_execute_tools", context.iteration))

        async def after_tool_results(self, context) -> None:
            events.append(("after_tool_results", context.iteration))

        async def after_iteration(self, context) -> None:
            events.append(("after_iteration", context.iteration))

    runtime = AgentRuntime(provider, max_iterations=3, runtime_hooks=[RecordingHook()])
    runtime.register_tool(
        "request_capability",
        lambda capability, objective: f"{capability} completed: {objective}",
    )

    result = asyncio.run(runtime.step(_payload("close the gripper")))

    assert result.tool == "final_response"
    assert events == [
        ("before_iteration", 1),
        ("after_model_response", 1),
        ("before_execute_tools", 1),
        ("after_tool_results", 1),
        ("before_iteration", 2),
        ("after_model_response", 2),
        ("after_iteration", 2),
    ]


def test_agent_runtime_batches_concurrency_safe_tools_and_serializes_exclusive_tools() -> (
    None
):
    calls = [
        ReasoningToolCall(id="tc1", name="read_a", arguments={}),
        ReasoningToolCall(id="tc2", name="read_b", arguments={}),
        ReasoningToolCall(id="tc3", name="move", arguments={}),
        ReasoningToolCall(id="tc4", name="read_a", arguments={}),
    ]
    provider = FakeProvider(
        ReasoningResponse(
            content="run tools",
            tool_calls=calls,
            finish_reason="tool_calls",
        )
    )
    runtime = AgentRuntime(provider, max_iterations=1)
    runtime.register_tool("read_a", lambda: "a", read_only=True)
    runtime.register_tool("read_b", lambda: "b", read_only=True)
    runtime.register_tool("move", lambda: "moved", exclusive=True)

    result = asyncio.run(runtime.step(_payload()))

    assert result.tool == "final_response"
    assert result.result == "a"
    assert [
        [call.name for call in batch] for batch in runtime._tool_call_batches(calls)
    ] == [
        ["read_a", "read_b"],
        ["move"],
        ["read_a"],
    ]
    assert [record.name for record in runtime.state.tool_calls] == [
        "read_a",
        "read_b",
        "move",
        "read_a",
    ]


def test_agent_runtime_handles_empty_text_and_failed_non_capability_tool() -> None:
    empty_runtime = AgentRuntime(FakeProvider("   "), max_iterations=1)
    empty_result = asyncio.run(empty_runtime.step(_payload()))

    assert empty_result.tool == "wait"
    assert empty_result.stop_reason == "empty_response"

    failed_provider = FakeProvider(
        {"tool": "read_sensor", "args": {}, "reason": "check state"}
    )
    failed_runtime = AgentRuntime(failed_provider, max_iterations=1)

    def failing_sensor() -> str:
        raise RuntimeError("sensor offline")

    failed_runtime.register_tool("read_sensor", failing_sensor, read_only=True)
    failed_result = asyncio.run(failed_runtime.step(_payload()))

    assert failed_result.tool == "wait"
    assert failed_result.stop_reason == "max_iterations"
    assert failed_runtime.state.tool_calls[-1].success is False


def test_agent_runtime_final_answer_policy_uses_generic_json_payload_and_plain_text() -> (
    None
):
    json_runtime = AgentRuntime(
        FakeProvider(
            {"tool": "summarize", "args": {"text": "done"}, "reason": "need summary"}
        ),
        max_iterations=1,
    )
    json_runtime.register_tool(
        "summarize",
        lambda text: json.dumps({"tool": "summarize", "summary": f"summary: {text}"}),
        read_only=True,
        result_policy="require_final_answer",
    )

    json_result = asyncio.run(json_runtime.step(_payload()))

    assert json_result.tool == "final_response"
    assert json_result.result == "summary: done"

    text_runtime = AgentRuntime(
        FakeProvider({"tool": "plain", "args": {}, "reason": "plain result"}),
        max_iterations=1,
    )
    text_runtime.register_tool(
        "plain",
        lambda: "plain final text",
        read_only=True,
        result_policy="require_final_answer",
    )

    text_result = asyncio.run(text_runtime.step(_payload()))

    assert text_result.tool == "final_response"
    assert text_result.result == "plain final text"


def test_agent_runtime_rejects_textual_tool_protocol_in_final_content() -> None:
    provider = FakeProvider(
        '让我看一下。\n<tool_calls>\n<invoke name="request_perception"></invoke>\n</tool_calls>'
    )
    runtime = AgentRuntime(provider, max_iterations=1)

    result = asyncio.run(runtime.step(_payload("what do you see")))

    assert result.tool == "wait"
    assert result.stop_reason == "invalid_tool_protocol"
    assert result.reason == "invalid_tool_protocol"


def test_agent_runtime_falls_back_after_successful_tool_when_final_content_is_tool_protocol() -> (
    None
):
    provider = FakeProvider(
        [
            {
                "tool": "request_capability",
                "args": {
                    "capability": "set_gripper",
                    "objective": "close gripper",
                    "slots": {"action": "close"},
                },
            },
            '<tool_calls>\n<invoke name="request_capability"></invoke>\n</tool_calls>',
        ]
    )
    runtime = AgentRuntime(provider, max_iterations=2)

    def submit_skill(*_: object, **__: object) -> str:
        return "gripper closed"

    runtime.register_tool("request_capability", submit_skill)

    result = asyncio.run(runtime.step(_payload("close gripper")))

    assert result.tool == "final_response"
    assert result.stop_reason == "text_response"
    assert result.result == "gripper closed"
    assert result.reason == "invalid_tool_protocol_after_successful_tool"


def test_agent_runtime_retries_when_final_content_is_internal_feedback() -> None:
    internal_feedback = (
        "Execution feedback for skill skill_move:\n"
        "- outcome: confirmed\n"
        "- subgoal_success: True\n"
        "- task_success: False\n"
        "- recommended_action: continue\n"
        "\n"
        "Task continuation:\n"
        "- remaining_goal: inspect the scene after moving closer."
    )
    provider = FakeProvider(
        [
            {
                "tool": "request_capability",
                "args": {
                    "capability": "move_base",
                    "objective": "move closer",
                    "slots": {"direction": "forward", "distance_cm": 25},
                },
            },
            internal_feedback,
            "I moved closer and need a fresh observation before continuing.",
        ]
    )
    runtime = AgentRuntime(provider, max_iterations=3)

    def submit_skill(*_: object, **__: object) -> str:
        return internal_feedback

    runtime.register_tool("request_capability", submit_skill, safety_level="actuate")

    result = asyncio.run(runtime.step(_payload("move closer and inspect")))

    assert result.tool == "final_response"
    assert result.stop_reason == "text_response"
    assert (
        result.result
        == "I moved closer and need a fresh observation before continuing."
    )
    assert "Execution feedback for skill" not in result.result
    assert provider.last_messages is not None
    assert "不要重复 execution feedback 标题" in provider.last_messages[-1].content


def test_execution_feedback_none_mode_respects_explicit_skill_failure() -> None:
    evaluator = DefaultExecutionFeedbackEvaluator()
    result = SkillResult(
        envelope=Envelope(robot_id="mock0"),
        skill_id="skill1",
        name="inspect_scene",
        status="completed",
        success=False,
        summary="camera image unavailable or degraded",
        error="camera image unavailable or degraded",
    )

    feedback = asyncio.run(
        evaluator.evaluate(
            task="inspect",
            skill_objective="inspect scene",
            result=result,
            snapshot=RobotSnapshot(robot_id="mock0"),
            mode="none",
        )
    )

    assert feedback.outcome == "failed"
    assert feedback.subgoal_success is False
    assert feedback.failure_reason == "camera image unavailable or degraded"


def test_agent_runtime_records_skill_memory_with_specialized_method() -> None:
    provider = FakeProvider(
        {
            "tool": "request_capability",
            "args": {
                "capability": "inspect_scene",
                "objective": "look",
                "arguments": {"zoom": 1},
            },
            "reason": "observe first",
        }
    )
    runtime = AgentRuntime(provider, max_iterations=1)

    def submit_camera(
        capability: str, objective: str, arguments: dict[str, Any] | None = None
    ) -> str:
        return f"{capability}:{objective}:{arguments}"

    runtime.register_tool("request_capability", submit_camera)

    class Memory:
        def __init__(self) -> None:
            self.records: list[dict[str, Any]] = []

        def record_tool_result(
            self,
            tool: str,
            args: dict[str, Any],
            result: str,
            success: bool,
            *,
            context_summary: str = "",
        ) -> None:
            self.records.append(
                {
                    "tool": tool,
                    "args": args,
                    "result": result,
                    "success": success,
                    "context_summary": context_summary,
                }
            )

    memory = Memory()
    runtime.memory = memory

    asyncio.run(runtime.step(_payload()))

    assert memory.records == [
        {
            "tool": "request_capability",
            "args": {
                "capability": "inspect_scene",
                "objective": "look",
                "arguments": {"zoom": 1},
            },
            "result": "inspect_scene:look:{'zoom': 1}",
            "success": True,
            "context_summary": "",
        }
    ]


def test_agent_runtime_rejects_dsml_style_textual_tool_protocol_in_final_content() -> (
    None
):
    provider = FakeProvider(
        "I will inspect first.\n```tool_call\nDSML.request_perception(question='front')\n```"
    )
    runtime = AgentRuntime(provider, max_iterations=1)

    result = asyncio.run(runtime.step(_payload("what do you see")))

    assert result.tool == "wait"
    assert result.stop_reason == "invalid_tool_protocol"
    assert result.reason == "invalid_tool_protocol"


def test_agent_runtime_rejects_fullwidth_dsml_tool_protocol_in_final_content() -> None:
    provider = FakeProvider(
        "再观察一下。\n<｜｜DSML｜｜tool_calls>\n"
        '<｜｜DSML｜｜invoke name="request_perception">\n'
        '<｜｜DSML｜｜parameter name="question">前方有什么？</｜｜DSML｜｜parameter>\n'
        "</｜｜DSML｜｜invoke>\n</｜｜DSML｜｜tool_calls>"
    )
    runtime = AgentRuntime(provider, max_iterations=1)

    result = asyncio.run(runtime.step(_payload("what do you see")))

    assert result.tool == "wait"
    assert result.stop_reason == "invalid_tool_protocol"
    assert result.reason == "invalid_tool_protocol"


def test_execution_feedback_none_mode_keeps_noncompleted_status_as_failure() -> None:
    evaluator = DefaultExecutionFeedbackEvaluator()
    result = SkillResult(
        envelope=Envelope(robot_id="mock0"),
        skill_id="skill1",
        name="move_base",
        status="failed",
        success=False,
        summary="motion timed out",
        error="motion timed out",
    )

    feedback = asyncio.run(
        evaluator.evaluate(
            task="move forward",
            skill_objective="move forward",
            result=result,
            snapshot=RobotSnapshot(robot_id="mock0"),
            mode="none",
        )
    )

    assert feedback.outcome == "failed"
    assert feedback.subgoal_success is False
    assert feedback.failure_reason == "motion timed out"


def test_agent_runtime_does_not_reject_normal_text_that_mentions_tool_word() -> None:
    provider = FakeProvider(
        "The tool is unavailable right now, so I cannot inspect further."
    )
    runtime = AgentRuntime(provider, max_iterations=1)

    result = asyncio.run(runtime.step(_payload("status update")))

    assert result.tool == "final_response"
    assert result.stop_reason == "text_response"
    assert (
        result.result
        == "The tool is unavailable right now, so I cannot inspect further."
    )


def test_agent_runtime_skips_grounding_when_matching_perception_evidence_already_exists() -> (
    None
):
    provider = FakeProvider("The latest camera view shows a bottle on the table.")
    runtime = AgentRuntime(provider, max_iterations=1)
    runtime.state.add_tool_call(
        "request_perception",
        {"question": "what do you see on the table?", "freshness": "fresh"},
        json.dumps(
            {
                "tool": "request_perception",
                "evidence": {"status": "ok", "summary": "bottle on table"},
            }
        ),
        success=True,
    )

    result = asyncio.run(runtime.step(_payload("what do you see on the table?")))

    assert result.tool == "final_response"
    assert result.result == "The latest camera view shows a bottle on the table."


def test_agent_runtime_injects_loop_warning_after_repeated_failed_attempts() -> None:
    provider = FakeProvider("I need a safer next step.")
    runtime = AgentRuntime(provider, max_iterations=1)
    runtime.state.add_tool_call(
        "request_capability",
        {"capability": "set_gripper", "objective": "pick the cup"},
        "target still not reachable",
        success=False,
    )
    runtime.state.add_tool_call(
        "request_capability",
        {"capability": "set_gripper", "objective": "pick the cup"},
        "target still not reachable",
        success=False,
    )
    runtime.state.add_tool_call(
        "request_capability",
        {"capability": "set_gripper", "objective": "pick the cup"},
        "target still not reachable",
        success=False,
    )

    result = asyncio.run(runtime.step(_payload("pick up the cup")))

    assert result.tool == "final_response"
    assert provider.last_messages is not None
    assert "Loop warning:" in provider.last_messages[1].content
    assert "repeated_failure" in provider.last_messages[1].content


def test_agent_runtime_appends_continuation_guidance_after_successful_capability_step() -> (
    None
):
    provider = FakeProvider(
        [
            {
                "tool": "request_capability",
                "args": {
                    "capability": "move_base",
                    "objective": "move closer to the cup",
                },
                "reason": "first move into range",
            },
            "Continue with grasp.",
        ]
    )
    runtime = AgentRuntime(provider, max_iterations=2)
    runtime.register_tool(
        "request_capability",
        lambda capability, objective: f"{capability} completed: {objective}",
        safety_level="actuate",
    )

    result = asyncio.run(
        runtime.step(_payload("pick up the cup and place it into the tray"))
    )

    assert result.tool == "final_response"
    assert result.result == "Continue with grasp."
    assert provider.last_messages is not None
    assert "Task continuation guidance:" in provider.last_messages[-1].content
    assert (
        "- original_task: pick up the cup and place it into the tray"
        in provider.last_messages[-1].content
    )
    assert (
        "- latest_completed_step: move closer to the cup"
        in provider.last_messages[-1].content
    )
    assert "- perception_required:" in provider.last_messages[-1].content
    assert "- one_motion_one_perception:" in provider.last_messages[-1].content


def test_agent_runtime_appends_recovery_guidance_after_failed_capability_step() -> None:
    provider = FakeProvider(
        [
            {
                "tool": "request_capability",
                "args": {"capability": "set_gripper", "objective": "grasp the cup"},
                "reason": "attempt grasp",
            },
            "I should inspect again before retrying.",
        ]
    )
    runtime = AgentRuntime(provider, max_iterations=2)

    def fail_capability(capability: str, objective: str) -> str:
        raise RuntimeError(f"{capability} failed: {objective}")

    runtime.register_tool("request_capability", fail_capability, safety_level="actuate")
    payload = _payload("pick up the cup")
    payload.recovery_context = "Recovery context: target may be occluded"

    result = asyncio.run(runtime.step(payload))

    assert result.tool == "final_response"
    assert result.result == "I should inspect again before retrying."
    assert provider.last_messages is not None
    assert "Task recovery guidance:" in provider.last_messages[-1].content
    assert "- failed_step: grasp the cup" in provider.last_messages[-1].content
    assert "- do_not_retry_blindly:" in provider.last_messages[-1].content
    assert "- recovery_context_active:" in provider.last_messages[-1].content


def test_agent_runtime_presents_internal_inspect_scene_result_for_user() -> None:
    provider = FakeProvider(
        [
            {
                "tool": "request_capability",
                "args": {"capability": "inspect_scene", "objective": "look"},
                "reason": "need visual evidence",
            },
            "scene inspected",
        ]
    )
    runtime = AgentRuntime(provider, max_iterations=1)

    def inspect_scene_result(**_kwargs: object) -> str:
        return "scene inspected"

    runtime.register_tool(
        "request_capability",
        inspect_scene_result,
        safety_level="observe",
    )

    result = asyncio.run(runtime.step(_payload("看一下前面有什么")))

    assert result.tool == "final_response"
    assert result.result == "我已经看了一下当前画面。"
    assert "scene inspected" not in result.result
    assert "inspect_scene" not in result.result


def test_agent_runtime_prefers_inspect_scene_user_summary_payload() -> None:
    provider = FakeProvider(
        [
            {
                "tool": "request_capability",
                "args": {"capability": "inspect_scene", "objective": "look"},
                "reason": "need visual evidence",
            },
            "inspect_scene completed",
        ]
    )
    runtime = AgentRuntime(provider, max_iterations=1)

    def inspect_scene_payload(**_kwargs: object) -> str:
        return json.dumps(
            {
                "success": True,
                "skill": "inspect_scene",
                "message": "scene inspected",
                "summary": "前方有一扇半开的门和一个行李箱。",
            },
            ensure_ascii=False,
        )

    runtime.register_tool(
        "request_capability",
        inspect_scene_payload,
        safety_level="observe",
    )

    result = asyncio.run(runtime.step(_payload("看一下前面有什么")))

    assert result.tool == "final_response"
    assert result.result == "前方有一扇半开的门和一个行李箱。"


def test_post_tool_guidance_sets_last_capability_safety_level_after_motion() -> None:
    provider = FakeProvider(
        [
            {
                "tool": "request_capability",
                "args": {"capability": "move_base", "objective": "move"},
                "reason": "move",
            },
            "Done.",
        ]
    )
    runtime = AgentRuntime(provider, max_iterations=2)
    runtime.register_tool(
        "request_capability",
        lambda capability, objective: f"{capability}: {objective}",
        safety_level="actuate",
    )

    asyncio.run(runtime.step(_payload("move forward")))

    assert runtime.state.last_capability_safety_level == "motion"
    assert runtime.state.last_capability_name == "move_base"


def test_post_tool_guidance_sets_last_capability_safety_level_after_observe() -> None:
    provider = FakeProvider(
        [
            {
                "tool": "request_capability",
                "args": {"capability": "inspect_scene", "objective": "look"},
                "reason": "look",
            },
            "Scene is clear.",
        ]
    )
    runtime = AgentRuntime(provider, max_iterations=2)
    runtime.register_tool(
        "request_capability",
        lambda capability, objective: f"{capability}: {objective}",
        safety_level="actuate",
    )

    asyncio.run(runtime.step(_payload("inspect the scene")))

    assert runtime.state.last_capability_safety_level == "observe"
    assert runtime.state.last_capability_name == "inspect_scene"


def test_post_tool_guidance_no_perception_required_after_observe() -> None:
    provider = FakeProvider(
        [
            {
                "tool": "request_capability",
                "args": {"capability": "inspect_scene", "objective": "check around"},
                "reason": "look",
            },
            "Camera inspection is complete.",
        ]
    )
    runtime = AgentRuntime(provider, max_iterations=4)
    runtime.register_tool(
        "request_capability",
        lambda capability, objective: f"{capability}: {objective}",
        safety_level="actuate",
    )

    result = asyncio.run(runtime.step(_payload("run a scene check")))
    assert result.tool == "final_response"
    assert provider.last_messages is not None
    guidance = provider.last_messages[-1].content
    assert "Task continuation guidance:" in guidance
    assert "do_not_stop_early" in guidance
    assert "perception_required" not in guidance


def test_post_tool_guidance_failed_motion_injects_inspect_guidance() -> None:
    provider = FakeProvider(
        [
            {
                "tool": "request_capability",
                "args": {"capability": "move_base", "objective": "move forward"},
                "reason": "move",
            },
            "I need to check the scene first.",
        ]
    )
    runtime = AgentRuntime(provider, max_iterations=2)

    def fail_move(capability: str, _objective: str) -> str:
        raise RuntimeError(f"{capability} failed: collision detected")

    runtime.register_tool("request_capability", fail_move, safety_level="actuate")

    result = asyncio.run(runtime.step(_payload("move forward")))
    assert result.tool == "final_response"
    assert provider.last_messages is not None
    guidance = provider.last_messages[-1].content
    assert "Task recovery guidance:" in guidance
    assert "- motion_failed:" in guidance


def test_failed_capability_does_not_update_last_capability_state() -> None:
    provider = FakeProvider(
        [
            {
                "tool": "request_capability",
                "args": {"capability": "turn_base", "objective": "turn left"},
                "reason": "turn",
            },
            "I need to check the scene first.",
        ]
    )
    runtime = AgentRuntime(provider, max_iterations=2)

    def fail_turn(capability: str, _objective: str) -> str:
        raise RuntimeError(f"{capability} failed: blocked")

    runtime.register_tool("request_capability", fail_turn, safety_level="actuate")

    asyncio.run(runtime.step(_payload("turn left")))

    assert runtime.state.last_capability_safety_level is None
    assert runtime.state.last_capability_name is None


def test_post_tool_guidance_returns_none_for_non_capability_tool() -> None:
    provider = FakeProvider(
        [
            {"tool": "get_robot_status", "args": {}, "reason": "check status"},
            "Robot is idle.",
        ]
    )
    runtime = AgentRuntime(provider, max_iterations=1)
    runtime.register_tool("get_robot_status", lambda: "robot is idle", read_only=True)

    asyncio.run(runtime.step(_payload("check robot status")))

    # _post_tool_guidance returns None for non-request_capability tools
    assert runtime.state.last_capability_safety_level is None
    assert runtime.state.last_capability_name is None
