from __future__ import annotations

import json

from hey_robot.memory import LongTermMemoryStore, MemoryRuntime


class _Autonomy:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, int | None]] = []

    def remember(
        self, event_type: str, summary: str, *, frame_id: int | None = None
    ) -> None:
        self.events.append((event_type, summary, frame_id))


def test_memory_runtime_builds_agent_context(tmp_path) -> None:
    store = LongTermMemoryStore(tmp_path / "long_term.jsonl")
    store.remember_entity(name="cup", summary="cup is on the table", location="table")
    store.remember_user_preference(
        name="response_language", value="zh-CN", summary="用中文回答"
    )
    store.remember_task_result(
        task_id="task1",
        root_goal="find the cup",
        status="completed",
        summary="found the cup on the table",
    )
    store.remember_skill_experience(
        skill_name="inspect_scene",
        arguments={"question": "where is the cup"},
        context_summary="desk scene",
        success=True,
        summary="inspect_scene confirmed the cup on the table",
    )
    runtime = MemoryRuntime(store)

    context = runtime.build_agent_context("where is the cup", "Current scene: kitchen")

    assert context is not None
    assert "Current scene: kitchen" in context
    assert "Long-term memory:" in context
    assert "Continuity cues:" in context
    assert "Known user preference: response_language=zh-CN." in context
    assert "cup is on the table" in context


def test_memory_runtime_searches_structured_records(tmp_path) -> None:
    store = LongTermMemoryStore(tmp_path / "long_term.jsonl")
    store.remember_place(name="dock", description="charging dock is near the wall")
    runtime = MemoryRuntime(store)

    result = runtime.search(query="dock", kind="place", mode="structured")

    records = json.loads(result)
    assert records[0]["kind"] == "place"
    assert records[0]["key"] == "dock"


def test_memory_runtime_writes_event_and_autonomy(tmp_path) -> None:
    autonomy = _Autonomy()
    runtime = MemoryRuntime(
        LongTermMemoryStore(tmp_path / "long_term.jsonl"), autonomy=autonomy
    )

    result = runtime.write(kind="event", name="observation", summary="saw a cup")

    record = json.loads(result)
    assert record["kind"] == "event"
    assert record["summary"] == "saw a cup"
    assert autonomy.events == [("observation", "saw a cup", None)]


def test_memory_runtime_writes_structured_preference_anchor_and_lesson(
    tmp_path,
) -> None:
    runtime = MemoryRuntime(LongTermMemoryStore(tmp_path / "long_term.jsonl"))

    preference = json.loads(
        runtime.write(
            kind="user_preference",
            name="response_language",
            value="zh-CN",
            summary="用户偏好中文回答",
        )
    )
    anchor = json.loads(
        runtime.write(
            kind="scene_anchor",
            name="纸巾",
            location="茶几右上角",
            summary="纸巾通常放在茶几右上角",
        )
    )
    lesson = json.loads(
        runtime.write(
            kind="task_lesson",
            name="pick_cup",
            skill_name="vla_manipulation",
            success=False,
            failure_mode="viewpoint_bad",
            recovery_hint="reposition",
            summary="上次抓取失败是因为视角太斜，应先 reposition",
            task="抓取杯子",
        )
    )

    assert preference["kind"] == "user_preference"
    assert preference["metadata"]["value"] == "zh-CN"
    assert anchor["kind"] == "scene_anchor"
    assert anchor["metadata"]["location"] == "茶几右上角"
    assert lesson["kind"] == "task_lesson"
    assert lesson["metadata"]["failure_mode"] == "viewpoint_bad"


def test_memory_runtime_records_capability_tool_result(tmp_path) -> None:
    store = LongTermMemoryStore(tmp_path / "long_term.jsonl")
    runtime = MemoryRuntime(store)

    runtime.record_tool_result(
        "request_capability",
        {"capability": "vla_manipulation", "slots": {"object": "cup"}},
        "picked cup",
        True,
        context_summary="cup on table",
    )

    records = store.query("pick cup", kind="skill_experience")
    assert records[0].metadata["skill_name"] == "vla_manipulation"
    assert records[0].metadata["arguments"] == {"object": "cup"}
    assert records[0].metadata["context_summary"] == "cup on table"


def test_memory_runtime_sanitizes_execution_feedback_before_storing(tmp_path) -> None:
    store = LongTermMemoryStore(tmp_path / "long_term.jsonl")
    runtime = MemoryRuntime(store)

    runtime.record_tool_result(
        "request_capability",
        {"capability": "reposition_for_view", "slots": {"angle_deg": 180}},
        (
            "Execution feedback for skill skill-turn:\n"
            "- outcome: failed\n"
            "- subgoal_success: False\n"
            "- task_success: False\n"
            "- summary: failed to write speed for wheel servo 9\n"
            "- failure_reason: write_failed\n"
            "- next_hint: inspect the base before retrying"
        ),
        False,
        context_summary="robot was turning in place",
    )

    records = store.query("servo 9 turn", kind="skill_experience")
    assert records[0].summary == "failed to write speed for wheel servo 9"
    assert records[0].metadata["failure_mode"] == "write_failed"
    assert records[0].metadata["recovery_hint"] == "inspect the base before retrying"
    assert records[0].metadata["verification_summary"] == (
        "outcome=failed; subgoal_success=False; task_success=False; "
        "summary=failed to write speed for wheel servo 9; failure_reason=write_failed; "
        "next_hint=inspect the base before retrying"
    )
    lessons = store.query("servo 9 retry", kind="task_lesson")
    assert lessons[0].metadata["failure_mode"] == "write_failed"
    assert lessons[0].metadata["recovery_hint"] == "inspect the base before retrying"


def test_memory_runtime_skips_transient_safety_gate_failures(tmp_path) -> None:
    store = LongTermMemoryStore(tmp_path / "long_term.jsonl")
    runtime = MemoryRuntime(store)

    runtime.record_tool_result(
        "request_capability",
        {"capability": "turn_base", "slots": {"direction": "left"}},
        (
            "RuntimeError: ConsecutiveMotionBlocked: last capability 'move_base' "
            "was also a motion/actuation skill. Run inspect_scene or "
            "request_perception to collect fresh visual evidence before issuing "
            "another motion command."
        ),
        False,
        context_summary="after moving forward",
    )

    assert store.query("turn_base", kind="skill_experience") == []
    assert store.query("turn_base", kind="task_lesson") == []


def test_memory_runtime_uses_execution_feedback_success_for_capability_result(
    tmp_path,
) -> None:
    store = LongTermMemoryStore(tmp_path / "long_term.jsonl")
    runtime = MemoryRuntime(store)

    runtime.record_tool_result(
        "request_capability",
        {
            "capability": "move_base",
            "objective": "move to the target",
            "slots": {"target": "red cup"},
        },
        (
            "Execution feedback for skill move_base:\n"
            "- outcome: failed\n"
            "- subgoal_success: False\n"
            "- task_success: False\n"
            "- summary: unknown skill: move_base\n"
            "- failure_reason: unknown_skill\n"
            "- next_hint: inspect skill catalog before retrying"
        ),
        True,
        context_summary="target was visible",
    )

    records = store.query("unknown skill move_base", kind="skill_experience")
    assert records[0].summary == "unknown skill: move_base"
    assert records[0].confidence == 0.4
    assert records[0].metadata["success"] is False
    assert records[0].metadata["failure_mode"] == "unknown_skill"
    assert (
        records[0].metadata["recovery_hint"] == "inspect skill catalog before retrying"
    )

    lessons = store.query("move_base retry", kind="task_lesson")
    assert lessons[0].metadata["success"] is False
    assert lessons[0].metadata["failure_mode"] == "unknown_skill"
