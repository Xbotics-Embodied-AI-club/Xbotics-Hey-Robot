from __future__ import annotations

from hey_robot.memory import LongTermMemoryStore


def test_long_term_memory_store_remember_and_query(tmp_path) -> None:
    store = LongTermMemoryStore(tmp_path / "long_term.jsonl")

    store.remember(
        kind="entity",
        key="cup",
        summary="cup is on the left table",
        metadata={"location": "left table"},
    )
    store.remember(
        kind="place",
        key="charging_station",
        summary="charging station is near the wall",
    )

    records = store.query("where is the cup", kind="entity")

    assert len(records) == 1
    assert records[0].key == "cup"
    assert "left table" in records[0].summary
    assert "Long-term memory:" in store.prompt_context("cup")


def test_long_term_memory_store_structured_records(tmp_path) -> None:
    store = LongTermMemoryStore(tmp_path / "long_term.jsonl")

    store.remember_entity(
        name="cup", summary="cup is on the table", location="table", frame_id=7
    )
    store.remember_place(
        name="table", description="table is in front of the robot", pose={"x": 1.0}
    )
    store.remember_skill_experience(
        skill_name="vla_manipulation",
        arguments={"object": "cup"},
        success=False,
        summary="cup slipped",
        failure_mode="grasp_failed",
        recovery_hint="scan_workspace",
        verification_summary="object not in gripper",
    )

    skill_records = store.query("grasp failed cup", kind="skill_experience")

    assert skill_records[0].metadata["skill_name"] == "vla_manipulation"
    assert skill_records[0].metadata["failure_mode"] == "grasp_failed"
    assert store.query("table", kind="place")[0].metadata["pose"] == {"x": 1.0}


def test_long_term_memory_query_prefers_latest_entity_key(tmp_path) -> None:
    store = LongTermMemoryStore(tmp_path / "long_term.jsonl")

    store.remember_entity(
        name="cup", summary="cup is on the left table", location="left table"
    )
    store.remember_entity(name="cup", summary="cup is in the bin", location="bin")

    records = store.query("\u676f\u5b50 cup \u5728\u54ea\u91cc", kind="entity")

    assert len(records) == 1
    assert records[0].key == "cup"
    assert records[0].metadata["location"] == "bin"


def test_long_term_memory_query_prefers_latest_skill_experience_for_same_arguments(
    tmp_path,
) -> None:
    store = LongTermMemoryStore(tmp_path / "long_term.jsonl")

    store.remember(
        kind="skill_experience",
        key="vla_manipulation",
        summary='Execution feedback:\n- outcome: failed\n- failure_reason: "unsupported mock skill"',
        metadata={
            "tool": "request_capability",
            "success": True,
            "arguments": {
                "name": "vla_manipulation",
                "arguments": {"object": "cup", "location": "bin"},
            },
        },
    )
    store.remember_skill_experience(
        skill_name="vla_manipulation",
        arguments={"object": "cup", "location": "bin"},
        success=True,
        summary="Placed cup at bin.",
    )

    records = store.query("relocate cup bin", kind="skill_experience")

    assert len(records) == 1
    assert records[0].summary == "Placed cup at bin."


def test_long_term_memory_task_result_is_compact(tmp_path) -> None:
    store = LongTermMemoryStore(tmp_path / "long_term.jsonl")

    record = store.remember_task_result(
        task_id="task-a",
        root_goal="charge the robot",
        status="completed",
        summary="reported task result",
    )

    assert record.summary == "reported task result"
    assert "subgoals" not in record.metadata


def test_long_term_memory_supports_preference_anchor_and_lesson_records(
    tmp_path,
) -> None:
    store = LongTermMemoryStore(tmp_path / "long_term.jsonl")

    store.remember_user_preference(
        name="response_language",
        value="zh-CN",
        summary="用户偏好使用中文回答",
    )
    store.record_scene_anchor(
        name="纸巾",
        location="茶几右上角",
        summary="纸巾通常放在茶几右上角",
        entity_type="object",
    )
    store.remember_task_lesson(
        key="pick_cup",
        task="抓取杯子",
        summary="上次抓取失败是因为视角太斜，应先 reposition",
        success=False,
        failure_mode="viewpoint_bad",
        recovery_hint="reposition",
        skill_name="vla_manipulation",
    )

    preference_records = store.query("偏好 中文", kind="user_preference")
    anchor_records = store.query("纸巾 在哪里", kind="scene_anchor")
    lesson_records = store.query("抓取杯子 失败 视角", kind="task_lesson")

    assert preference_records[0].metadata["value"] == "zh-CN"
    assert anchor_records[0].metadata["location"] == "茶几右上角"
    assert lesson_records[0].metadata["failure_mode"] == "viewpoint_bad"


def test_long_term_memory_prompt_context_groups_structured_sections(tmp_path) -> None:
    store = LongTermMemoryStore(tmp_path / "long_term.jsonl")

    store.remember_user_preference(
        name="response_language",
        value="zh-CN",
        summary="用户偏好中文回答",
    )
    store.record_scene_anchor(
        name="纸巾",
        location="茶几右上角",
        summary="纸巾通常放在茶几右上角",
    )
    store.remember_task_lesson(
        key="pick_cup",
        task="抓取杯子",
        summary="上次抓取失败是因为视角太斜，应先 reposition",
        success=False,
        failure_mode="viewpoint_bad",
        recovery_hint="reposition",
        skill_name="vla_manipulation",
    )
    store.remember_entity(name="杯子", summary="杯子在桌面中央", location="桌面中央")

    context = store.prompt_context("抓取杯子并找纸巾")

    assert context
    assert "- User preferences:" in context
    assert "response_language=zh-CN" in context
    assert "- Scene anchors:" in context
    assert "纸巾 @ 茶几右上角" in context
    assert "- Task lessons:" in context
    assert "failure_mode=viewpoint_bad" in context
    assert "- Other relevant memory:" in context
