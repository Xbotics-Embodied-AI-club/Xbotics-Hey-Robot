from __future__ import annotations

import tempfile
from pathlib import Path

from hey_robot.memory.scene import (
    SceneMemoryRecord,
    SceneMemoryStore,
    SceneSummarizer,
    _dict,
    _sanitize,
)
from hey_robot.protocol import (
    ArtifactRef,
    Envelope,
    ImageRef,
    RobotObservation,
    RobotStatus,
)


def make_observation(
    frame_id: int = 0,
    episode_id: str | None = "ep1",
    robot_id: str | None = "r1",
    images: list[ImageRef] | None = None,
    artifacts: list[ArtifactRef] | None = None,
    task: str | None = None,
    proprioception: list[float] | None = None,
    raw: dict | None = None,
) -> RobotObservation:
    if images is None:
        images = [ImageRef(uri="http://x.com/frame.jpg", camera="cam0")]
    if proprioception is None:
        proprioception = []
    return RobotObservation(
        envelope=Envelope(trace_id="tr1", episode_id=episode_id, robot_id=robot_id),
        frame_id=frame_id,
        images=images,
        artifacts=artifacts or [],
        proprioception=proprioception,
        task=task,
        raw=raw or {},
    )


def make_status(
    frame_id: int | None = 5,
    state: str = "idle",
    skill_id: str | None = "sk1",
    success: bool | None = True,
    error: str | None = None,
    metrics: dict | None = None,
) -> RobotStatus:
    return RobotStatus(
        envelope=Envelope(trace_id="tr1", robot_id="r1"),
        frame_id=frame_id,
        state=state,
        skill_id=skill_id,
        success=success,
        error=error,
        metrics=metrics or {},
    )


class TestSanitize:
    def test_simple(self) -> None:
        assert _sanitize("hello") == "hello"

    def test_path_characters(self) -> None:
        assert _sanitize("hello world!@#") == "hello_world___"

    def test_keeps_safe_chars(self) -> None:
        assert _sanitize("valid-name_123.test") == "valid-name_123.test"

    def test_empty(self) -> None:
        assert _sanitize("") == ""


class TestDict:
    def test_returns_dict(self) -> None:
        assert _dict({"a": 1}) == {"a": 1}

    def test_returns_empty_for_none(self) -> None:
        assert _dict(None) == {}

    def test_returns_empty_for_list(self) -> None:
        assert _dict([1, 2, 3]) == {}


class TestSceneMemoryRecord:
    def test_to_dict(self) -> None:
        record = SceneMemoryRecord(
            record_id="rec1",
            episode_id="ep1",
            robot_id="r1",
            frame_id=5,
            summary="frame=5 images=1 state=idle",
            task="inspect",
        )
        d = record.to_dict()
        assert d["record_id"] == "rec1"
        assert d["task"] == "inspect"

    def test_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        import pytest

        record = SceneMemoryRecord(
            record_id="rec1",
            episode_id="ep1",
            robot_id="r1",
            frame_id=5,
            summary="test",
        )

        with pytest.raises(FrozenInstanceError):
            record.summary = "other"  # type: ignore[misc]


class TestSceneSummarizer:
    def test_summarize_with_images(self) -> None:
        summarizer = SceneSummarizer()
        obs = make_observation(frame_id=10, task="inspect")
        status = make_status(state="running", skill_id="sk1")

        record = summarizer.summarize(obs, status)
        assert record.frame_id == 10
        assert "frame=10" in record.summary
        assert "images=1" in record.summary
        assert record.task == "inspect"
        assert record.image_count == 1
        assert record.confidence == 0.65  # has images
        assert record.episode_id == "ep1"
        assert record.robot_id == "r1"

    def test_summarize_without_images_lower_confidence(self) -> None:
        summarizer = SceneSummarizer()
        obs = make_observation(frame_id=5, images=[])
        status = make_status()

        record = summarizer.summarize(obs, status)
        assert record.image_count == 0
        assert record.confidence == 0.35  # no images

    def test_summarize_with_artifacts(self) -> None:
        summarizer = SceneSummarizer()
        obs = make_observation(
            frame_id=3,
            artifacts=[ArtifactRef(uri="data/depth.pkl", artifact_type="depth")],
        )
        record = summarizer.summarize(obs, make_status())
        assert record.artifact_count == 1

    def test_summarize_with_battery_metrics(self) -> None:
        summarizer = SceneSummarizer()
        obs = make_observation(frame_id=1)
        status = make_status(metrics={"battery": {"status": "normal", "voltage": 12.5}})

        record = summarizer.summarize(obs, status)
        assert "battery=normal" in record.summary
        assert "voltage=12.5" in record.summary
        assert record.battery == {"status": "normal", "voltage": 12.5}

    def test_summarize_with_camera_metrics(self) -> None:
        summarizer = SceneSummarizer()
        obs = make_observation(frame_id=2)
        status = make_status(
            metrics={"camera": {"frame_available": True, "image_shape": [480, 640]}}
        )

        record = summarizer.summarize(obs, status)
        assert "camera=available" in record.summary
        assert record.camera == {"frame_available": True, "image_shape": [480, 640]}

    def test_summarize_with_arm_metrics(self) -> None:
        summarizer = SceneSummarizer()
        obs = make_observation(frame_id=3)
        status = make_status(metrics={"arm_status": {"position": "home"}})

        record = summarizer.summarize(obs, status)
        assert record.arm == {"position": "home"}

    def test_summarize_without_status_falls_back_to_raw(self) -> None:
        summarizer = SceneSummarizer()
        obs = make_observation(
            frame_id=5,
            raw={
                "camera": {"frame_available": False},
                "arm_status": {"position": "extended"},
                "battery": {"status": "low"},
                "state": "idle",
            },
        )
        record = summarizer.summarize(obs, None)
        assert record.camera == {"frame_available": False}
        assert record.arm == {"position": "extended"}
        assert record.battery == {"status": "low"}
        assert "state=idle" in record.summary

    def test_summarize_status_with_error(self) -> None:
        summarizer = SceneSummarizer()
        obs = make_observation(frame_id=8)
        status = make_status(state="error", error="motor timeout", success=False)

        record = summarizer.summarize(obs, status)
        assert record.status["error"] == "motor timeout"
        assert record.status["success"] is False

    def test_summarize_without_task(self) -> None:
        summarizer = SceneSummarizer()
        obs = make_observation(frame_id=1, task=None)
        record = summarizer.summarize(obs, make_status())
        assert record.task is None
        assert "task=" not in record.summary

    def test_from_understanding(self) -> None:
        from hey_robot.perception.scene import SceneObject, SceneUnderstanding

        summarizer = SceneSummarizer()
        obs = make_observation(frame_id=5, task="inspect")
        understanding = SceneUnderstanding(
            summary="A table with a red cup on it",
            confidence=0.9,
            objects=[SceneObject(name="cup", location="table", confidence=0.95)],
        )
        record = summarizer.from_understanding(obs, understanding, make_status())
        assert record.summary == "A table with a red cup on it"
        assert record.confidence == 0.9
        assert record.metadata["source"] == "scene_captioner"
        assert "understanding" in record.metadata


class TestSceneMemoryStore:
    def test_append_and_recent(self) -> None:
        store = SceneMemoryStore(Path(tempfile.mkdtemp()))
        record = SceneMemoryRecord(
            record_id="rec1",
            episode_id="ep1",
            robot_id="r1",
            frame_id=5,
            summary="frame=5 images=1 state=idle",
        )
        stored = store.append(record)
        assert stored.record_id == "rec1"

        recent = store.recent("ep1")
        assert len(recent) == 1
        assert recent[0].record_id == "rec1"

    def test_recent_respects_limit(self) -> None:
        store = SceneMemoryStore(Path(tempfile.mkdtemp()))
        for i in range(10):
            store.append(
                SceneMemoryRecord(
                    record_id=f"rec{i}",
                    episode_id="ep1",
                    robot_id="r1",
                    frame_id=i,
                    summary=f"frame={i}",
                )
            )
        recent = store.recent("ep1", limit=3)
        assert len(recent) == 3
        assert recent[0].frame_id is not None
        assert recent[-1].frame_id is not None
        assert recent[0].frame_id > recent[-1].frame_id  # sorted by timestamp desc

    def test_recent_uses_append_order_when_timestamps_match(self) -> None:
        store = SceneMemoryStore(Path(tempfile.mkdtemp()))
        for i in range(3):
            store.append(
                SceneMemoryRecord(
                    record_id=f"rec{i}",
                    episode_id="ep1",
                    robot_id="r1",
                    frame_id=i,
                    summary=f"frame={i}",
                    timestamp=1.0,
                )
            )
        recent = store.recent("ep1", limit=3)
        assert [record.frame_id for record in recent] == [2, 1, 0]

    def test_recent_without_episode_id(self) -> None:
        store = SceneMemoryStore(Path(tempfile.mkdtemp()))
        store.append(
            SceneMemoryRecord(
                record_id="r1",
                episode_id="ep1",
                robot_id="r1",
                frame_id=1,
                summary="ep1 frame",
            )
        )
        store.append(
            SceneMemoryRecord(
                record_id="r2",
                episode_id="ep2",
                robot_id="r1",
                frame_id=1,
                summary="ep2 frame",
            )
        )
        recent = store.recent()
        assert len(recent) == 2

    def test_recent_nonexistent_episode(self) -> None:
        store = SceneMemoryStore(Path(tempfile.mkdtemp()))
        assert store.recent("nonexistent") == []

    def test_prompt_context_with_records(self) -> None:
        store = SceneMemoryStore(Path(tempfile.mkdtemp()))
        store.append(
            SceneMemoryRecord(
                record_id="r1",
                episode_id="ep1",
                robot_id="r1",
                frame_id=5,
                summary="frame=5 state=idle",
            )
        )
        ctx = store.prompt_context("ep1", limit=5)
        assert ctx is not None
        assert "Recent scene memory:" in ctx
        assert "frame=5" in ctx

    def test_prompt_context_empty(self) -> None:
        store = SceneMemoryStore(Path(tempfile.mkdtemp()))
        assert store.prompt_context("ep1") is None

    def test_prompt_context_ordered_oldest_first(self) -> None:
        store = SceneMemoryStore(Path(tempfile.mkdtemp()))
        store.append(
            SceneMemoryRecord(
                record_id="r1",
                episode_id="ep1",
                robot_id="r1",
                frame_id=1,
                summary="first",
            )
        )
        store.append(
            SceneMemoryRecord(
                record_id="r2",
                episode_id="ep1",
                robot_id="r1",
                frame_id=2,
                summary="second",
            )
        )
        ctx = store.prompt_context("ep1", limit=5)
        assert ctx is not None
        assert ctx.index("first") < ctx.index("second")

    def test_prompt_context_keeps_append_order_when_timestamps_match(self) -> None:
        store = SceneMemoryStore(Path(tempfile.mkdtemp()))
        for i, summary in enumerate(("first", "second", "third"), start=1):
            store.append(
                SceneMemoryRecord(
                    record_id=f"r{i}",
                    episode_id="ep1",
                    robot_id="r1",
                    frame_id=i,
                    summary=summary,
                    timestamp=1.0,
                )
            )
        ctx = store.prompt_context("ep1", limit=2)
        assert ctx is not None
        assert "first" not in ctx
        assert ctx.index("second") < ctx.index("third")

    def test_max_items_clamped_to_one_for_zero(self) -> None:
        store = SceneMemoryStore(Path(tempfile.mkdtemp()), max_items=0)
        assert store.max_items == 1

    def test_max_items_clamped_to_one_for_negative(self) -> None:
        store = SceneMemoryStore(Path(tempfile.mkdtemp()), max_items=-5)
        assert store.max_items == 1

    def test_trim_keeps_most_recent(self) -> None:
        store = SceneMemoryStore(Path(tempfile.mkdtemp()), max_items=5)
        for i in range(10):
            store.append(
                SceneMemoryRecord(
                    record_id=f"rec{i}",
                    episode_id="ep1",
                    robot_id="r1",
                    frame_id=i,
                    summary=f"frame={i}",
                )
            )
        recent = store.recent("ep1", limit=20)
        assert len(recent) == 5
        frame_ids = [r.frame_id for r in recent if r.frame_id is not None]
        assert min(frame_ids) >= 5  # oldest 5 were trimmed

    def test_sanitized_episode_id_filename(self) -> None:
        store = SceneMemoryStore(Path(tempfile.mkdtemp()))
        store.append(
            SceneMemoryRecord(
                record_id="r1",
                episode_id="web/user 1",
                robot_id="r1",
                frame_id=1,
                summary="test",
            )
        )
        recent = store.recent("web/user 1")
        assert len(recent) == 1

    def test_read_skips_corrupted_lines(self) -> None:
        """Store should skip corrupted JSON lines without crashing."""
        root = Path(tempfile.mkdtemp())
        store = SceneMemoryStore(root)
        store.append(
            SceneMemoryRecord(
                record_id="good",
                episode_id="ep1",
                robot_id="r1",
                frame_id=1,
                summary="good",
            )
        )
        # Manually write a bad line to the file
        from hey_robot.memory.scene import _sanitize

        path = root / f"{_sanitize('ep1')}.scene.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write("this is not valid json\n")
        store.append(
            SceneMemoryRecord(
                record_id="also_good",
                episode_id="ep1",
                robot_id="r1",
                frame_id=2,
                summary="also good",
            )
        )
        recent = store.recent("ep1")
        assert len(recent) == 2

    def test_append_skips_duplicate_summary(self) -> None:
        store = SceneMemoryStore(Path(tempfile.mkdtemp()))
        store.append(
            SceneMemoryRecord(
                record_id="rec1",
                episode_id="ep1",
                robot_id="r1",
                frame_id=5,
                summary="NASA行李箱+木门, 清晰可见",
            )
        )
        store.append(
            SceneMemoryRecord(
                record_id="rec2",
                episode_id="ep1",
                robot_id="r1",
                frame_id=6,
                summary="NASA行李箱+木门, 清晰可见",
            )
        )
        recent = store.recent("ep1", limit=10)
        assert len(recent) == 1
        assert recent[0].record_id == "rec1"

    def test_append_allows_different_summary_after_duplicate(self) -> None:
        store = SceneMemoryStore(Path(tempfile.mkdtemp()))
        store.append(
            SceneMemoryRecord(
                record_id="rec1",
                episode_id="ep1",
                robot_id="r1",
                frame_id=5,
                summary="NASA行李箱+木门",
            )
        )
        store.append(
            SceneMemoryRecord(
                record_id="rec2",
                episode_id="ep1",
                robot_id="r1",
                frame_id=7,
                summary="NASA行李箱+木门",
            )
        )
        store.append(
            SceneMemoryRecord(
                record_id="rec3",
                episode_id="ep1",
                robot_id="r1",
                frame_id=9,
                summary="白色门和旁边的桌子",
            )
        )
        recent = store.recent("ep1", limit=10)
        assert len(recent) == 2
        summaries = {r.summary for r in recent}
        assert summaries == {"NASA行李箱+木门", "白色门和旁边的桌子"}

    def test_append_allows_same_summary_after_different_in_between(self) -> None:
        store = SceneMemoryStore(Path(tempfile.mkdtemp()))
        store.append(
            SceneMemoryRecord(
                record_id="rec1",
                episode_id="ep1",
                robot_id="r1",
                frame_id=1,
                summary="NASA行李箱+木门",
            )
        )
        store.append(
            SceneMemoryRecord(
                record_id="rec2",
                episode_id="ep1",
                robot_id="r1",
                frame_id=2,
                summary="左侧桌子",
            )
        )
        store.append(
            SceneMemoryRecord(
                record_id="rec3",
                episode_id="ep1",
                robot_id="r1",
                frame_id=3,
                summary="NASA行李箱+木门",
            )
        )
        recent = store.recent("ep1", limit=10)
        assert len(recent) == 3

    def test_append_does_not_cross_dedup_across_episodes(self) -> None:
        store = SceneMemoryStore(Path(tempfile.mkdtemp()))
        store.append(
            SceneMemoryRecord(
                record_id="rec1",
                episode_id="ep1",
                robot_id="r1",
                frame_id=1,
                summary="NASA行李箱+木门",
            )
        )
        store.append(
            SceneMemoryRecord(
                record_id="rec2",
                episode_id="ep2",
                robot_id="r1",
                frame_id=1,
                summary="NASA行李箱+木门",
            )
        )
        assert len(store.recent("ep1")) == 1
        assert len(store.recent("ep2")) == 1

    def test_append_empty_no_crash(self) -> None:
        store = SceneMemoryStore(Path(tempfile.mkdtemp()))
        record = SceneMemoryRecord(
            record_id="rec1",
            episode_id="ep1",
            robot_id="r1",
            frame_id=1,
            summary="first record",
        )
        stored = store.append(record)
        assert stored.record_id == "rec1"
        assert len(store.recent("ep1")) == 1
