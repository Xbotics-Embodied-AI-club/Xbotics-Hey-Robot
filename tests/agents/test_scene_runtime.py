from __future__ import annotations

import asyncio
import time

import pytest

from hey_robot.agents.scene_runtime import (
    SceneFreshnessAssessment,
    SceneRuntime,
    _observation_satisfies,
    assess_observation_freshness,
)
from hey_robot.memory import SceneMemoryRecord
from hey_robot.perception.scene import SceneUnderstanding
from hey_robot.protocol import Envelope, RobotObservation, RobotStatus, SkillResult
from hey_robot.protocol.messages import ImageRef


class TestObservationSatisfies:
    def test_freshness_any_returns_true_regardless_of_frame(self) -> None:
        obs = RobotObservation(
            envelope=Envelope(timestamp=time.time()),
            frame_id=5,
            images=[],
        )
        assert _observation_satisfies(obs, baseline_frame_id=10, freshness="any")
        assert _observation_satisfies(obs, baseline_frame_id=10, freshness="old")

    def test_baseline_none_returns_true(self) -> None:
        obs = RobotObservation(
            envelope=Envelope(timestamp=time.time()),
            frame_id=5,
            images=[],
        )
        assert _observation_satisfies(obs, baseline_frame_id=None, freshness="fresh")

    def test_newer_frame_satisfies_fresh(self) -> None:
        obs = RobotObservation(
            envelope=Envelope(timestamp=time.time()),
            frame_id=10,
            images=[],
        )
        assert _observation_satisfies(obs, baseline_frame_id=5, freshness="fresh")

    def test_same_or_older_frame_fails_fresh(self) -> None:
        obs = RobotObservation(
            envelope=Envelope(timestamp=time.time()),
            frame_id=5,
            images=[],
        )
        assert not _observation_satisfies(obs, baseline_frame_id=5, freshness="fresh")
        assert not _observation_satisfies(obs, baseline_frame_id=10, freshness="fresh")

    def test_freshness_new_and_latest_behave_like_fresh(self) -> None:
        obs = RobotObservation(
            envelope=Envelope(timestamp=time.time()),
            frame_id=15,
            images=[],
        )
        assert _observation_satisfies(obs, baseline_frame_id=10, freshness="new")
        assert _observation_satisfies(obs, baseline_frame_id=10, freshness="latest")


class TestAssessObservationFreshness:
    def test_observation_none_returns_missing(self) -> None:
        result = assess_observation_freshness(None, robot_id="r1")
        assert result.status == "missing"
        assert result.needs_refresh is True
        assert "no observation is available" in result.reason

    def test_no_images_returns_no_image(self) -> None:
        obs = RobotObservation(
            envelope=Envelope(timestamp=time.time()),
            frame_id=1,
            images=[],
        )
        result = assess_observation_freshness(obs, robot_id="r1")
        assert result.status == "no_image"
        assert result.needs_refresh is True
        assert result.frame_id == 1

    def test_stale_observation(self) -> None:
        obs = RobotObservation(
            envelope=Envelope(timestamp=time.time() - 100),
            frame_id=5,
            images=[ImageRef(uri="base64...")],
        )
        result = assess_observation_freshness(obs, robot_id="r1", max_age_sec=1.0)
        assert result.status == "stale"
        assert result.needs_refresh is True
        assert result.age_sec is not None
        assert result.age_sec > 1.0

    def test_fresh_observation(self) -> None:
        obs = RobotObservation(
            envelope=Envelope(timestamp=time.time()),
            frame_id=10,
            images=[ImageRef(uri="base64...")],
        )
        result = assess_observation_freshness(obs, robot_id="r1", max_age_sec=60.0)
        assert result.status == "fresh"
        assert result.needs_refresh is False


class FakeCaptioner:
    def __init__(
        self, understanding: SceneUnderstanding | Exception | None = None
    ) -> None:
        self._understanding = understanding
        self.calls: list[tuple] = []

    async def caption(
        self, observation: RobotObservation, status: RobotStatus | None
    ) -> SceneUnderstanding:
        self.calls.append((observation, status))
        if isinstance(self._understanding, Exception):
            raise self._understanding
        if self._understanding is not None:
            return self._understanding
        return SceneUnderstanding(summary="a table with objects")


class FakeTaskRuntime:
    def __init__(self) -> None:
        self.remembered_scenes: list[SceneMemoryRecord] = []

    def record_scene_memory(self, record: SceneMemoryRecord) -> None:
        self.remembered_scenes.append(record)


def _make_scene_runtime(**overrides):
    from hey_robot.agents.task_runtime import RobotStateCache

    cache = overrides.pop("robot_cache", RobotStateCache())
    task_rt = overrides.pop("task_runtime", FakeTaskRuntime())
    captioner = overrides.pop("captioner", FakeCaptioner())
    summarizer = overrides.pop("summarizer", None)
    return SceneRuntime(
        agent_id="agent1",
        robot_cache=cache,
        task_runtime=task_rt,
        captioner=captioner,
        max_memory_tasks=overrides.pop("max_memory_tasks", 2),
        summarizer=summarizer,
    )


def _make_observation(
    episode_id: str = "ep1", frame_id: int = 1, images: int = 1
) -> RobotObservation:
    imgs: list[ImageRef] = [ImageRef(uri=f"img_{i}") for i in range(images)]
    return RobotObservation(
        envelope=Envelope(
            episode_id=episode_id,
            agent_id="agent1",
            robot_id="r1",
            timestamp=time.time(),
        ),
        frame_id=frame_id,
        images=imgs,
    )


class TestSceneRuntimeMemoryScheduling:
    def test_schedule_memory_skips_when_max_tasks_zero(self) -> None:
        rt = _make_scene_runtime(max_memory_tasks=0)
        obs = _make_observation()

        async def noop(_):
            pass

        rt.schedule_memory(obs, None, progress_callback=noop)
        assert len(rt._memory_tasks) == 0

    @pytest.mark.asyncio
    async def test_schedule_memory_skips_when_pool_full(self) -> None:
        rt = _make_scene_runtime(max_memory_tasks=1)
        rt._memory_tasks.add(asyncio.create_task(asyncio.sleep(0.2)))

        async def noop(_):
            pass

        rt.schedule_memory(_make_observation("ep1", 1), None, progress_callback=noop)
        assert len(rt._memory_tasks) == 1

    @pytest.mark.asyncio
    async def test_record_scene_memory_calls_captioner_and_stores_record(self) -> None:
        task_rt = FakeTaskRuntime()
        rt = _make_scene_runtime(task_runtime=task_rt)
        obs = _make_observation()

        async def noop(_):
            pass

        record = await rt.record_scene_memory(obs, None, progress_callback=noop)
        assert isinstance(record, SceneMemoryRecord)
        assert len(task_rt.remembered_scenes) == 1

    @pytest.mark.asyncio
    async def test_record_scene_understanding_stores_record(self) -> None:
        task_rt = FakeTaskRuntime()
        rt = _make_scene_runtime(task_runtime=task_rt)
        obs = _make_observation()
        understanding = SceneUnderstanding(summary="a table with a cup")

        async def noop(_):
            pass

        record = await rt.record_scene_understanding(
            obs, None, understanding, progress_callback=noop
        )
        assert isinstance(record, SceneMemoryRecord)
        assert len(task_rt.remembered_scenes) == 1

    @pytest.mark.asyncio
    async def test_stop_cancels_memory_tasks(self) -> None:
        rt = _make_scene_runtime(max_memory_tasks=5)
        obs = _make_observation()

        async def slow_progress(_):
            pass

        rt.schedule_memory(obs, None, progress_callback=slow_progress)
        assert len(rt._memory_tasks) == 1
        await rt.stop()
        assert len(rt._memory_tasks) == 0


class TestSkillResultText:
    @pytest.mark.asyncio
    async def test_non_completed_result_returns_base(self) -> None:
        rt = _make_scene_runtime()
        result = SkillResult(
            envelope=Envelope(episode_id="ep1", robot_id="r1"),
            skill_id="cmd1",
            name="move_forward",
            status="failed",
            summary="movement failed",
        )

        text = await rt.skill_result_text_for_agent(result)
        assert text == "movement failed"

    @pytest.mark.asyncio
    async def test_non_perception_skill_returns_base(self) -> None:
        rt = _make_scene_runtime()
        result = SkillResult(
            envelope=Envelope(episode_id="ep1", robot_id="r1"),
            skill_id="cmd1",
            name="move_forward",
            status="completed",
            summary="moved 10cm",
        )

        text = await rt.skill_result_text_for_agent(result)
        assert text == "moved 10cm"

    @pytest.mark.asyncio
    async def test_perception_skill_no_observation_in_cache(self) -> None:
        rt = _make_scene_runtime()
        result = SkillResult(
            envelope=Envelope(episode_id="ep1", robot_id="r1"),
            skill_id="cmd1",
            name="inspect_scene",
            status="completed",
            summary="captured",
        )

        text = await rt.skill_result_text_for_agent(result)
        assert text == "captured"

    @pytest.mark.asyncio
    async def test_perception_skill_with_observation_and_caption(self) -> None:
        rt = _make_scene_runtime()
        obs = _make_observation(frame_id=42, images=3)
        status = RobotStatus(
            envelope=Envelope(episode_id="ep1", robot_id="r1", timestamp=time.time()),
            state="idle",
        )
        rt.robot_cache.observation["r1"] = obs
        rt.robot_cache.status["r1"] = status

        result = SkillResult(
            envelope=Envelope(episode_id="ep1", robot_id="r1"),
            skill_id="cmd1",
            name="inspect_scene",
            status="completed",
            summary="captured",
        )

        text = await rt.skill_result_text_for_agent(result)
        assert "captured" in text
        assert "frame=42" in text
        assert "images=3" in text

    @pytest.mark.asyncio
    async def test_perception_skill_caption_failure(self) -> None:
        rt = _make_scene_runtime(captioner=FakeCaptioner(RuntimeError("model offline")))
        obs = _make_observation(frame_id=7, images=1)
        rt.robot_cache.observation["r1"] = obs

        result = SkillResult(
            envelope=Envelope(episode_id="ep1", robot_id="r1"),
            skill_id="cmd1",
            name="inspect_scene",
            status="completed",
            summary="inspected",
        )

        text = await rt.skill_result_text_for_agent(result)
        assert "Scene caption unavailable" in text


class TestSceneFreshnessAssessment:
    def test_to_dict_roundtrip(self) -> None:
        assessment = SceneFreshnessAssessment(
            status="fresh",
            needs_refresh=False,
            reason="ok",
            robot_id="r1",
            frame_id=10,
            image_count=3,
            age_sec=0.5,
        )
        d = assessment.to_dict()
        assert d["status"] == "fresh"
        assert d["needs_refresh"] is False
        assert d["frame_id"] == 10
        assert d["age_sec"] == 0.5

    def test_defaults(self) -> None:
        assessment = SceneFreshnessAssessment(
            status="not_required",
            needs_refresh=False,
            reason="no visual grounding needed",
            robot_id="r1",
        )
        assert assessment.frame_id is None
        assert assessment.image_count == 0
        assert assessment.age_sec is None
