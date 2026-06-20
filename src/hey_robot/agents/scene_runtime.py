from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from hey_robot.agents.perception_query import SceneEvidence
from hey_robot.agents.progress import RobotAgentProgress
from hey_robot.agents.runtime.grounding import (
    is_perception_skill_name,
    needs_perception_grounding,
)
from hey_robot.agents.task_runtime import RobotStateCache, TaskRunManager
from hey_robot.logging import HeyRobotLogger
from hey_robot.memory import SceneMemoryRecord, SceneSummarizer
from hey_robot.perception.scene import SceneUnderstanding
from hey_robot.protocol import RobotObservation, RobotStatus, SkillResult

logger = HeyRobotLogger(name="scene")

ProgressCallback = Callable[[RobotAgentProgress], Awaitable[None]]


@dataclass(frozen=True)
class SceneFreshnessAssessment:
    status: str
    needs_refresh: bool
    reason: str
    robot_id: str
    frame_id: int | None = None
    image_count: int = 0
    age_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "needs_refresh": self.needs_refresh,
            "reason": self.reason,
            "robot_id": self.robot_id,
            "frame_id": self.frame_id,
            "image_count": self.image_count,
            "age_sec": self.age_sec,
        }


class SceneRuntime:
    """Scene perception and memory boundary for RobotAgentService."""

    def __init__(
        self,
        *,
        agent_id: str,
        robot_cache: RobotStateCache,
        task_runtime: TaskRunManager,
        captioner: Any,
        max_memory_tasks: int,
        summarizer: SceneSummarizer | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.robot_cache = robot_cache
        self.task_runtime = task_runtime
        self.captioner = captioner
        self.summarizer = summarizer or SceneSummarizer()
        self.max_memory_tasks = max_memory_tasks
        self._memory_tasks: set[asyncio.Task] = set()

    async def stop(self) -> None:
        for task in list(self._memory_tasks):
            task.cancel()
        if self._memory_tasks:
            await asyncio.gather(*self._memory_tasks, return_exceptions=True)

    def schedule_memory(
        self,
        observation: RobotObservation,
        status: RobotStatus | None,
        *,
        progress_callback: ProgressCallback,
    ) -> None:
        if self.max_memory_tasks <= 0:
            return
        if len(self._memory_tasks) >= self.max_memory_tasks:
            return
        task = asyncio.create_task(
            self.record_scene_memory(
                observation, status, progress_callback=progress_callback
            )
        )
        self._memory_tasks.add(task)

        def _done(done: asyncio.Task) -> None:
            self._memory_tasks.discard(done)
            try:
                done.result()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning(
                    f"scene memory 更新失败 agent={self.agent_id}: {type(exc).__name__}: {exc}"
                )

        task.add_done_callback(_done)

    async def record_scene_memory(
        self,
        observation: RobotObservation,
        status: RobotStatus | None,
        *,
        progress_callback: ProgressCallback,
    ) -> SceneMemoryRecord:
        understanding = await self.captioner.caption(observation, status)
        return await self.record_scene_understanding(
            observation,
            status,
            understanding,
            progress_callback=progress_callback,
        )

    async def record_scene_understanding(
        self,
        observation: RobotObservation,
        status: RobotStatus | None,
        understanding: SceneUnderstanding,
        *,
        progress_callback: ProgressCallback,
    ) -> SceneMemoryRecord:
        record = self.summarizer.from_understanding(observation, understanding, status)
        self.task_runtime.record_scene_memory(record)
        if observation.envelope.episode_id:
            await progress_callback(
                RobotAgentProgress(
                    phase="scene_memory",
                    summary=record.summary,
                    episode_id=observation.envelope.episode_id,
                    agent_id=observation.envelope.agent_id or self.agent_id,
                    robot_id=observation.envelope.robot_id,
                    trace_id=observation.envelope.trace_id,
                    metadata={"scene_record": record.to_dict()},
                )
            )
        return record

    async def skill_result_text_for_agent(self, result: SkillResult) -> str:
        base = result.summary or result.status or "skill completed"
        if result.status != "completed" or not is_perception_skill_name(result.name):
            return base
        observation = self.robot_cache.observation.get(result.envelope.robot_id or "")
        if observation is None:
            return base
        status = self.robot_cache.status.get(result.envelope.robot_id or "")
        try:
            understanding = await self.captioner.caption(observation, status)
        except Exception as exc:
            logger.warning(
                f"skill result 场景 caption 失败 agent={self.agent_id}: {type(exc).__name__}: {exc}"
            )
            return (
                f"{base}\n"
                f"Observation frame={observation.frame_id}, images={len(observation.images)}. "
                "Scene caption unavailable."
            )
        summary = understanding.summary.strip()
        if not summary:
            return f"{base}\nObservation frame={observation.frame_id}, images={len(observation.images)}."
        return (
            f"{base}\n"
            f"Observation frame={observation.frame_id}, images={len(observation.images)}.\n"
            f"Scene summary: {summary}"
        )

    async def query_scene_evidence(
        self,
        *,
        robot_id: str | None,
        default_robot: str | None,
        question: str,
        baseline_frame_id: int | None = None,
        freshness: str = "fresh",
        timeout_sec: float = 2.0,
    ) -> SceneEvidence:
        resolved = robot_id or default_robot or ""
        observation = await self._wait_for_scene_observation(
            resolved,
            baseline_frame_id=baseline_frame_id,
            freshness=freshness,
            timeout_sec=timeout_sec,
        )
        if observation is None:
            return SceneEvidence(
                status="no_observation",
                summary="No robot observation is available.",
                metadata={"question": question, "baseline_frame_id": baseline_frame_id},
            )
        status = self.robot_cache.status.get(resolved)
        try:
            understanding = await self.captioner.caption(observation, status)
        except Exception as exc:
            logger.warning(
                f"scene evidence caption 失败 agent={self.agent_id}: {type(exc).__name__}: {exc}"
            )
            return SceneEvidence(
                status="caption_failed",
                frame_id=observation.frame_id,
                image_count=len(observation.images),
                summary=f"Scene caption unavailable: {type(exc).__name__}: {exc}",
                metadata={"question": question, "baseline_frame_id": baseline_frame_id},
            )
        return SceneEvidence.from_understanding(
            observation,
            understanding,
            metadata={"question": question, "baseline_frame_id": baseline_frame_id},
        )

    def assess_turn_freshness(
        self,
        *,
        robot_id: str | None,
        default_robot: str | None,
        text: str,
        max_age_sec: float = 15.0,
    ) -> SceneFreshnessAssessment:
        resolved = robot_id or default_robot or ""
        if not needs_perception_grounding(text):
            return SceneFreshnessAssessment(
                status="not_required",
                needs_refresh=False,
                reason="turn does not require current visual grounding",
                robot_id=resolved,
            )
        observation = self.robot_cache.observation.get(resolved)
        return assess_observation_freshness(
            observation,
            robot_id=resolved,
            max_age_sec=max_age_sec,
        )

    async def _wait_for_scene_observation(
        self,
        robot_id: str,
        *,
        baseline_frame_id: int | None,
        freshness: str,
        timeout_sec: float,
    ) -> RobotObservation | None:
        deadline = time.monotonic() + max(0.0, timeout_sec)
        while True:
            observation = self.robot_cache.observation.get(robot_id)
            if observation is not None and _observation_satisfies(
                observation,
                baseline_frame_id=baseline_frame_id,
                freshness=freshness,
            ):
                return observation
            if time.monotonic() >= deadline:
                return observation
            await asyncio.sleep(0.05)


def _observation_satisfies(
    observation: RobotObservation,
    *,
    baseline_frame_id: int | None,
    freshness: str,
) -> bool:
    if str(freshness or "").lower() not in {"fresh", "new", "latest"}:
        return True
    if baseline_frame_id is None:
        return True
    return observation.frame_id > baseline_frame_id


def assess_observation_freshness(
    observation: RobotObservation | None,
    *,
    robot_id: str,
    max_age_sec: float = 15.0,
) -> SceneFreshnessAssessment:
    if observation is None:
        return SceneFreshnessAssessment(
            status="missing",
            needs_refresh=True,
            reason="no observation is available",
            robot_id=robot_id,
        )
    image_count = len(observation.images)
    if image_count <= 0:
        return SceneFreshnessAssessment(
            status="no_image",
            needs_refresh=True,
            reason="latest observation has no image evidence",
            robot_id=robot_id,
            frame_id=observation.frame_id,
            image_count=image_count,
        )
    age_sec = max(
        0.0, time.time() - float(observation.envelope.timestamp or time.time())
    )
    if age_sec > max(0.0, max_age_sec):
        return SceneFreshnessAssessment(
            status="stale",
            needs_refresh=True,
            reason=f"latest observation is stale for {age_sec:.1f}s",
            robot_id=robot_id,
            frame_id=observation.frame_id,
            image_count=image_count,
            age_sec=age_sec,
        )
    return SceneFreshnessAssessment(
        status="fresh",
        needs_refresh=False,
        reason="latest observation is fresh",
        robot_id=robot_id,
        frame_id=observation.frame_id,
        image_count=image_count,
        age_sec=age_sec,
    )
