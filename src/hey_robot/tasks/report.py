from __future__ import annotations

from dataclasses import asdict
from typing import Any

from hey_robot.agents.checkpoint import RobotAgentCheckpointStore
from hey_robot.agents.task_run import TaskRunStore
from hey_robot.episode import RobotEpisodeStateStore
from hey_robot.memory.scene import SceneMemoryStore
from hey_robot.skills import SkillStore
from hey_robot.tasks.view import TaskSessionQueryService


def build_task_report(
    *,
    episode_id: str,
    task_store: TaskRunStore,
    checkpoint_store: RobotAgentCheckpointStore,
    robot_states: RobotEpisodeStateStore,
    skill_store: SkillStore,
    scene_memory: SceneMemoryStore | None = None,
) -> dict[str, Any]:
    tasks = task_store.list_for_episode(episode_id)
    task = tasks[0] if tasks else None
    checkpoint = checkpoint_store.load(episode_id)
    robot_state = robot_states.load(episode_id)
    skill_ids = _skill_ids(task, checkpoint.skill_id if checkpoint else None)
    skills = [
        record
        for skill_id in skill_ids
        if (record := skill_store.get(skill_id)) is not None
    ]
    scene_memory = scene_memory or SceneMemoryStore(
        task_store.root.parent / "scene_memory"
    )
    scene_events = task_store.events.recent(episode_id, limit=20, kind="scene_observed")

    session_view = None
    if task is not None:
        query = TaskSessionQueryService(
            task_store=task_store,
            scene_memory=scene_memory,
            skill_store=skill_store,
        )
        session_view = query.view_for_episode(episode_id)

    return {
        "episode_id": episode_id,
        "task": task.to_dict() if task else None,
        "checkpoint": checkpoint.to_dict() if checkpoint else None,
        "robot_state": robot_state.to_dict() if robot_state else None,
        "skills": [asdict(record) for record in skills],
        "session": session_view.to_dict() if session_view else None,
        "scene_events": [event.to_dict() for event in scene_events],
        "recovery": task.recovery if task else None,
        "summary": _summary(task, skills),
    }


def _skill_ids(task: Any | None, active_skill_id: str | None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for skill_id in [active_skill_id, *(task.skill_ids if task else [])]:
        if skill_id and skill_id not in seen:
            seen.add(skill_id)
            out.append(skill_id)
    return out


def _summary(task: Any | None, skills: list[Any]) -> dict[str, Any]:
    terminal = [skill for skill in skills if getattr(skill, "terminal", False)]
    return {
        "status": task.status if task else "unknown",
        "task_success": task.task_success if task else None,
        "last_step_success": task.last_step_success if task else None,
        "skill_count": len(skills),
        "terminal_skill_count": len(terminal),
        "recovery_count": task.recovery_count if task else 0,
        "retry_count": task.retry_count if task else 0,
        "failure_reason": task.failure_reason if task else None,
        "started_at": task.created_at if task else None,
        "finished_at": task.finished_at if task else None,
    }
