from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from hey_robot.agents.checkpoint import RobotAgentCheckpointStore
from hey_robot.agents.task_run import TaskRunStore
from hey_robot.config import DeploymentConfig
from hey_robot.episode import JsonlEpisodeStore, RobotEpisodeStateStore
from hey_robot.events.store import RuntimeEventStore
from hey_robot.media import LocalMediaStore
from hey_robot.skills import SkillStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect or export a Hey Robot episode"
    )
    sub = parser.add_subparsers(dest="action", required=True)

    inspect = sub.add_parser("inspect", help="Print one materialized episode summary")
    inspect.add_argument("episode_id")
    inspect.add_argument("--config", required=True)
    inspect.add_argument("--limit", type=int, default=20)
    inspect.set_defaults(func=_inspect)

    export = sub.add_parser(
        "export", help="Export episode, skills, events, and media index"
    )
    export.add_argument("episode_id")
    export.add_argument("--config", required=True)
    export.add_argument("--out", default=None)
    export.add_argument("--copy-media", action="store_true")
    export.set_defaults(func=_export)

    args = parser.parse_args()
    args.func(args)


def _bundle(
    config: DeploymentConfig, episode_id: str, *, history_limit: int = 200
) -> dict[str, Any]:
    episodes = JsonlEpisodeStore(config.resources.episodes_root)
    robot_states = RobotEpisodeStateStore(config.resources.episodes_root)
    checkpoints = RobotAgentCheckpointStore(config.resources.episodes_root)
    task_runs = TaskRunStore(config.resources.episodes_root)
    skills = SkillStore(
        Path(config.resources.runtime_dir) / "skills",
        max_items=config.resources.events_max_items,
    )
    events = RuntimeEventStore(
        Path(config.resources.runtime_dir) / "events",
        max_items=config.resources.events_max_items,
    )
    media = LocalMediaStore(
        config.resources.media_root, max_items=config.resources.media_max_items
    )

    history = episodes.history(episode_id, limit=history_limit)
    state = robot_states.load(episode_id)
    checkpoint = checkpoints.load(episode_id)
    tasks = task_runs.list_for_episode(episode_id)
    skill_ids = set()
    if state and state.active_skill_id:
        skill_ids.add(state.active_skill_id)
    for task in tasks:
        skill_ids.update(task.skill_ids)
    skill_records = [
        record for item in skill_ids if (record := skills.get(item)) is not None
    ]
    recent_events = [
        event
        for event in events.recent(config.resources.events_max_items)
        if event.get("episode_id") == episode_id
    ]
    recent_media = media.recent(limit=200)
    return {
        "episode_id": episode_id,
        "history": [record.__dict__ for record in history],
        "robot_state": state.to_dict() if state is not None else None,
        "agent_checkpoint": checkpoint.to_dict() if checkpoint is not None else None,
        "tasks": [task.to_dict() for task in tasks],
        "skills": [record.__dict__ for record in skill_records],
        "events": recent_events,
        "media": recent_media,
    }


def _inspect(args: argparse.Namespace) -> None:
    config = DeploymentConfig.from_yaml(args.config)
    data = _bundle(config, args.episode_id, history_limit=max(1, int(args.limit)))
    state = data["robot_state"] or {}
    tasks = data["tasks"]
    skills = data["skills"]
    lines = [
        f"episode: {args.episode_id}",
        f"history: {len(data['history'])} records",
        f"robot: {state.get('robot_id') or 'unknown'}",
        f"active_task: {state.get('active_task') or (tasks[0]['root_task'] if tasks else 'none')}",
        f"active_skill: {state.get('active_skill_id') or 'none'}",
        f"recovery_required: {bool(state.get('recovery_required'))}",
        f"tasks: {len(tasks)}",
        f"skills: {len(skills)}",
        f"events: {len(data['events'])}",
        f"media_index: {len(data['media'])}",
    ]
    checkpoint = data["agent_checkpoint"]
    if checkpoint:
        lines.append(f"agent_phase: {checkpoint.get('phase')}")
        lines.append(f"pending_turns: {len(checkpoint.get('pending_turns') or [])}")
    sys.stdout.write("\n".join(lines) + "\n")


def _export(args: argparse.Namespace) -> None:
    config = DeploymentConfig.from_yaml(args.config)
    data = _bundle(config, args.episode_id)
    out = (
        Path(args.out)
        if args.out
        else Path(config.resources.runtime_dir)
        / "exports"
        / _default_export_name(args.episode_id)
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "episode_bundle.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.copy_media:
        _copy_media_index(
            data.get("media", []),
            LocalMediaStore(config.resources.media_root),
            out / "media",
        )
    sys.stdout.write(f"exported: {out}\n")


def _copy_media_index(
    items: list[dict[str, Any]], store: LocalMediaStore, target: Path
) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for item in items:
        uri = item.get("uri")
        if not isinstance(uri, str):
            continue
        with contextlib.suppress(Exception):
            source = store.path_for_uri(uri)
            rel = source.relative_to(store.root.resolve())
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)


def _default_export_name(episode_id: str) -> str:
    safe = "".join(
        ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in episode_id
    )
    return f"{safe}_{int(time.time())}"
