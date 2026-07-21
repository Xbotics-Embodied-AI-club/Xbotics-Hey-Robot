"""Run one or more RoboCasa trials through the real Hey Robot Agent entrypoint.

The worker is started separately.  This harness owns trial lifecycle, while
the user request itself is submitted to the HTTP conversation channel.  The
worker's Runtime and ModelService share one EpisodeManager.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx

from evaluation.robocasa365.producers import producer_for
from evaluation.robocasa365.worker.contract import ALLOWED_TASKS, load_manifest
from hey_robot.robot_runtime.robocasa_remote.client import GrpcRoboCasaRuntimeClient


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RoboCasa365 full Hey Robot benchmark")
    parser.add_argument("--task", required=True, choices=sorted(ALLOWED_TASKS))
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--producer", choices=("b0", "b1", "b2"), default="b1")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/evaluation/robocasa365.tasks.yaml"),
    )
    parser.add_argument("--agent-url", default="http://127.0.0.1:8080/turn")
    parser.add_argument("--runtime-target", default="grpc://127.0.0.1:9092")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poll-sec", type=float, default=1.0)
    parser.add_argument("--timeout-sec", type=float, default=1800.0)
    return parser


async def run_trial(args: argparse.Namespace) -> dict[str, object]:
    trial_id = f"trial-{args.task}-{args.seed}-{uuid.uuid4().hex[:8]}"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = load_manifest(args.manifest)
    if args.task not in manifest["tasks"]:
        raise ValueError(f"task {args.task!r} is not in manifest {args.manifest}")
    producer = producer_for(args.producer)
    runtime = GrpcRoboCasaRuntimeClient(
        args.runtime_target, timeout_sec=600.0, role="evaluator"
    )
    data_runtime = GrpcRoboCasaRuntimeClient(
        args.runtime_target, timeout_sec=30.0, role="data"
    )
    started = time.time()
    agent_trace: list[dict[str, object]] = []
    try:
        health = await runtime.health()
        if not health.get("online") or not health.get("loaded"):
            raise RuntimeError(f"RoboCasa runtime is not ready: {health.get('error')}")
        (args.output_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "manifest": manifest,
                    "task": args.task,
                    "seed": args.seed,
                    "objective": args.objective,
                    "producer": producer.name,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "runtime_metadata.json").write_text(
            json.dumps(_runtime_metadata(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        initial = await runtime.begin_trial(
            trial_id=trial_id, task=args.task, seed=args.seed
        )
        (args.output_dir / "trial_spec.json").write_text(
            json.dumps(
                {"trial_id": trial_id, "task": args.task, "seed": args.seed}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "root_task.json").write_text(
            json.dumps({"objective": args.objective, "chat_id": trial_id}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        agent_turn_task = asyncio.create_task(
            _send_agent_turn(
                args.agent_url,
                {
                    "text": producer.prompt(args.objective),
                    "chat_id": trial_id,
                    "sender_id": "robocasa365-benchmark",
                    "metadata": {
                        "trial_id": trial_id,
                        "task": args.task,
                        "seed": args.seed,
                    },
                },
                timeout_sec=args.timeout_sec + 60.0,
            )
        )
        observation = initial
        observations = [{"frame_id": initial.frame_id, "done": initial.done}]
        frames = [initial.images[0].data] if initial.images else []
        event_trace: list[dict[str, object]] = []
        agent_task: dict[str, object] | None = None
        termination_reason = "wall_clock_timeout"
        while time.time() - started < args.timeout_sec:
            if agent_turn_task.done():
                await agent_turn_task
            if observation.done:
                termination_reason = "episode_done"
                break
            await asyncio.sleep(max(0.05, args.poll_sec))
            observation = await data_runtime.observe()
            observations.append(
                {"frame_id": observation.frame_id, "done": observation.done}
            )
            if observation.images:
                frames.append(observation.images[0].data)
            tasks = await asyncio.to_thread(_read_agent_tasks, args.agent_url)
            event_trace.append(
                await asyncio.to_thread(_read_runtime_summary, args.agent_url)
            )
            agent_task = _find_trial_task(
                tasks, objective=args.objective, started=started
            )
            agent_trace.append(
                {
                    "timestamp": time.time(),
                    "frame_id": observation.frame_id,
                    "task": agent_task,
                }
            )
            if agent_task is not None and agent_task.get("status") != "active":
                termination_reason = f"agent_{agent_task['status']}"
                break
        if agent_turn_task.done():
            await agent_turn_task
        truth = await runtime.read_truth()
        evaluator_events = _evaluator_events(truth)
        model_options = [
            item
            for item in evaluator_events
            if item.get("kind") == "model_service_option"
        ]
        actions = [item for item in evaluator_events if item.get("kind") == "action"]
        planner_steps = (
            int(_as_float(agent_task.get("step_count"))) if agent_task else 0
        )
        result = {
            "trial_id": trial_id,
            "task": args.task,
            "seed": args.seed,
            "official_success": bool(truth["official_success"]),
            "episode_done": bool(truth["done"]),
            "frame_id": int(truth["frame_id"]),
            "duration_sec": round(time.time() - started, 3),
            "termination_reason": termination_reason,
            "agent_task": agent_task,
            "producer": producer.name,
            "agent_completion": bool(
                agent_task
                and agent_task.get("status") in {"completed", "failed", "cancelled"}
            ),
            "false_completion": bool(
                agent_task
                and agent_task.get("status") == "completed"
                and not truth["official_success"]
            ),
            "planner_steps": planner_steps,
            "option_count": len(model_options),
            "action_count": len(actions),
            "observation_count": len(observations),
            "failure_stage": _failure_stage(
                termination_reason=termination_reason,
                agent_task=agent_task,
                model_options=model_options,
                official_success=bool(truth["official_success"]),
            ),
        }
        (args.output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (args.output_dir / "agent_trace.jsonl").write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in agent_trace),
            encoding="utf-8",
        )
        (args.output_dir / "observations.jsonl").write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in observations),
            encoding="utf-8",
        )
        (args.output_dir / "agent_events.jsonl").write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in event_trace),
            encoding="utf-8",
        )
        _write_event_artifacts(args.output_dir, event_trace, trial_id=trial_id)
        _write_video(args.output_dir / "video.mp4", frames)
        for name in (
            "model_service_events.jsonl",
            "actions.jsonl",
        ):
            (args.output_dir / name).write_text("", encoding="utf-8")
        (args.output_dir / "evaluator_truth.json").write_text(
            json.dumps(truth, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_worker_event_artifacts(args.output_dir, truth)
        (args.output_dir / "summary.json").write_text(
            json.dumps({"trials": [result], "count": 1}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return result
    finally:
        turn_task = locals().get("agent_turn_task")
        if isinstance(turn_task, asyncio.Task) and not turn_task.done():
            turn_task.cancel()
            with suppress(asyncio.CancelledError):
                await turn_task
        try:
            if "initial" in locals():
                await runtime.end_trial(reason="benchmark_finished")
        except Exception:
            logging.getLogger(__name__).exception("failed to close RoboCasa trial")
        await runtime.close()
        await data_runtime.close()


async def _send_agent_turn(
    url: str, payload: dict[str, object], *, timeout_sec: float
) -> None:
    # The Agent turn runs concurrently with evaluator polling and may include
    # several planner / perception / option cycles before the response closes.
    async with httpx.AsyncClient(timeout=timeout_sec) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()


def _read_agent_tasks(turn_url: str) -> list[dict[str, object]]:
    parsed = urlsplit(turn_url)
    tasks_url = urlunsplit(
        (parsed.scheme, parsed.netloc, "/api/tasks", "limit=100", "")
    )
    with urllib.request.urlopen(tasks_url, timeout=10) as response:  # noqa: S310
        payload = json.load(response)
    tasks = payload.get("tasks", []) if isinstance(payload, dict) else []
    return [item for item in tasks if isinstance(item, dict)]


def _read_runtime_summary(turn_url: str) -> dict[str, object]:
    parsed = urlsplit(turn_url)
    summary_url = urlunsplit(
        (parsed.scheme, parsed.netloc, "/api/runtime-summary", "limit=100", "")
    )
    with urllib.request.urlopen(summary_url, timeout=10) as response:  # noqa: S310
        payload = json.load(response)
    return payload if isinstance(payload, dict) else {}


def _write_video(path: Path, frames: list[bytes]) -> None:
    if not frames:
        return
    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError:
        return
    writer = imageio.get_writer(path, fps=5)
    try:
        for encoded in frames:
            try:
                writer.append_data(imageio.imread(encoded, format="jpg"))  # type: ignore[arg-type]
            except Exception:  # noqa: S112
                continue
    finally:
        writer.close()


def _write_event_artifacts(
    root: Path, snapshots: list[dict[str, object]], *, trial_id: str
) -> None:
    # Runtime summaries are cumulative snapshots. Writing every snapshot again
    # multiplies the same skill records into hundred-megabyte artifacts.
    latest = snapshots[-1] if snapshots else {}
    current_skills = latest.get("skills", [])
    skills = (
        [
            item
            for item in current_skills
            if isinstance(item, dict) and _skill_belongs_to_trial(item, trial_id)
        ]
        if isinstance(current_skills, list)
        else []
    )
    (root / "skill_events.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in skills),
        encoding="utf-8",
    )
    options = [
        item
        for item in skills
        if isinstance(item, dict) and item.get("name") == "robocasa_option"
    ]
    (root / "options.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in options),
        encoding="utf-8",
    )


def _write_worker_event_artifacts(root: Path, truth: dict[str, object]) -> None:
    entries = _evaluator_events(truth)
    model_events = [
        item for item in entries if item.get("kind") == "model_service_option"
    ]
    actions = [item for item in entries if item.get("kind") == "action"]
    (root / "model_service_events.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in model_events),
        encoding="utf-8",
    )
    (root / "actions.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in actions),
        encoding="utf-8",
    )
    # Worker ledger entries are the canonical one-record-per-option artifact.
    (root / "options.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in model_events),
        encoding="utf-8",
    )


def _skill_belongs_to_trial(item: dict[str, object], trial_id: str) -> bool:
    timeline = item.get("timeline", [])
    if not isinstance(timeline, list):
        return False
    for event in timeline:
        if not isinstance(event, dict):
            continue
        envelope = event.get("envelope", {})
        if isinstance(envelope, dict) and envelope.get("chat_id") == trial_id:
            return True
    return False


def _evaluator_events(truth: dict[str, object]) -> list[dict[str, object]]:
    metrics = truth.get("metrics", {})
    events = metrics.get("events", []) if isinstance(metrics, dict) else []
    return [item for item in events if isinstance(item, dict)]


def _failure_stage(
    *,
    termination_reason: str,
    agent_task: dict[str, object] | None,
    model_options: list[dict[str, object]],
    official_success: bool,
) -> str | None:
    if official_success:
        return None
    if termination_reason == "wall_clock_timeout":
        return "planner_or_budget"
    if agent_task and agent_task.get("status") == "completed":
        return "verifier_false_completion"
    if model_options:
        latest = model_options[-1]
        if not latest.get("success"):
            return str(latest.get("failure_mode") or "vla_or_action")
    if termination_reason.startswith("agent_"):
        return "planner_or_skill_os"
    if termination_reason == "episode_done":
        return "environment"
    return "unknown"


def _find_trial_task(
    tasks: list[dict[str, object]], *, objective: str, started: float
) -> dict[str, object] | None:
    matches = [
        task
        for task in tasks
        if task.get("objective") == objective
        and _as_float(task.get("created_at")) >= started - 1.0
    ]
    return max(
        matches, key=lambda item: _as_float(item.get("created_at")), default=None
    )


def _as_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _runtime_metadata() -> dict[str, object]:
    revision = "unknown"
    git = shutil.which("git")
    with suppress(OSError, subprocess.CalledProcessError):
        if git is None:
            return _runtime_metadata_without_git(revision)
        revision = subprocess.check_output(  # noqa: S603
            [git, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    return _runtime_metadata_without_git(revision)


def _runtime_metadata_without_git(revision: str) -> dict[str, object]:
    return {
        "git_revision": revision,
        "python": sys.version,
        "platform": platform.platform(),
        "policy_path": os.environ.get("ROBOCASA_POLICY", ""),
        "policy_revision": os.environ.get("ROBOCASA_POLICY_REVISION", ""),
    }


def main() -> None:
    args = _parser().parse_args()
    result = asyncio.run(run_trial(args))
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    raise SystemExit(0 if result["official_success"] else 2)


if __name__ == "__main__":
    main()
