"""Run one or more RoboCasa trials through the real Hey Robot Agent entrypoint.

The managed backend is started by ``hey-robot run``. This harness owns the
evaluator trial lifecycle, while the user request itself is submitted to the
HTTP conversation channel. Runtime and ModelService share one EpisodeManager.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
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

from evaluation.robocasa365.conditions import condition_for
from hey_robot.config import DeploymentConfig
from hey_robot.foundation.transport.grpc.client import GrpcModelServiceClient
from hey_robot.robot_runtime.robocasa_remote.client import GrpcRoboCasaRuntimeClient
from hey_robot.robot_runtime.robocasa_remote.contract import (
    ALLOWED_TASKS,
    load_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RoboCasa365 full Hey Robot benchmark")
    parser.add_argument("--task", required=True, choices=sorted(ALLOWED_TASKS))
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--objective",
        help=(
            "Optional language-instruction override. By default the benchmark uses "
            "the canonical instruction returned by the live RoboCasa environment."
        ),
    )
    parser.add_argument("--condition", choices=("b0", "b1", "b2"), default="b1")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/evaluation/robocasa365.tasks.yaml"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/robocasa365.agent.yaml"),
        help="Canonical Hey Robot deployment configuration",
    )
    parser.add_argument("--agent-url", default="http://127.0.0.1:8080/turn")
    parser.add_argument("--runtime-target", default="grpc://127.0.0.1:9092")
    parser.add_argument(
        "--credentials-file",
        type=Path,
        default=Path("runtime/robocasa365.agent/robocasa.credentials.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poll-sec", type=float, default=1.0)
    parser.add_argument("--timeout-sec", type=float, default=7200.0)
    return parser


async def run_trial(args: argparse.Namespace) -> dict[str, object]:
    trial_id = f"trial-{args.task}-{args.seed}-{uuid.uuid4().hex[:8]}"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = load_manifest(args.manifest)
    if args.task not in manifest["tasks"]:
        raise ValueError(f"task {args.task!r} is not in manifest {args.manifest}")
    condition = condition_for(args.condition)
    config = DeploymentConfig.from_yaml(args.config)
    model_candidates = [
        (service_id, spec)
        for service_id, spec in config.model_services.items()
        if spec.enabled and spec.type == "robocasa_lerobot_policy"
    ]
    if len(model_candidates) != 1:
        raise ValueError("config must contain exactly one robocasa_lerobot_policy")
    model_service_id, model_spec = model_candidates[0]
    credentials = json.loads(args.credentials_file.read_text(encoding="utf-8"))
    runtime = GrpcRoboCasaRuntimeClient(
        args.runtime_target,
        timeout_sec=600.0,
        role="evaluator",
        token=str(credentials["evaluator_token"]),
    )
    data_runtime = GrpcRoboCasaRuntimeClient(
        args.runtime_target,
        timeout_sec=30.0,
        role="data",
        token=str(credentials["data_token"]),
    )
    model_service = GrpcModelServiceClient(
        model_service_id,
        model_spec,
        auth_token=str(credentials["data_token"]),
    )
    started = time.time()
    agent_trace: list[dict[str, object]] = []
    try:
        health = await runtime.health()
        if not health.get("online") or not health.get("loaded"):
            raise RuntimeError(f"RoboCasa runtime is not ready: {health.get('error')}")
        model_health = await model_service.health()
        if not model_health.online or not model_health.loaded:
            raise RuntimeError(
                f"RoboCasa model service is not ready: {model_health.error}"
            )
        (args.output_dir / "runtime_metadata.json").write_text(
            json.dumps(
                _runtime_metadata(
                    config_path=args.config,
                    model_service_id=model_service_id,
                    model_settings=dict(model_spec.settings),
                    model_health={
                        "name": model_health.name,
                        "version": model_health.version,
                        "metrics": dict(model_health.metrics),
                    },
                    runtime_health=health,
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        initial = await runtime.begin_trial(
            trial_id=trial_id,
            task=args.task,
            seed=args.seed,
            split=str(manifest["split"]),
            registries=tuple(manifest["registries"]),
        )
        official_objective = str(
            initial.metadata.get("policy_task") or initial.task
        ).strip()
        root_objective = str(args.objective or official_objective).strip()
        if not root_objective:
            raise ValueError("RoboCasa trial objective is empty")
        agent_objective = condition.prompt(root_objective)
        (args.output_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "manifest": manifest,
                    "task": args.task,
                    "seed": args.seed,
                    "objective": root_objective,
                    "official_objective": official_objective,
                    "objective_source": (
                        "cli_override" if args.objective else "environment"
                    ),
                    "condition": condition.name,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        confirmed_spec = {
            "trial_id": initial.episode_id,
            "task": initial.task,
            "seed": initial.metadata.get("seed"),
            "split": initial.metadata.get("split"),
            "registries": initial.metadata.get("registries"),
        }
        (args.output_dir / "trial_spec.json").write_text(
            json.dumps(confirmed_spec, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "root_task.json").write_text(
            json.dumps(
                {
                    "objective": root_objective,
                    "official_objective": official_objective,
                    "objective_source": (
                        "cli_override" if args.objective else "environment"
                    ),
                    "chat_id": trial_id,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        agent_turn_task = asyncio.create_task(
            _send_agent_turn(
                args.agent_url,
                {
                    "text": agent_objective,
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
        last_recorded_frame = initial.frame_id
        runtime_summary: dict[str, object] = {}
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
            if observation.frame_id != last_recorded_frame or observation.done:
                observations.append(
                    {"frame_id": observation.frame_id, "done": observation.done}
                )
                if observation.images:
                    frames.append(observation.images[0].data)
                last_recorded_frame = observation.frame_id
            tasks = await asyncio.to_thread(_read_agent_tasks, args.agent_url)
            runtime_summary = await asyncio.to_thread(
                _read_runtime_summary, args.agent_url
            )
            agent_task = _find_trial_task(
                tasks, objective=agent_objective, started=started
            )
            agent_trace.append(
                {
                    "timestamp": time.time(),
                    "frame_id": observation.frame_id,
                    "task": agent_task,
                }
            )
            if agent_turn_task.done() and agent_task is None:
                termination_reason = "agent_no_task"
                break
            if condition.manipulate_call_limit is not None:
                terminal_options = [
                    item
                    for item in _option_records([runtime_summary], trial_id=trial_id)
                    if item.get("phase") in {"completed", "failed", "cancelled"}
                    or item.get("ended_at") is not None
                ]
                if len(terminal_options) >= condition.manipulate_call_limit:
                    termination_reason = "condition_manipulate_limit"
                    break
            if agent_task is not None and agent_task.get("status") != "active":
                termination_reason = f"agent_{agent_task['status']}"
                break
        if agent_turn_task.done():
            await agent_turn_task
        truth = await runtime.read_truth()
        evaluator_events = _evaluator_events(truth)
        model_options = _option_records([runtime_summary], trial_id=trial_id)
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
            "condition": condition.name,
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
        compact_events = runtime_summary.get("events", [])
        (args.output_dir / "agent_events.jsonl").write_text(
            "".join(
                json.dumps(item, sort_keys=True) + "\n"
                for item in compact_events
                if isinstance(item, dict)
            ),
            encoding="utf-8",
        )
        _write_event_artifacts(args.output_dir, [runtime_summary], trial_id=trial_id)
        _write_video(args.output_dir / "video.mp4", frames)
        (args.output_dir / "evaluator_truth.json").write_text(
            json.dumps(truth, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_evaluator_action_artifact(args.output_dir, truth)
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
        await model_service.close()


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
                writer.append_data(imageio.imread(encoded))  # type: ignore[arg-type]
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
        if isinstance(item, dict) and item.get("name") == "manipulate"
    ]
    (root / "options.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in options),
        encoding="utf-8",
    )


def _write_evaluator_action_artifact(root: Path, truth: dict[str, object]) -> None:
    entries = _evaluator_events(truth)
    actions = [item for item in entries if item.get("kind") == "action"]
    (root / "actions.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in actions),
        encoding="utf-8",
    )
    # Option lifecycle belongs to Skill OS and is written by
    # _write_event_artifacts. The evaluator ledger is canonical only for
    # simulator actions and privileged truth.


def _option_records(
    snapshots: list[dict[str, object]], *, trial_id: str
) -> list[dict[str, object]]:
    latest = snapshots[-1] if snapshots else {}
    skills = latest.get("skills", [])
    if not isinstance(skills, list):
        return []
    return [
        item
        for item in skills
        if isinstance(item, dict)
        and item.get("name") == "manipulate"
        and _skill_belongs_to_trial(item, trial_id)
    ]


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
    if termination_reason == "condition_manipulate_limit":
        return "condition_budget"
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


def _runtime_metadata(
    *,
    config_path: Path,
    model_service_id: str,
    model_settings: dict[str, object],
    model_health: dict[str, object],
    runtime_health: dict[str, object],
) -> dict[str, object]:
    revision = "unknown"
    git = shutil.which("git")
    with suppress(OSError, subprocess.CalledProcessError):
        if git is None:
            return _runtime_metadata_without_git(
                revision,
                config_path=config_path,
                model_service_id=model_service_id,
                model_settings=model_settings,
                model_health=model_health,
                runtime_health=runtime_health,
            )
        revision = subprocess.check_output(  # noqa: S603
            [git, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    return _runtime_metadata_without_git(
        revision,
        config_path=config_path,
        model_service_id=model_service_id,
        model_settings=model_settings,
        model_health=model_health,
        runtime_health=runtime_health,
    )


def _runtime_metadata_without_git(
    revision: str,
    *,
    config_path: Path,
    model_service_id: str,
    model_settings: dict[str, object],
    model_health: dict[str, object],
    runtime_health: dict[str, object],
) -> dict[str, object]:
    return {
        "git_revision": revision,
        "python": sys.version,
        "platform": platform.platform(),
        "config_path": str(config_path),
        "model_service_id": model_service_id,
        "model_settings": model_settings,
        "model_health": model_health,
        "runtime_health": runtime_health,
    }


def main() -> None:
    args = _parser().parse_args()
    result = asyncio.run(run_trial(args))
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    raise SystemExit(0 if result["official_success"] else 2)


if __name__ == "__main__":
    main()
