"""Run one or more RoboCasa trials through the real Hey Robot Agent entrypoint.

The managed backend is started by ``hey-robot run``. This harness owns the
evaluator trial lifecycle, while the user request itself is submitted to the
HTTP conversation channel. The runtime owns the co-located foundation policy.
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
from hey_robot.robocasa_backend.contract import (
    ALLOWED_TASKS,
    load_manifest,
)
from hey_robot.robot_backends.robocasa_remote.client import GrpcRoboCasaRuntimeClient


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RoboCasa365 full-system benchmark")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--task", action="append", choices=sorted(ALLOWED_TASKS), default=[]
    )
    parser.add_argument("--suite", action="append", default=[])
    parser.add_argument("--seeds", default="1000")
    parser.add_argument(
        "--objective",
        help=(
            "Optional language-instruction override. By default the benchmark uses "
            "the canonical instruction returned by the live RoboCasa environment."
        ),
    )
    parser.add_argument(
        "--condition",
        action="append",
        choices=("b0", "b1"),
        default=[],
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("evaluation/robocasa365/tasks.yaml"),
    )
    parser.add_argument(
        "--split",
        help="Optional RoboCasa dataset split override (for example, pretrain).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/robocasa365.rldx.yaml"),
        help="RLDX policy deployment configuration",
    )
    parser.add_argument("--agent-url", default="http://127.0.0.1:8080/turn")
    parser.add_argument("--runtime-target", default="grpc://127.0.0.1:9092")
    parser.add_argument(
        "--credentials-file",
        type=Path,
        default=Path("runtime/robocasa365.rldx/robocasa.credentials.json"),
    )
    parser.add_argument("--poll-sec", type=float, default=1.0)
    parser.add_argument(
        "--agent-task-startup-timeout-sec",
        type=float,
        default=180.0,
        help=(
            "Maximum time to wait for the asynchronously accepted agent turn "
            "to create its task before reporting agent_no_task."
        ),
    )
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
        if spec.enabled
        and spec.type == "robot_policy"
        and str(spec.settings.get("runtime") or "") in {"lerobot", "rldx", "xiaomi"}
        and str(spec.settings.get("embodiment") or "") == "robocasa"
    ]
    if len(model_candidates) != 1:
        raise ValueError(
            "config must contain exactly one supported RoboCasa robot_policy service"
        )
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
    started = time.time()
    agent_trace: list[dict[str, object]] = []
    try:
        health = await runtime.health()
        if not health.get("online") or not health.get("loaded"):
            raise RuntimeError(f"RoboCasa runtime is not ready: {health.get('error')}")
        (args.output_dir / "runtime_metadata.json").write_text(
            json.dumps(
                _runtime_metadata(
                    config_path=args.config,
                    model_service_id=model_service_id,
                    model_settings=dict(model_spec.settings),
                    model_health={"ownership": "co_located_in_robocasa_backend"},
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
            split=str(args.split or manifest["split"]),
            registries=tuple(manifest["registries"]),
            execution_artifact_dir=str(args.output_dir),
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
        last_recorded_frame = initial.frame_id
        runtime_summary: dict[str, object] = {}
        agent_task: dict[str, object] | None = None
        termination_reason = "wall_clock_timeout"
        runtime_error: str | None = None
        while time.time() - started < args.timeout_sec:
            if agent_turn_task.done():
                await agent_turn_task
            if observation.done:
                termination_reason = "episode_done"
                break
            await asyncio.sleep(max(0.05, args.poll_sec))
            tasks = await asyncio.to_thread(_read_agent_tasks, args.agent_url)
            runtime_summary = await asyncio.to_thread(
                _read_runtime_summary, args.agent_url
            )
            agent_task = _find_trial_task(
                tasks, objective=agent_objective, started=started
            )
            # A policy option owns the simulator exclusively. Do not send an
            # Observe RPC while it is stepping: that would queue behind the
            # option and turn the evaluator's short polling timeout into a
            # false runtime failure. The completed option returns its final
            # observation before the agent becomes observable again.
            if agent_task is not None and agent_task.get("status") == "active":
                # The evaluator may read only its own official truth. A very
                # short deadline makes this non-intrusive while an option owns
                # the simulator, yet catches environment termination as soon
                # as the option releases it.
                try:
                    live_truth = await asyncio.wait_for(
                        runtime.read_truth(), timeout=1.0
                    )
                except TimeoutError:
                    live_truth = None
                if live_truth and bool(live_truth["done"]):
                    termination_reason = "episode_done"
                    break
                agent_trace.append(
                    {
                        "timestamp": time.time(),
                        "frame_id": last_recorded_frame,
                        "task": agent_task,
                    }
                )
                continue
            try:
                observation = await data_runtime.observe()
            except Exception as exc:
                runtime_error = f"{type(exc).__name__}: {exc}"
                termination_reason = "runtime_unavailable"
                break
            if observation.frame_id != last_recorded_frame or observation.done:
                observations.append(
                    {"frame_id": observation.frame_id, "done": observation.done}
                )
                last_recorded_frame = observation.frame_id
            agent_trace.append(
                {
                    "timestamp": time.time(),
                    "frame_id": observation.frame_id,
                    "task": agent_task,
                }
            )
            # /turn acknowledges durable message intake, not task creation.
            # Initializing a local policy or model can briefly delay the agent
            # consumer, so absence from /api/tasks immediately after that ACK
            # is a race rather than an execution failure.
            if (
                agent_turn_task.done()
                and agent_task is None
                and time.time() - started
                >= float(getattr(args, "agent_task_startup_timeout_sec", 180.0))
            ):
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
                    latest_option = terminal_options[-1]
                    termination_reason = (
                        "option_failed"
                        if latest_option.get("phase") in {"failed", "cancelled"}
                        or latest_option.get("success") is False
                        else "condition_manipulate_limit"
                    )
                    break
            if agent_task is not None and agent_task.get("status") != "active":
                termination_reason = f"agent_{agent_task['status']}"
                break
        if agent_turn_task.done():
            await agent_turn_task
        try:
            truth = await asyncio.wait_for(runtime.read_truth(), timeout=30.0)
        except Exception as exc:
            runtime_error = runtime_error or f"{type(exc).__name__}: {exc}"
            truth = {
                "done": False,
                "official_success": False,
                "frame_id": last_recorded_frame,
                "metrics": {"runtime_error": runtime_error},
            }
        if truth["done"] and truth["official_success"] and agent_task:
            await _mark_environment_complete(
                args.agent_url,
                str(agent_task["task_id"]),
                reason="RoboCasa official evaluator reported episode completion.",
            )
            tasks = await asyncio.to_thread(_read_agent_tasks, args.agent_url)
            agent_task = _find_trial_task(
                tasks, objective=agent_objective, started=started
            )
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
            "error": runtime_error,
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
        compact_events_value = runtime_summary.get("events", [])
        compact_events = (
            compact_events_value if isinstance(compact_events_value, list) else []
        )
        (args.output_dir / "agent_events.jsonl").write_text(
            "".join(
                json.dumps(item, sort_keys=True) + "\n"
                for item in compact_events
                if isinstance(item, dict)
            ),
            encoding="utf-8",
        )
        _write_event_artifacts(args.output_dir, [runtime_summary], trial_id=trial_id)
        # The backend owns the simulator and closes the writer after the last
        # environment step. The evaluator never samples frames for recording.
        await asyncio.wait_for(
            runtime.end_trial(reason="benchmark_artifacts_finalized"), timeout=30.0
        )
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
                await asyncio.wait_for(
                    runtime.end_trial(reason="benchmark_finished"), timeout=10.0
                )
        except Exception:
            logging.getLogger(__name__).exception("failed to close RoboCasa trial")
        await runtime.close()
        await data_runtime.close()


async def run_batch(args: argparse.Namespace) -> dict[str, object]:
    """Run selected tasks sequentially; one task is simply batch size one."""
    manifest = load_manifest(args.manifest)
    tasks = list(getattr(args, "task", []))
    if not tasks:
        suites = args.suite or sorted(manifest["suites"])
        invalid = sorted(set(suites) - set(manifest["suites"]))
        if invalid:
            raise ValueError(f"unknown manifest suites: {invalid}")
        tasks = [task for suite in suites for task in manifest["suites"][suite]]
    conditions = args.condition or ["b1"]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    args.output_root.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, object]] = []
    for task in tasks:
        for seed in seeds:
            for condition in conditions:
                output_dir = args.output_root / "trials" / f"{condition}-{task}-{seed}"
                trial_args = argparse.Namespace(
                    task=task,
                    seed=seed,
                    objective=getattr(args, "objective", None),
                    condition=condition,
                    manifest=args.manifest,
                    split=getattr(args, "split", None),
                    config=args.config,
                    agent_url=args.agent_url,
                    runtime_target=args.runtime_target,
                    credentials_file=args.credentials_file,
                    output_dir=output_dir,
                    poll_sec=args.poll_sec,
                    agent_task_startup_timeout_sec=getattr(
                        args, "agent_task_startup_timeout_sec", 180.0
                    ),
                    timeout_sec=args.timeout_sec,
                )
                try:
                    result = await run_trial(trial_args)
                except Exception as exc:
                    result = {
                        "task": task,
                        "seed": seed,
                        "condition": condition,
                        "official_success": False,
                        "false_completion": False,
                        "failure_stage": "trial_exception",
                        "termination_reason": "trial_exception",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    output_dir.mkdir(parents=True, exist_ok=True)
                    (output_dir / "result.json").write_text(
                        json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                results.append(result)
    summary = {
        "manifest": manifest,
        "count": len(results),
        "trials": results,
        "official_successes": sum(bool(item["official_success"]) for item in results),
        "false_completions": sum(bool(item["false_completion"]) for item in results),
        "trial_errors": sum(
            item.get("failure_stage") == "trial_exception" for item in results
        ),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


async def _send_agent_turn(
    url: str, payload: dict[str, object], *, timeout_sec: float
) -> None:
    # The Agent turn runs concurrently with evaluator polling and may include
    # several planner / perception / option cycles before the response closes.
    async with httpx.AsyncClient(timeout=timeout_sec) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()


async def _mark_environment_complete(
    agent_url: str, task_id: str, *, reason: str
) -> None:
    parsed = urlsplit(agent_url)
    endpoint = urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            f"/api/tasks/{task_id}/environment-complete",
            "",
            "",
        )
    )
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(endpoint, json={"reason": reason})
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
    envelope = item.get("envelope", {})
    if isinstance(envelope, dict) and envelope.get("chat_id") == trial_id:
        return True
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
    if termination_reason == "option_failed":
        return "vla_or_action"
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
    summary = asyncio.run(run_batch(_parser().parse_args()))
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    raise SystemExit(0 if summary["official_successes"] == summary["count"] else 2)


if __name__ == "__main__":
    main()
