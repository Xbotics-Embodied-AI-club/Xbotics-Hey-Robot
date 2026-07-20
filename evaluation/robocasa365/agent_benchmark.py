"""RoboCasa365 embodied-agent benchmark entry point.

This is evaluation code, not Hey Robot core runtime code.  It drives the
RoboCasa365 worker-side option runner through the B1/B2 embodied-agent path:
root task -> bounded option command -> VLA -> native 12-D RoboCasa actions ->
environment success predicate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.robocasa365.worker.option_runner import (
    OptionRequest,
    RoboCasaOptionRunner,
)
from evaluation.robocasa365.worker.rollout import (
    ALLOWED_TASKS,
    DEFAULT_POLICY,
    DEFAULT_SPLIT,
    _versions,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RoboCasa365 through the embodied-agent option path"
    )
    parser.add_argument("--task", default="CloseFridge", choices=sorted(ALLOWED_TASKS))
    parser.add_argument(
        "--tasks",
        help="Comma-separated task list. Overrides --task when provided.",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--seeds",
        help="Comma-separated seed list. Overrides --seed/--episodes when provided.",
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--experiment-id", default="robocasa365-agent")
    parser.add_argument("--group-id", default="agent-vla")
    parser.add_argument(
        "--protocol",
        default="vision-only",
        choices=("vision-only", "privileged-oracle"),
    )
    parser.add_argument(
        "--policy-path", default=os.environ.get("ROBOCASA_POLICY", DEFAULT_POLICY)
    )
    parser.add_argument(
        "--option-command",
        help="Language command sent to the low-level VLA. Defaults to the task name.",
    )
    parser.add_argument(
        "--option-plan-json",
        type=Path,
        help=(
            "Optional JSON file mapping task names to a list of bounded option "
            "commands. This is the B2 oracle-decomposition path."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument(
        "--device", default=os.environ.get("ROBOCASA_POLICY_DEVICE", "cuda")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    tasks = _task_list(args)
    seeds = _seed_list(args)
    if args.episodes < 1 and not args.seeds:
        raise SystemExit("--episodes must be >= 1")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    option_plan = _load_option_plan(args)

    runner = RoboCasaOptionRunner()
    manifest: dict[str, Any] = {
        "benchmark": "robocasa365",
        "mode": "embodied_agent_option",
        "experiment_id": args.experiment_id,
        "group_id": args.group_id,
        "protocol": args.protocol,
        "tasks": tasks,
        "split": DEFAULT_SPLIT,
        "episodes_per_task": len(seeds),
        "seeds": seeds,
        "policy_path": args.policy_path,
        "option_command": args.option_command,
        "option_plan_json": str(args.option_plan_json)
        if args.option_plan_json
        else None,
        "max_steps": args.max_steps,
        "device": args.device,
        "versions": _versions(),
        "started_at_unix": time.time(),
        "trial_results": [],
    }
    (output_dir / "agent_benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    successes = 0
    trial_index = 0
    for task in tasks:
        commands = _commands_for_task(task, args, option_plan)
        for episode_index, seed in enumerate(seeds):
            trial_dir = output_dir / f"trial_{trial_index:04d}_{task}_{seed}"
            trial_dir.mkdir()
            payload = _run_trial(
                runner=runner,
                task=task,
                commands=commands,
                seed=seed,
                episode_index=episode_index,
                trial_index=trial_index,
                args=args,
                trial_dir=trial_dir,
            )
            manifest["trial_results"].append(payload)
            successes += int(payload["success"])
            trial_index += 1

    manifest["finished_at_unix"] = time.time()
    manifest["trials"] = trial_index
    manifest["success_count"] = successes
    manifest["success_rate"] = successes / trial_index if trial_index else 0.0
    (output_dir / "agent_benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    raise SystemExit(0 if successes == trial_index else 2)


def _run_trial(
    *,
    runner: RoboCasaOptionRunner,
    task: str,
    commands: list[str],
    seed: int,
    episode_index: int,
    trial_index: int,
    args: argparse.Namespace,
    trial_dir: Path,
) -> dict[str, Any]:
    started_at = time.time()
    session_id = f"agent-{task}-{seed}-{trial_index}"
    option_results: list[dict[str, Any]] = []
    success = False
    for option_index, command in enumerate(commands):
        result = runner.run(
            OptionRequest(
                skill_id=f"{session_id}-option-{option_index}",
                session_id=session_id,
                objective=command,
                task=task,
                option_command=command,
                seed=seed,
                max_steps=args.max_steps,
                policy_path=args.policy_path,
                close_episode=option_index == len(commands) - 1,
                device=args.device,
            )
        )
        payload = result.to_dict()
        payload["option_index"] = option_index
        payload["command"] = command
        option_results.append(payload)
        (trial_dir / f"option_{option_index:04d}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if result.success:
            success = True
            if not payload.get("metrics", {}).get("episode_done"):
                runner.close_session(session_id)
            break

    if not success:
        runner.close_session(session_id)

    result_payload = {
        "benchmark": "robocasa365",
        "mode": "embodied_agent_option",
        "task": task,
        "seed": seed,
        "episode_index": episode_index,
        "trial_index": trial_index,
        "session_id": session_id,
        "policy_path": args.policy_path,
        "commands": commands,
        "options_per_episode": len(option_results),
        "planner_calls": len(option_results),
        "success": success,
        "status": "completed" if success else "failed",
        "failure_mode": None if success else _last_failure_mode(option_results),
        "started_at_unix": started_at,
        "finished_at_unix": time.time(),
        "option_results": option_results,
    }
    (trial_dir / "result.json").write_text(
        json.dumps(result_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result_payload


def _task_list(args: argparse.Namespace) -> list[str]:
    if not args.tasks:
        return [args.task]
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    unknown = sorted(set(tasks) - set(ALLOWED_TASKS))
    if unknown:
        raise SystemExit(f"unknown task(s): {', '.join(unknown)}")
    return tasks


def _seed_list(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    return list(range(args.seed, args.seed + args.episodes))


def _load_option_plan(args: argparse.Namespace) -> dict[str, list[str]]:
    if not args.option_plan_json:
        return {}
    data = json.loads(args.option_plan_json.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("--option-plan-json must contain an object")
    plan: dict[str, list[str]] = {}
    for task, commands in data.items():
        if task not in ALLOWED_TASKS:
            raise SystemExit(f"unknown task in option plan: {task}")
        if not isinstance(commands, list) or not all(
            isinstance(command, str) and command.strip() for command in commands
        ):
            raise SystemExit(f"option plan for {task} must be a list of strings")
        plan[task] = [command.strip() for command in commands]
    return plan


def _commands_for_task(
    task: str, args: argparse.Namespace, option_plan: dict[str, list[str]]
) -> list[str]:
    if task in option_plan:
        return option_plan[task]
    if args.option_command:
        return [args.option_command]
    return [_default_command(task)]


def _default_command(task: str) -> str:
    return {
        "CloseFridge": "Close the fridge door.",
        "OpenDrawer": "Open the right drawer.",
        "OpenCabinet": "Open the cabinet door.",
        "TurnOnMicrowave": "Press the start button on the microwave.",
        "TurnOffStove": "Turn off the stove.",
    }.get(task, task)


def _last_failure_mode(option_results: list[dict[str, Any]]) -> str | None:
    for result in reversed(option_results):
        failure_mode = result.get("failure_mode")
        if failure_mode:
            return str(failure_mode)
    return None


if __name__ == "__main__":
    main()
