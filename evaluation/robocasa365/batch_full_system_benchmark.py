"""Run manifest-selected RoboCasa trials through the single full-system entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from evaluation.robocasa365.full_system_benchmark import run_trial
from hey_robot.robot_runtime.robocasa_remote.contract import load_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RoboCasa365 batch full-system benchmark"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/evaluation/robocasa365.tasks.yaml"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/robocasa365.agent.yaml"),
    )
    parser.add_argument("--suite", action="append", default=[])
    parser.add_argument(
        "--condition", action="append", choices=("b0", "b1", "b2"), default=[]
    )
    parser.add_argument("--seeds", default="1000")
    parser.add_argument(
        "--objective-template",
        help=(
            "Optional paraphrase template. Omit it to use each live environment's "
            "canonical RoboCasa language instruction."
        ),
    )
    parser.add_argument("--agent-url", default="http://127.0.0.1:8080/turn")
    parser.add_argument("--runtime-target", default="grpc://127.0.0.1:9092")
    parser.add_argument(
        "--credentials-file",
        type=Path,
        default=Path("runtime/robocasa365.agent/robocasa.credentials.json"),
    )
    parser.add_argument("--poll-sec", type=float, default=1.0)
    parser.add_argument("--timeout-sec", type=float, default=7200.0)
    return parser


async def run_batch(args: argparse.Namespace) -> dict[str, object]:
    manifest = load_manifest(args.manifest)
    suites = args.suite or sorted(manifest["suites"])
    invalid = sorted(set(suites) - set(manifest["suites"]))
    if invalid:
        raise ValueError(f"unknown manifest suites: {invalid}")
    conditions = args.condition or ["b1"]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    args.output_root.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, object]] = []
    for suite in suites:
        for task in manifest["suites"][suite]:
            for seed in seeds:
                for condition in conditions:
                    child = (
                        args.output_root
                        / "trials"
                        / f"{suite}-{condition}-{task}-{seed}"
                    )
                    trial_args = argparse.Namespace(
                        task=task,
                        seed=seed,
                        objective=(
                            args.objective_template.format(task=task)
                            if args.objective_template
                            else None
                        ),
                        condition=condition,
                        manifest=args.manifest,
                        config=args.config,
                        agent_url=args.agent_url,
                        runtime_target=args.runtime_target,
                        credentials_file=args.credentials_file,
                        output_dir=child,
                        poll_sec=args.poll_sec,
                        timeout_sec=args.timeout_sec,
                    )
                    result = await run_trial(trial_args)
                    results.append({"suite": suite, **result})
    summary = {
        "manifest": manifest,
        "count": len(results),
        "official_successes": sum(bool(item["official_success"]) for item in results),
        "false_completions": sum(bool(item["false_completion"]) for item in results),
        "trials": results,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    summary = asyncio.run(run_batch(_parser().parse_args()))
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
