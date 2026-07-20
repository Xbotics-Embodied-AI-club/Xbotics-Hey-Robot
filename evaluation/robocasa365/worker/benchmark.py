"""Administrative multi-episode RoboCasa benchmark entry point.

This is intentionally separate from the agent-facing ModelService worker: an
agent may request one atomic rollout, while reproducibility evaluation is an
operator-controlled batch job.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    from rollout import (
        ALLOWED_TASKS,
        DEFAULT_POLICY,
        DEFAULT_SPLIT,
        _versions,
        evaluation_rename_map,
    )
except ModuleNotFoundError:
    from evaluation.robocasa365.worker.rollout import (
        ALLOWED_TASKS,
        DEFAULT_POLICY,
        DEFAULT_SPLIT,
        _versions,
        evaluation_rename_map,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a reproducible RoboCasa benchmark"
    )
    parser.add_argument("--task", default="CloseFridge", choices=sorted(ALLOWED_TASKS))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--policy-path", default=os.environ.get("ROBOCASA_POLICY", DEFAULT_POLICY)
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _command(args: argparse.Namespace) -> list[str]:
    command = [
        os.environ.get("ROBOCASA_EVAL_BINARY", "lerobot-eval"),
        f"--policy.path={args.policy_path}",
        "--env.type=robocasa",
        f"--env.task={args.task}",
        f"--env.split={DEFAULT_SPLIT}",
        "--env.obj_registries=[lightwheel]",
        "--eval.batch_size=1",
        f"--eval.n_episodes={args.episodes}",
        "--eval.use_async_envs=false",
        f"--seed={args.seed}",
        "--policy.device=cuda",
        f"--output_dir={args.output_dir}",
    ]
    rename_map = evaluation_rename_map()
    if rename_map:
        command.append(f"--rename_map={json.dumps(rename_map, separators=(',', ':'))}")
    return command


def _gpu_info() -> dict[str, Any]:
    try:
        import torch

        return {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
        }
    except Exception as exc:  # pragma: no cover - defensive provenance collection
        return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    args = _parser().parse_args()
    if args.episodes != 20:
        raise SystemExit(
            "administrative RoboCasa benchmark requires exactly --episodes 20"
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    command = _command(args)
    provenance = {
        "task": args.task,
        "split": DEFAULT_SPLIT,
        "episodes": args.episodes,
        "seeds": list(range(args.seed, args.seed + args.episodes)),
        "policy_path": args.policy_path,
        "command": command,
        "versions": _versions(),
        "gpu": _gpu_info(),
        "started_at_unix": time.time(),
    }
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (
        (output_dir / "stdout.log").open("w", encoding="utf-8") as stdout,
        (output_dir / "stderr.log").open("w", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(  # noqa: S603 - command is constructed from allowlisted args
            command, stdout=stdout, stderr=stderr, check=False
        )
    provenance["finished_at_unix"] = time.time()
    provenance["return_code"] = completed.returncode
    eval_info = output_dir / "eval_info.json"
    if eval_info.is_file():
        provenance["eval_info"] = json.loads(eval_info.read_text(encoding="utf-8"))
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
