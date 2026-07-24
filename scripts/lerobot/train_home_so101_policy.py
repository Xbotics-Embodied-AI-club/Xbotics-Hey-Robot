from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch LeRobot training for the XLeRobot home SO101 dataset."
    )
    parser.add_argument(
        "--config-path",
        default="configs/examples/smolvla_home_so101.yaml",
        help="LeRobot training YAML.",
    )
    parser.add_argument(
        "--train-script",
        default=None,
        help=(
            "Training script to run. Defaults to D:/agent_robot/lerobot-mujoco-tutorial/train_model.py "
            "when present, otherwise falls back to python -m lerobot.scripts.train."
        ),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config_path)
    if not config_path.is_file():
        raise SystemExit(f"training config not found: {config_path}")

    train_script = args.train_script
    if train_script is None:
        candidate = Path("D:/agent_robot/lerobot-mujoco-tutorial/train_model.py")
        train_script = str(candidate) if candidate.is_file() else ""

    if train_script:
        command = [
            str(args.python),
            str(Path(train_script)),
            "--config_path",
            str(config_path),
        ]
    else:
        command = [
            str(args.python),
            "-m",
            "lerobot.scripts.train",
            "--config_path",
            str(config_path),
        ]

    print("Running:", " ".join(command))
    if args.dry_run:
        return
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(subprocess.call(command, env=env))  # noqa: S603


if __name__ == "__main__":
    main()
