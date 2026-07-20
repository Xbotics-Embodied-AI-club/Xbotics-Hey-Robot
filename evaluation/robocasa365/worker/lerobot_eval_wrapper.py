#!/usr/bin/env python3
"""Register checkpoint-specific processors, then run the upstream evaluator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from .policy_probe import _load_raw_config, register_policy_processors
except ImportError:  # Executed as a standalone script in the worker image.
    from policy_probe import _load_raw_config, register_policy_processors


def _policy_path(argv: list[str]) -> str | None:
    for index, argument in enumerate(argv):
        if argument.startswith("--policy.path="):
            return argument.split("=", 1)[1]
        if argument == "--policy.path" and index + 1 < len(argv):
            return argv[index + 1]
    return None


def _has_rename_map(argv: list[str]) -> bool:
    return any(
        argument == "--rename_map" or argument.startswith("--rename_map=")
        for argument in argv
    )


def _checkpoint_rename_map(policy_path: str) -> dict[str, str]:
    path = Path(policy_path).expanduser()
    if path.is_dir():
        processor_path = path / "policy_preprocessor.json"
    else:
        from huggingface_hub import hf_hub_download

        try:
            processor_path = Path(
                hf_hub_download(policy_path, "policy_preprocessor.json")
            )
        except Exception:
            return {}

    if not processor_path.is_file():
        return {}
    try:
        processor = json.loads(processor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    for step in processor.get("steps", []):
        if step.get("registry_name") != "rename_observations_processor":
            continue
        rename_map = step.get("config", {}).get("rename_map", {})
        if isinstance(rename_map, dict) and all(
            isinstance(source, str) and isinstance(target, str)
            for source, target in rename_map.items()
        ):
            return rename_map
    return {}


def main() -> None:
    arguments = sys.argv[1:]
    policy_path = _policy_path(arguments)
    if policy_path:
        config, _ = _load_raw_config(policy_path)
        policy_type = config.get("type")
        if isinstance(policy_type, str):
            register_policy_processors(policy_type)
        if not _has_rename_map(arguments):
            rename_map = _checkpoint_rename_map(policy_path)
            if rename_map:
                sys.argv.append(
                    "--rename_map=" + json.dumps(rename_map, separators=(",", ":"))
                )

    from lerobot.scripts.lerobot_eval import main as lerobot_eval_main

    lerobot_eval_main()


if __name__ == "__main__":
    main()
